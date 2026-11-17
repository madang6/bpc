"""Bounded, exact-shape CUDA graphs for independent adaptive readout groups.

Retrieval and grouping stay outside capture. Every graph captures and
replays on the SAME retained stream, so captured BLAS workspaces sharing a stream
are serialized. Multirow solves retain their original backend outside capture.
"""
from collections import OrderedDict
import threading
import weakref

import torch

from .bpc import Policy
from .lda_selector import Selector
from .retrieval import FisherLookup
from .readout_bookkeeping import scatter_readout
from .readout_contracts import supports_readout


MAX_GRAPHS = 64
MAX_BUILDS = 128
CACHE_BYTES = 1024 * 1024**2
_CAPTURE_LOCK = threading.Lock()
# The same declaration supplies captured state and its mutation signature.
# Tensor values stay shared; containers are snapshotted without unrelated policy state.
_STATE_FIELDS = (
    'method', 'g_supports', 'history', 'horizon', 'state_dim', 'action_dim', 'proprio_dim',
    'samples', 'g_state_offset', 'g_lookup', 'g_lookup_rates', 'g_align_history',
    'g_align_future', 'g_rate_scaled_action_dims', 'g_native', 'g_constraint',
    'g_solver', 'g_weights', 'g_l1', 'fixed_shape_readout', 'residual_action_source',
    'rff_dtype', 'rff_dim', 'rff_scale', 'dtype', 'device',
    'state', 'action', 'uf', 'pf', 'pf_norm', 'hankel_starts', 'hankel_trajectory',
    'trajectory_starts', 'action_mean', 'action_scale', 'g_lookup_matrix',
    'rff_bank_weight', 'rff_query_weight', 'rff_bias', 'residual_weights', '_rate_keys',
)
_SELECTOR_FIELDS = ('omega', 'phase', 'banks')


def _layout(t):
    if (not isinstance(t, torch.Tensor) or t.layout != torch.strided
            or t.is_neg() or t.is_conj() or t.requires_grad):
        raise ValueError('unsupported graph tensor')
    return tuple(t.shape), tuple(t.stride()), t.dtype, t.device


def _tensor_key(t):
    # Unversioned inference tensors deliberately use the eager implementation.
    return id(t), t._version, t.data_ptr(), t.storage_offset(), _layout(t)


def _freeze(value):
    if isinstance(value, torch.Tensor):
        return _tensor_key(value)
    if isinstance(value, dict):
        return tuple((name, _freeze(item)) for name, item in value.items())
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_freeze(item) for item in value)
    return type(value), value


def _state(p):
    values = {name: getattr(p, name, None) for name in _STATE_FIELDS}
    values['selector'] = {name: getattr(p.support_selector, name) for name in _SELECTOR_FIELDS}
    return values


def _snapshot(value):
    if isinstance(value, dict):
        return {name: _snapshot(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    return value


def _fingerprint(p):
    backend = (torch.backends.cuda.preferred_linalg_library(),
               torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision(),
               torch.are_deterministic_algorithms_enabled())
    return _freeze(_state(p)), backend, (MAX_GRAPHS, CACHE_BYTES, MAX_BUILDS)


def _eligible(p, actions, states):
    if getattr(p, 'method', None) == 'bpc_nnc':
        return False
    from .ragged_readout import _eligible as temporal_eligible
    return (getattr(p, 'concurrent_readout', False) and actions.is_cuda
            and p.fixed_shape_readout and not p.g_l1 and p.g_weights == 'optimized'
            and temporal_eligible(p, actions, states)
            and not torch.cuda.is_current_stream_capturing()
            and torch.backends.cuda.preferred_linalg_library() != torch._C._LinalgBackend.Magma
            and supports_readout(p, online=True, captured=True))


class _SelectorState(Selector):
    """Only the fitted features and prepared banks used by temporal readout."""
    def __init__(self, policy, values):
        self.policy = weakref.proxy(policy)
        self.__dict__.update(values)


class _Core(Policy):
    """Explicit captured state; inherited arithmetic uses class-defined adapters."""
    def __init__(self, p):
        values = _snapshot(_state(p))
        selector = values.pop('selector')
        self.__dict__.update(values)
        self.support_selector = _SelectorState(self, selector)
        self.retrieval = FisherLookup()
        self._temporal_static_cache = {}
        self.choices = self.action.new_tensor([1., *[r for r, *_ in self._rate_keys]])
        self.flags = []
        # Check all native residual indices before speculative feature gathers.
        width, demos = len(self.hankel_starts), len(self.trajectory_starts)
        if (not width or not demos or self.hankel_starts.dtype != torch.long
                or self.hankel_trajectory.dtype != torch.long
                or self.hankel_trajectory.shape != (width,)
                or self.trajectory_starts.dtype != torch.long
                or self.trajectory_starts.ndim != 1
                or self.state.ndim != 2 or self.action.ndim != 2
                or self.state.shape[1] != self.state_dim or self.action.shape[1] != self.action_dim
                or self.state.dtype != self.dtype or self.action.dtype != self.dtype
                or not self.state.is_contiguous() or not self.action.is_contiguous()
                or any(len(bank) != width for bank in self.support_selector.banks)):
            raise ValueError('unsupported graph bank layout')
        trajectories = self.hankel_trajectory
        endpoints = self._support_endpoints(torch.arange(width, device=self.device))
        if not bool(((trajectories >= -demos) & (trajectories < demos)
                     & (endpoints >= 0) & (endpoints < len(self.state))
                     & (endpoints < len(self.action))).all()):
            raise ValueError('invalid native metadata')

    def _solve_affine(self, design, query):
        return super()._solve_affine(design, query, _deferred_errors=self.flags)

    def _rate_trace(self, queries, supports, target_rows=None):
        return self.support_selector.rate_trace(queries, supports, target_rows,
                                                _choices=self.choices)

    def _checked(self, value, before):
        if value is None or len(self.flags) != before+1:
            raise ValueError('temporal gather cannot be captured')
        return torch.where(self.flags[-1], torch.zeros_like(value), value)

    def _reconstruction_columns(self, supports, queries, dtype=None, target_rows=None, rates=None):
        from .fused_temporal_history import reconstruction_columns
        if dtype is not None or target_rows is not None:
            raise ValueError('unsupported captured history call')
        before = len(self.flags)
        return self._checked(reconstruction_columns(self, supports, rates,
                                                    _errors=self.flags), before)

    def _future_columns(self, supports, queries=None, target_rows=None, rates=None, states=False):
        from .fused_temporal_future import future_columns
        if target_rows is not None:
            raise ValueError('unsupported captured future call')
        before = len(self.flags)
        return self._checked(future_columns(self, supports, rates, states,
                                           _errors=self.flags), before)

    def __call__(self, actions, states, columns, distances):
        width = len(self.hankel_starts)
        self.flags = [((columns < -width) | (columns >= width)).any()]
        safe = columns.clamp(-width, width-1)
        result = self.direct_actions(actions, states, selected=(safe, distances))
        return result, torch.stack(self.flags)

    def constants(self):
        values = [self.choices]
        for entry in self._temporal_static_cache.values():
            values.extend(entry[2])
        return tuple(_tensor_key(t) for t in values)


class _Graph:
    def __init__(self, core, values, stream):
        self.stream = stream
        self.inputs = [torch.empty_strided(t.shape, t.stride(), device=t.device,
                                          dtype=t.dtype).copy_(t) for t in values]
        self.done = None
        current = torch.cuda.current_stream(values[0].device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                core(*self.inputs)
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output, self.flags = core(*self.inputs)
        core.flags = []
        current.wait_stream(stream)

    def launch(self, values):
        current = torch.cuda.current_stream(values[0].device)
        if self.done is not None:
            current.wait_event(self.done)
        for target, value in zip(self.inputs, values):
            target.copy_(value)
        self.stream.wait_stream(current)
        # Never replay on another stream: captured BLAS workspaces are tied to
        # capture-stream ownership, even when graph allocation pools differ.
        with torch.cuda.stream(self.stream):
            self.graph.replay()

    def join(self):
        torch.cuda.current_stream(self.inputs[0].device).wait_stream(self.stream)

    def consumed(self):
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(self.inputs[0].device))

    def close(self):
        if self.done is not None:
            self.done.synchronize()
        self.graph.reset()


class _Cache:
    def __init__(self, p):
        self.policy = weakref.ref(p)
        self.lock = threading.Lock()
        self.fingerprint = _fingerprint(p)
        self.core = _Core(p)
        self.streams = [torch.cuda.Stream(device=p.device) for _ in range(2)]
        if self.streams[0].cuda_stream == self.streams[1].cuda_stream:
            self.streams = self.streams[:1]
        self.entries, self.seen, self.rejected = OrderedDict(), OrderedDict(), set()
        self.bytes = self.builds = self.attempts = self.evictions = 0
        self.constants = None
        self.last = {}

    def clear(self):
        for graph, _ in self.entries.values():
            graph.close()
        self.entries.clear()
        self.bytes = 0

    def get(self, values, protected):
        key = tuple(_layout(t) for t in values)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][0]
        if key in self.rejected or self.attempts >= MAX_BUILDS:
            return None
        self.seen[key] = self.seen.get(key, 0)+1
        self.seen.move_to_end(key)
        if len(self.seen) > 2*MAX_GRAPHS:
            old, _ = self.seen.popitem(last=False)
            self.rejected.discard(old)
        if self.seen[key] < 2:
            return None  # Do not capture a shape on its first observation.
        p, count = self.core, values[2].shape[1]
        estimate = 16*count*(p.rff_dim+p.history*(p.action_dim+p.state_dim)+count)+16*1024**2
        if estimate > CACHE_BYTES or MAX_GRAPHS < 1:
            self.rejected.add(key)
            return None
        while self.entries and (len(self.entries) >= MAX_GRAPHS or self.bytes+estimate > CACHE_BYTES):
            victim = next((k for k, (g, _) in self.entries.items() if id(g) not in protected), None)
            if victim is None:
                return None
            old, size = self.entries.pop(victim)
            old.close()
            self.rejected.add(victim)
            self.bytes -= size
            self.evictions += 1
        if not _CAPTURE_LOCK.acquire(blocking=False):
            return None
        graph = None
        try:
            free, _ = torch.cuda.mem_get_info(values[0].device)
            if free < estimate+512*1024**2:
                return None
            allocated = torch.cuda.memory_allocated(values[0].device)
            reserved = torch.cuda.memory_reserved(values[0].device)
            self.attempts += 1
            stream = self.streams[self.builds % len(self.streams)]
            graph = _Graph(self.core, values, stream)
            # Admit a graph only after an actual replay matches the unchanged
            # selected path byte for byte on this device/backend and shape.
            graph.launch(values)
            graph.join()
            if bool(graph.flags.any()):
                graph.consumed()
                graph.close()
                self.rejected.add(key)
                return None
            reference = self.policy().direct_actions(values[0], values[1],
                                                     selected=(values[2], values[3]))
            checks = []
            def compare(left, right):
                if torch.is_tensor(left):
                    if left.shape != right.shape or left.dtype != right.dtype:
                        raise ValueError('graph result layout differs')
                    checks.append((left.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
                                   == right.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)).all())
                elif isinstance(left, dict):
                    if left.keys() != right.keys():
                        raise ValueError('graph trace keys differ')
                    for name in left:
                        compare(left[name], right[name])
                else:
                    if len(left) != len(right):
                        raise ValueError('graph result length differs')
                    for a, b in zip(left, right):
                        compare(a, b)
            compare(graph.output, reference)
            exact = bool(torch.stack(checks).all())
            graph.consumed()
            if not exact:
                graph.close()
                self.rejected.add(key)
                return None
            cost = max(0, torch.cuda.memory_allocated(values[0].device)-allocated,
                       torch.cuda.memory_reserved(values[0].device)-reserved)
            if self.bytes+cost > CACHE_BYTES:
                graph.close()
                self.rejected.add(key)
                return None
            self.entries[key] = graph, cost
            self.bytes += cost
            self.builds += 1
            self.constants = self.core.constants()
            return graph
        except (RuntimeError, ValueError):
            if graph is not None:
                graph.close()
            self.rejected.add(key)
            return None
        finally:
            _CAPTURE_LOCK.release()


def clear_cache(p):
    """Release graph storage and captured parameters when the route is disabled."""
    cache = getattr(p, '_concurrent_readout_cache', None)
    if cache is not None and cache.lock.acquire(blocking=False):
        try:
            cache.clear()
            p.__dict__.pop('_concurrent_readout_cache', None)
        finally:
            cache.lock.release()


@torch.no_grad()
def direct_actions(p, actions, states, selected):
    """Return owned scattered outputs, or None for the existing grouped path.

    This function runs AFTER retrieval; None must reuse the same
    selected supports, so fallback never repeats retrieval.
    """
    if not _eligible(p, actions, states):
        clear_cache(p)
        return None
    from .ragged_supports import support_groups
    from .ragged_readout import prepare_readout
    cache = None
    launched = []
    locked = False
    try:
        groups = list(support_groups(selected, actions.device))
        if not any(len(rows) == 1 for rows, _, _ in groups):
            return None
        p.support_selector.prepare()
        cache = getattr(p, '_concurrent_readout_cache', None)
        fingerprint = _fingerprint(p)
        if cache is not None:
            if not cache.lock.acquire(blocking=False):
                return None
            locked = True
            if (fingerprint != cache.fingerprint or
                    cache.constants is not None and cache.constants != cache.core.constants()):
                cache.clear()
                p.__dict__.pop('_concurrent_readout_cache', None)
                cache.lock.release()
                locked = False
                cache = None
        if cache is None:
            cache = _Cache(p)
            p._concurrent_readout_cache = cache
            cache.lock.acquire()
            locked = True
        entries, protected = [], set()
        for rows, columns, distances in groups:
            u, y = actions[rows], states[rows]
            values = (u, y, columns, distances)
            graph = None
            if (len(rows) == 1 and columns.dtype == torch.long
                    and all(t.device == actions.device for t in values)
                    and u.dtype == p.dtype and y.dtype == p.dtype
                    and distances.dtype == p.dtype):
                graph = cache.get(values, protected)
                if graph is not None:
                    protected.add(id(graph))
            entries.append((rows, columns, distances, u, y, values, graph))
        for *_, values, graph in entries:
            if graph is not None:
                graph.launch(values)
                launched.append(graph)
        cache.last = dict(groups=len(groups), graphs=len(launched), fallback=False)
        if not launched:
            return None
        for graph in launched:
            graph.join()
        if bool(torch.cat([graph.flags for graph in launched]).any()):
            cache.last['fallback'] = True
            return None
        missing = [(rows, columns, distances) for rows, columns, distances, *_, graph in entries
                   if graph is None]
        prepared = prepare_readout(p, actions, states, selected, _groups=missing)
        ready = {} if prepared is None else {columns.shape[1]: item[-1]
            for item in prepared for columns in (item[3],)}
        output, traces = actions.new_empty(len(actions), p.action_dim), {}
        for rows, columns, distances, u, y, _, graph in entries:
            kwargs = {} if columns.shape[1] not in ready else {'_prepared': ready[columns.shape[1]]}
            result = (graph.output if graph is not None else
                      p.direct_actions(u, y, selected=(columns, distances), **kwargs))
            scatter_readout(output, traces, rows, result)
        return output, traces
    except (RuntimeError, ValueError, IndexError, TypeError, AttributeError):
        # The unchanged grouped path preserves original fallback and error order.
        if cache is not None and not locked:
            clear_cache(p)
        return None
    finally:
        for graph in launched:
            graph.join()
            graph.consumed()  # Protect caller-stream copies before reuse/eviction.
        if locked:
            cache.lock.release()
