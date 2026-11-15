"""Bounded, version-checked constants for the optional temporal CUDA gathers."""
import weakref

import torch


_CACHE_LIMIT = 8


def gathered_result(result, error, shape, kind, deferred_errors):
    """Check temporal-kernel status before exposing its output."""
    if deferred_errors is not None:
        # Staged callers check every flag before consuming potentially unwritten output.
        deferred_errors.append(error.reshape(()) != 0)
    else:
        code = int(error.item())
        if code & 4:  # Preserve failure of the original initial native gather.
            return None
        if code & 1:
            raise ValueError(f'interpolated {kind} crosses a demonstration boundary')
        if code:
            return None
    return result.reshape(shape)


def _signature(value):
    try:
        version = value._version
    except RuntimeError:  # Inference-mode tensors have no mutation counter.
        return None
    return (id(value), version, value.data_ptr(), value.device, value.dtype, tuple(value.shape),
            tuple(value.stride()), value.storage_offset())


def _cached(policy, slot, configuration, inputs, build):
    signatures = tuple(_signature(value) for value in inputs)
    if any(signature is None for signature in signatures):
        return build()
    key = (configuration, signatures)
    cache = getattr(policy, '_temporal_static_cache', None)
    entry = None if cache is None else cache.get(slot)
    if entry is not None:
        previous, references, values, outputs = entry
        if (previous == key and all(reference() is value for reference, value in zip(references, inputs))
                and outputs == tuple(_signature(value) for value in values)):
            return values
    values = build()
    outputs = tuple(_signature(value) for value in values)
    if any(signature is None for signature in outputs):
        return values
    if cache is None:
        cache = {}
        policy._temporal_static_cache = cache
    if slot not in cache and len(cache) >= _CACHE_LIMIT:
        del cache[next(iter(cache))]
    # Weak input references avoid retaining discarded FP32 bank copies or old
    # trajectories; identity checks also prevent false hits if Python reuses IDs.
    cache[slot] = (key, tuple(weakref.ref(value) for value in inputs), values, outputs)
    return values


@torch.no_grad()
def temporal_constants(policy, source, configured, *, actions=True):
    """Return exactly the old rates/mean/scale/mask tensors, reusing valid ones.

    Source values are not cached. Version checks conservatively include source
    identity/layout as well as the action scaling tensors; inference-mode inputs
    or outputs use the original allocations on every call.
    """
    configured = tuple(configured)
    dimensions = tuple(policy.g_rate_scaled_action_dims) if actions else ()
    rates = tuple(policy.g_lookup_rates)
    # Equal Python numbers can have different indexing semantics: bools form a
    # mask while ints select positions. Preserve runtime configuration changes,
    # including the original indexing error for a malformed Boolean mask.
    dimension_key = tuple((type(value), value) for value in dimensions)
    configuration = (configured, rates, dimension_key, actions)
    slot = ('constants', actions, source.device, source.dtype, source.shape[1])
    inputs = (source, policy.action_mean, policy.action_scale) if actions else (source,)

    def build():
        rate_tensor = torch.tensor(configured, device=source.device, dtype=torch.float64)
        if actions:
            mean = policy.action_mean.to(dtype=source.dtype).contiguous()
            scale = policy.action_scale.to(dtype=source.dtype).contiguous()
            scaled = torch.zeros(source.shape[1], device=source.device, dtype=torch.bool)
            if dimensions:
                scaled[list(dimensions)] = True
        else:
            mean = scale = source.new_empty(0)
            scaled = torch.empty(0, device=source.device, dtype=torch.bool)
        return rate_tensor, mean, scale, scaled

    return _cached(policy, slot, configuration, inputs, build)


@torch.no_grad()
def trajectory_stops(policy):
    """Reuse the standard starts/samples-derived stops; retain private overrides."""
    from .bpc import Policy
    method = policy._trajectory_stops
    if getattr(method, '__func__', None) is not Policy._trajectory_stops:
        return method()
    return _cached(policy, ('trajectory_stops',),
                   (policy.samples, method.__func__), (policy.trajectory_starts,),
                   lambda: (method(),))[0]
