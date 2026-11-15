"""Radius-normalized sparsemax Fisher retrieval with temporal G/RFF readout."""
import math
import time
import weakref

import torch

from .lda import fit_discriminant
from .precision import ieee_fp32
from .fast_neighbors import _BankBounds, _maximum_norm, _error_bounds
from .sparse_qp import sparsemax_support, sparsemax


def fixed_projection(inputs, matrix):
    """Project every row with the same 32-row IEEE GEMM, including partial batches."""
    rows, weight = inputs.reshape(-1, inputs.shape[-1]).float(), matrix.float().contiguous()
    block = rows.new_zeros(32, rows.shape[1])
    output = rows.new_empty(len(rows), weight.shape[1])
    with ieee_fp32(restore_tf32_first=True):
        for start in range(0, len(rows), 32):
            count = min(32, len(rows) - start)
            block.zero_()
            block[:count].copy_(rows[start:start + count])
            output[start:start + count] = (block @ weight)[:count]
    return output.reshape(*inputs.shape[:-1], weight.shape[1])


class Selector:
    def __init__(self, policy, payload=None):
        self.policy, self.payload = weakref.proxy(policy), payload
        self.config = policy.config['g']['lda']
        from .pipeline import validate_method_config
        validate_method_config(policy.config)
        if self.config.get('teacher_storage', 'device') not in ('device', 'host'):
            raise ValueError('Fisher teacher_storage must be device or host')
        self.banks = None
        self.fields = ('omega', 'phase')
        if payload is not None:
            if payload.get('version') != 1:
                raise ValueError('unsupported Fisher artifact')
            validate_method_config(dict(g=dict(lookup='lda', lda=payload.get('config', {}))))
            if (payload.get('diagnostics', {}).get('teacher_kind') != 'sparsemax'
                    or any(name in payload for name in ('prototypes', 'bias'))
                    or not math.isfinite(payload.get('support_scale', 0))
                    or payload.get('support_scale', 0) <= 0):
                raise ValueError('refit artifact with radius-normalized sparsemax Fisher coverage')
            for name in self.fields:
                setattr(self, name, payload[name].to(policy.device))

    def prepare(self):
        if self.banks is None:
            p = self.policy
            if p.pf is None:
                p._project_hankel()
            self.banks = [p.pf.float(), *[bank.float() for _, bank, _, _, _
                                        in p._interpolated_keys()]]
            bound = max(_maximum_norm(bank.double()) for bank in self.banks)
            self.bounds = _BankBounds(bound, bound, 0.)
        return self.banks

    def release(self):
        self.banks = None
        self._density_search = None

    def _distance(self, keys, target_rows=None, supports=None):
        p = self.policy
        banks = self.prepare()

        def distance_to(bank):
            if supports is not None:
                return (keys[:, None].double() - bank[supports].double()).square().sum(-1)
            with ieee_fp32(restore_tf32_first=True):
                return torch.cdist(keys, bank).square()

        distance = distance_to(banks[0])
        rates = torch.zeros_like(distance, dtype=torch.uint8)
        second = torch.full_like(distance, torch.inf) if supports is None else None
        for index, (rate, _, _, first, valid) in enumerate(p._interpolated_keys(), 1):
            warped = distance_to(banks[index])
            starts = p.hankel_starts
            if supports is not None:
                first, valid, starts = first[supports], valid[supports], starts[supports]
            warped.masked_fill_(~valid, torch.inf)
            if target_rows is not None:
                warped.masked_fill_(p._rate_target_overlap(
                    first, starts, rate, target_rows), torch.inf)
            better = warped < distance
            if second is not None:
                torch.minimum(second, torch.maximum(distance, warped), out=second)
            rates[better] = index
            torch.minimum(distance, warped, out=distance)
        if second is not None and len(banks) > 1:
            # Certify the same winner as direct FP64 evaluation of these fixed
            # keys. Refine only ambiguous pairs, with bounded pair-by-feature scratch.
            error, _ = _error_bounds(keys.double(), keys, self.bounds)
            ambiguous = (second.sub_(distance) <= 2 * error[:, None]).flatten()
            for start in range(0, ambiguous.numel(), 2**20):
                positions = torch.nonzero(ambiguous[start:start + 2**20]).flatten() + start
                if not len(positions):
                    continue
                for pair in positions.split(4096):
                    selected, col = pair // distance.shape[1], pair % distance.shape[1]
                    exact, choice = self._distance(
                        keys[selected], None if target_rows is None else target_rows[selected], col[:, None])
                    distance[selected, col], rates[selected, col] = exact[:, 0].to(distance.dtype), choice[:, 0]
        return distance, rates

    def features(self, queries):
        keys = fixed_projection(queries, self.omega)
        return (keys.double() + self.phase.double()).cos() * math.sqrt(2 / len(self.phase))

    @torch.no_grad()
    def fit(self):
        started = time.perf_counter()
        p, c = self.policy, self.config
        teacher_storage = c.get('teacher_storage', 'device')
        if torch.device(p.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(p.device)
        generator = torch.Generator(device=p.device).manual_seed(int(c['seed']))
        columns = torch.arange(len(p.hankel_starts), device=p.device)
        anchors = columns[torch.randperm(len(columns), device=p.device, generator=generator)[:int(c['anchors'])]]
        history = p._past_columns(anchors)
        distance = torch.pdist(history[:512].float())
        positive = distance[distance > 0]
        sigma = (positive.median() if len(positive) else distance.new_tensor(1.)) * float(c['sigma_scale'])
        width = int(c['features'])
        self.omega = torch.randn(history.shape[1], width, device=p.device, generator=generator)/sigma
        self.phase = torch.rand(width, device=p.device, generator=generator)*math.tau
        full = bool(c.get('fit_all_histories', False))
        if full:
            from .full_fisher import fit_all_histories
            fitted = fit_all_histories(p)
            fitted.update(teacher_storage='sparse_host', teacher_kind='sparsemax', teacher_distance_scale='radius')
        else:
            future = p._action_neighborhood_features()
            _, near_distance = p._metric_topk(anchors, columns, future,
                min(4*p.g_supports, len(columns)-1), 32)
            radius = near_distance[:, -1].double().clamp_min(1e-12)
            future = future.double()
            targets = (torch.empty((len(anchors), len(columns)), dtype=torch.float64, device='cpu')
                       if teacher_storage == 'host' else [])
            support_min, support_max, support_sum = len(columns), 0, 0
            for offset, batch in enumerate(anchors.split(32)):
                sl = slice(offset*32, offset*32+len(batch))
                distance = torch.cdist(future[batch], future).square()
                overlap = ((p.hankel_trajectory[batch, None] == p.hankel_trajectory[None])
                    & ((p.hankel_starts[batch, None]-p.hankel_starts[None]).abs() < p.history+p.horizon))
                logits = -distance/radius[sl, None]
                distribution = sparsemax(logits.masked_fill(overlap, -torch.inf))
                positive = (distribution > 0).sum(1)
                support_min = min(support_min, int(positive.min()))
                support_max = max(support_max, int(positive.max()))
                support_sum += int(positive.sum())
                if teacher_storage == 'host':
                    targets[sl].copy_(distribution.cpu())
                else:
                    targets.append(distribution)
            if teacher_storage == 'device':
                targets = torch.cat(targets)
            fitted = fit_discriminant(self.features(history), targets, float(c['shrinkage']), fisher_rank=c['fisher_rank'])
            fitted['teacher_storage'] = teacher_storage
            fitted['teacher_kind'] = 'sparsemax'
            fitted['teacher_distance_scale'] = 'radius'
            fitted['teacher_positive_support'] = dict(minimum=support_min,
                mean=support_sum/len(anchors), maximum=support_max)
        p.g_lookup_matrix = fitted.pop('transform').to(device=p.device, dtype=p.dtype)
        p.g_lookup_scale = 1.
        self.payload = dict(version=1, **{name: getattr(self, name).cpu()
            for name in self.fields}, config=c, diagnostics=fitted)
        self.payload['support_scale'] = 1.
        if not full:
            del targets, future
        del history
        p._project_hankel()
        fitted.update(self._calibrate_support(len(anchors)))
        print(f'[fit] LDA {fitted}', flush=True)
        memory = (dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(p.device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(p.device))
                  if torch.device(p.device).type == 'cuda' else {})
        return dict(name='lda', seconds=time.perf_counter()-started, anchors=len(columns) if full else len(anchors), **fitted, **memory)

    def _calibrate_support(self, fitted_count):
        p = self.policy
        generator = torch.Generator(device=p.device).manual_seed(int(self.config['seed']))
        anchors = torch.randperm(len(p.hankel_starts), device=p.device, generator=generator)[fitted_count:fitted_count+64]
        if not len(anchors):
            raise ValueError('adaptive calibration needs histories outside the fitting anchors')
        rows = p.hankel_starts[anchors]+p.history
        return self._calibrate_coverage(anchors, rows)

    def _calibrate_coverage(self, anchors, rows):
        p = self.policy
        future = p._action_neighborhood_features().double()
        scales = [.25*2**(-i/4) for i in range(9)]
        mass = {scale: 0. for scale in scales}
        reference = 0.
        for batch, query_rows in zip(anchors.split(16), rows.split(16)):
            distance = torch.cdist(future[batch], future).square()
            distance.masked_fill_(p.hankel_trajectory[batch, None] == p.hankel_trajectory[None], torch.inf)
            nearest = distance.topk(min(4*p.g_supports, distance.shape[1]), largest=False).values
            radius = nearest.masked_fill(~torch.isfinite(nearest), 0).amax(1).clamp_min(1e-12)
            teacher = sparsemax(-distance/radius[:, None])
            queries, demos = p._queries(query_rows), p.trajectory_of[query_rows]
            # The nine scales share histories, exclusions, distances and density
            # normalization. Retain each identical retrieval request, rather
            # than projecting and searching the temporal bank nine times.
            topk = self._adaptive_topk(queries, query_rows, demos, cache=True)
            columns, _ = topk(p.g_supports, None)
            reference += float(teacher.gather(1, columns).sum())/len(rows)
            for scale in scales:
                self.payload['support_scale'] = scale
                selected = self._adaptive_select(topk)
                # One device-to-host transfer, preserving the original Python
                # sum order and each row's original reduction arithmetic.
                covered = torch.stack([teacher[i, col].sum()
                                       for i, (col, _) in enumerate(selected)])
                mass[scale] += sum(covered.tolist())/len(rows)
        feasible = [scale for scale in scales if mass[scale] >= reference]
        if not feasible:
            raise ValueError('coverage calibration grid cannot recover reference teacher mass')
        self.payload['support_scale'] = max(feasible)
        return dict(support_scale=max(feasible), support_calibration_histories=len(rows),
                    support_calibration_mass=mass, support_reference_mass=reference)

    def _coverage_density_topk(self, queries, count, target_rows, target_demos, allowed):
        # The default fused route retains full FP64 GEMM/topk shapes and bits.
        # Mixed retrieval instead uses FP32 screening and candidate-only FP64
        # refinement; it preserves the metric but may change rounding/tie order.
        mode = getattr(self, '_density_search_mode',
                       'fused' if queries.is_cuda else 'reference')
        from .bpc import Policy
        canonical = ('_hankel_topk', '_rate_target_overlap',
                     '_hankel_exclusion', '_exclude_hankel')
        if any(getattr(getattr(self.policy, name, None), '__func__', None)
               is not getattr(Policy, name) for name in canonical):
            return self.policy._hankel_topk(queries, count, target_rows, target_demos, allowed)
        if mode in ('fused', 'mixed'):
            if mode == 'mixed':
                from .mixed_density_search import MixedDensitySearch as Search
            else:
                from .fused_density_search import FusedDensitySearch as Search
            search = getattr(self, '_density_search', None)
            if not isinstance(search, Search):
                search = self._density_search = Search()
            result = search.select(self.policy, queries, count,
                                   target_rows, target_demos, allowed)
            if result is not None:
                return result
        return self.policy._hankel_topk(queries, count, target_rows, target_demos, allowed)

    def _adaptive_topk(self, queries, target_rows=None, target_demos=None,
                       allowed=None, cache=False):
        """Share exact density prefixes without changing tie or subset rules.

        A larger top-k prefix can order tied columns differently. The default
        exact route reuses only strictly ordered prefixes. Mixed retrieval
        permits changed tie order and reuses its sorted density prefix too.
        Calibration may also reuse identical (count, query-row subset) requests
        across scales. In particular, no subset query is replaced with a new
        batch shape when the density prefix cannot safely answer it.
        """
        p = self.policy
        prefix = None
        columns, values, _ = self._coverage_density_topk(
            queries, min(4*p.g_supports, len(p.hankel_starts)),
            target_rows, target_demos, allowed)
        farthest = values.masked_fill(~torch.isfinite(values), 0).amax(1)
        spread = (farthest.square()-values[:, 0].square()).clamp_min(1e-12)
        if (getattr(self, '_density_search_mode', None) == 'mixed'
                or bool((values[:, 1:] > values[:, :-1]).all())):
            prefix = columns, -values.square()/spread[:, None]
        saved = {}

        def topk(count, rows):
            if prefix is not None and count <= prefix[0].shape[1]:
                return tuple(x[:, :count] if rows is None else x[rows, :count]
                             for x in prefix)
            key = (count, None if rows is None else tuple(rows.tolist())) if cache else None
            if cache and key in saved:
                return saved[key]
            batch = [x if rows is None or x is None else x[rows]
                     for x in (queries, target_rows, target_demos, allowed)]
            search = (self._coverage_density_topk
                      if getattr(self, '_density_search_mode', None) == 'mixed'
                      else p._hankel_topk)
            columns, values, _ = search(batch[0], count, *batch[1:])
            scores = -values.square()
            scores = scores/(spread if rows is None else spread[rows])[:, None]
            result = columns, scores
            if cache:
                saved[key] = result
            return result

        return topk

    def _adaptive_select(self, topk):
        return [(columns, scores[0]-scores) for columns, scores in
                sparsemax_support(topk, len(self.policy.hankel_starts),
                                  self.payload['support_scale'])]

    @torch.no_grad()
    def select(self, queries, target_rows=None, target_demos=None, allowed=None):
        if self.payload is None or 'support_scale' not in self.payload:
            raise ValueError('fit Fisher coverage before selecting supports')
        if self.policy.pf is None:
            self.policy._project_hankel()
        return self._adaptive_select(self._adaptive_topk(
            queries, target_rows, target_demos, allowed))

    def rate_trace(self, queries, supports, target_rows=None, *, _choices=None):
        p = self.policy
        if len(p.g_lookup_rates) == 1:
            return {}
        keys = p._g_lookup_projection(queries.to(p.g_lookup_matrix.dtype)).float()
        _, rates = self._distance(keys, target_rows, supports)
        choices = (queries.new_tensor([1., *[rate for rate, *_ in p._interpolated_keys()]])
                   if _choices is None else _choices)
        ordinary = (keys[:, None] - self.banks[0][supports]).square().sum(-1)
        return {"hankel_rates": choices[rates.long()],
                "hankel_unwarped_distance": ordinary.sqrt().to(queries.dtype)}
