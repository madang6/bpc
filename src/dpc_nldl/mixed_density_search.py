"""FP32 temporal screening with selective FP64 density/support distances.

The certificate protects completeness of each requested nearest-neighbor set,
not the reduction order or tie bits of the historical full-bank FP64 GEMM.
Ambiguous rows retain exhaustive FP64 retrieval. No support-size cap is added.
"""
import math

import torch

from .density_search import _signature, _stored_norm_error, bank_tensors, valid_inputs
from .fast_neighbors import CertifiedTopK, _BankBounds, _error_bounds
from .precision import ieee_fp32


class MixedDensitySearch:
    """Drop-in top-k search, including inference and recovery eligibility masks.

    A count requests both the density neighborhood and, when sparsemax expands
    its search, an arbitrarily large support prefix. ``max_prefix`` limits only
    speculative work; larger or uncertified requests use exhaustive retrieval.
    """

    def __init__(self, max_prefix=4096, min_batch=2):
        self.max_prefix = max_prefix
        self.min_batch = min_batch
        self.signature = None
        self.screens = None
        self.last = {}
        self.totals = dict(calls=0, rows=0, certified_rows=0,
                           fallback_rows=0, small_batch_rows=0, candidate_distances=0)

    @staticmethod
    def _rate_mask(policy, rate, first, valid, rows, columns=None):
        starts = policy.hankel_starts
        if columns is not None:
            first, valid, starts = first[columns], valid[columns], starts[columns]
        invalid = ~valid
        if rows is not None:
            invalid = invalid | policy._rate_target_overlap(first, starts, rate, rows)
        return invalid

    def _refine(self, policy, projected, candidates, rows, rates):
        # Bound the temporary gather without padding or changing requested k.
        batch, count = candidates.shape
        chunk = max(1, (16 * 1024 * 1024) // (count * projected.shape[1]))
        pieces = []
        banks = [(None, policy.pf, policy.pf_norm, None, None)] + rates
        for offset in range(0, batch, chunk):
            selected = candidates[offset:offset+chunk]
            query = projected[offset:offset+chunk]
            target = None if rows is None else rows[offset:offset+chunk]
            norm = torch.einsum('ij,ij->i', query, query)[:, None]
            distance = None
            for rate, bank, bank_norm, first, valid in banks:
                # Same mathematical expansion as the historical distance.
                # Candidate-only reductions intentionally need not share its bits.
                product = torch.einsum('bd,bkd->bk', query, bank[selected])
                value = product.mul_(-2).add_(norm).add_(bank_norm[selected]).clamp_min_(0)
                if rate is not None:
                    value.masked_fill_(self._rate_mask(
                        policy, rate, first, valid, target, selected), math.inf)
                if distance is None:
                    distance = value
                else:
                    torch.minimum(distance, value, out=distance)
            pieces.append(distance)
        return torch.cat(pieces) if len(pieces) > 1 else pieces[0]

    @torch.no_grad()
    def select(self, policy, queries, count, rows=None, demos=None, allowed=None):
        self.last = dict(route='fallback', rows=len(queries), certified=0,
                         fallback_rows=len(queries), reason='unsupported',
                         small_batch_rows=0, candidate_distances=0,
                         fp32_products=0, fused_screen=False, prefixes=[])
        bank = getattr(policy, 'pf', None)
        if not isinstance(bank, torch.Tensor) or bank.ndim != 2:
            return None
        rates = policy._interpolated_keys()
        if not valid_inputs(policy, queries, count, rows, demos, allowed, rates):
            return None
        if queries.is_cuda and len(queries) < self.min_batch:
            # Screening's launch/certificate overhead exceeds its arithmetic
            # saving for singleton queries on the measured GPU. Keep the mixed
            # bank cache alive across singleton recovery and batched inference.
            from .fused_density_search import FusedDensitySearch
            if not hasattr(self, '_small_batch_search'):
                self._small_batch_search = FusedDensitySearch()
            result = self._small_batch_search.select(policy, queries, count, rows, demos, allowed)
            route = 'fused_small_batch'
            if result is None:
                result = policy._hankel_topk(queries, count, rows, demos, allowed)
                route = 'reference_small_batch'
            self.last.update(route=route, fallback_rows=0, small_batch_rows=len(queries), reason=None)
            self.totals['calls'] += 1
            self.totals['rows'] += len(queries)
            self.totals['small_batch_rows'] += len(queries)
            return result
        signature = _signature(policy, rates)
        if signature is None:
            return None
        if signature != self.signature:
            banks = [(bank, policy.pf_norm)] + [(keys, norm) for _, keys, norm, _, _ in rates]
            norm_error = _stored_norm_error(banks)
            if norm_error is None:
                return None
            screens = [CertifiedTopK(keys) for keys, _ in banks]
            if not all(math.isfinite(value) for screen in screens for value in
                       (screen.bounds.original, screen.bounds.cast, screen.bounds.cast_error)):
                return None
            self.screens, self.norm_error, self.signature = screens, norm_error, signature
            # Every term in the error bound is monotone in these nonnegative
            # radii. Componentwise maxima cover all temporal rates while
            # computing identical query norms only once per retrieval batch.
            self.bounds = _BankBounds(*(max(getattr(screen.bounds, name) for screen in screens)
                                        for name in ('original', 'cast', 'cast_error')))
            # Retain tensors so cached version identities cannot be recycled.
            self.cache_tensors = bank_tensors(policy, rates)
        projected = queries if policy.g_lookup == 'past' else policy._g_lookup_projection(queries)
        if projected.shape != (len(queries), bank.shape[1]) or projected.dtype != torch.float64:
            return None
        width, batch, device = len(bank), len(queries), bank.device
        certified = torch.zeros(batch, dtype=torch.bool, device=device)
        output = torch.empty(batch, count, dtype=torch.long, device=device)
        values = projected.new_empty((batch, count))
        removed = torch.zeros(batch, dtype=torch.long, device=device)
        excluded = None
        if rows is not None or demos is not None:
            excluded = policy._hankel_exclusion(policy.hankel_starts, rows, demos)
            removed = excluded.sum(1)

        prefix = min(self.max_prefix, max(128, math.ceil(count * 1.5)))
        if count < prefix < width:
            # Bounds require full FP32 products.
            with ieee_fp32():
                query32 = projected.float()
                reference_error = projected.new_full((batch,), self.norm_error)
                error, refinement = _error_bounds(projected, query32, self.bounds, reference_error)
                from .fused_mixed_screen import screen as fused_screen
                fused = fused_screen(policy, query32, self.screens, rates, rows, demos, allowed)
                if fused is not None:
                    coarse, finite = fused
                    self.last['fused_screen'] = True
                else:
                    coarse, norm32 = None, query32.square().sum(1)
                    finite = torch.ones(batch, device=device, dtype=torch.bool)
                    for index, screen in enumerate(self.screens):
                        distance = torch.addmm(screen.norm32, query32, screen.bank32.T,
                                               beta=1, alpha=-2)
                        distance.add_(norm32[:, None]).clamp_min_(0)
                        finite &= torch.isfinite(distance).all(1)
                        if index:
                            rate, _, _, first, valid = rates[index-1]
                            distance.masked_fill_(self._rate_mask(
                                policy, rate, first, valid, rows), math.inf)
                        if coarse is None:
                            coarse = distance
                        else:
                            torch.minimum(coarse, distance, out=coarse)
                    if excluded is not None:
                        coarse.masked_fill_(excluded, math.inf)
                    if allowed is not None:
                        coarse.masked_fill_(~allowed, math.inf)
                self.last['fp32_products'] = len(self.screens)
                blocked = ~finite | ~torch.isfinite(error) | ~torch.isfinite(refinement)
                while count < prefix < width:
                    active = torch.nonzero(~certified & ~blocked).flatten()
                    if not len(active):
                        break
                    coarse_values, candidates = coarse[active].topk(prefix + 1, largest=False)
                    selected = candidates[:, :prefix]
                    refined = self._refine(policy, projected[active], selected,
                        None if rows is None else rows[active], rates)
                    # topk may include masked candidates when fewer than prefix
                    # are eligible. Restore those masks after FP64 refinement.
                    refined.masked_fill_(~torch.isfinite(coarse_values[:, :prefix]), math.inf)
                    best, positions = refined.topk(count, largest=False)
                    upper = torch.nextafter(best[:, -1] + refinement[active],
                                            torch.full_like(error[active], math.inf))
                    outside = torch.nextafter(coarse_values[:, prefix].double() - error[active],
                                              torch.full_like(error[active], -math.inf))
                    accepted = torch.isfinite(best).all(1) & (upper < outside)
                    chosen = active[accepted]
                    output[chosen] = selected[accepted].gather(1, positions[accepted])
                    values[chosen] = best[accepted].sqrt()
                    certified[chosen] = True
                    self.last['candidate_distances'] += len(active) * prefix * len(self.screens)
                    self.last['prefixes'].append(prefix)
                    if prefix == self.max_prefix:
                        break
                    prefix = min(self.max_prefix, prefix * 2)

        fallback = torch.nonzero(~certified).flatten()
        if len(fallback):
            arguments = [None if x is None else x[fallback] for x in (rows, demos, allowed)]
            columns, distance, fallback_removed = policy._hankel_topk(queries[fallback], count, *arguments)
            output[fallback], values[fallback], removed[fallback] = columns, distance, fallback_removed
        number = batch-len(fallback)
        self.last.update(route=('mixed_certified' if number == batch else
                                'mixed_partial' if number else 'exhaustive'),
                         certified=number, fallback_rows=len(fallback), reason=None)
        self.totals['calls'] += 1
        self.totals['rows'] += batch
        self.totals['certified_rows'] += number
        self.totals['fallback_rows'] += len(fallback)
        self.totals['candidate_distances'] += self.last['candidate_distances']
        return output, values, removed
