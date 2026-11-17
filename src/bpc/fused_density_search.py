"""Exhaustive density search with exact fused pointwise FP64 arithmetic."""
import math

import torch

from .density_search import _signature, bank_tensors, valid_inputs
from .fast_cuda import extension


class FusedDensitySearch:
    """Keep reference GEMM/topk shapes, and fuse only elementwise processing."""

    def __init__(self):
        self.signature = None
        self.last = {}

    @torch.no_grad()
    def select(self, policy, queries, count, rows=None, demos=None, allowed=None):
        self.last = dict(route='fallback', rows=len(queries), reason='unsupported')
        bank = getattr(policy, 'pf', None)
        if (not isinstance(bank, torch.Tensor) or not bank.is_cuda
                or bank.ndim != 2 or bank.dtype != torch.float64
                or queries.ndim != 2 or queries.dtype != torch.float64
                or queries.device != bank.device or not len(queries)
                or not 0 < count <= len(bank)):
            return None
        batch, width, device = len(queries), len(bank), bank.device
        rates = policy._interpolated_keys()
        if not valid_inputs(policy, queries, count, rows, demos, allowed, rates):
            return None
        signature = _signature(policy, rates)
        if signature is None:
            return None
        if signature != self.signature:
            # Fail closed on NaNs, infinities or overflow-prone inputs. The
            # reference remains responsible for their exact backend behavior.
            banks = [(bank, policy.pf_norm)]+[(keys, norm) for _, keys, norm, _, _ in rates]
            safe = torch.stack([((keys.abs() <= 1e50).all()
                                 & (norm.abs() <= 1e100).all()) for keys, norm in banks])
            if not bool(safe.all()):
                return None
            self.signature = signature
            self.cache_tensors = bank_tensors(policy, rates)
        projected = (queries if policy.g_lookup == 'past' else
                     policy._g_lookup_projection(queries))
        if projected.shape != (batch, bank.shape[1]) or projected.dtype != torch.float64:
            return None
        if not bool((projected.abs() <= 1e50).all()):
            return None
        module = extension(device.index, kernel='exact_density')
        if module is None:
            self.last['reason'] = 'extension_unavailable'
            return None
        empty_index = torch.empty(0, device=device, dtype=torch.long)
        empty_mask = torch.empty(0, device=device, dtype=torch.bool)
        query_rows = empty_index if rows is None else rows.contiguous()
        query_demos = empty_index if demos is None else demos.contiguous()
        allowed = empty_mask if allowed is None else allowed.contiguous()
        starts = policy.hankel_starts.contiguous()
        trajectories = policy.hankel_trajectory.contiguous()
        query_norm = torch.einsum('ij,ij->i', projected, projected)
        distance = projected @ bank.T
        module.postprocess(distance, distance, query_norm, policy.pf_norm.contiguous(),
            starts, query_rows, query_demos, trajectories, empty_index, empty_mask,
            allowed, policy.history, 0, True, not rates)
        for index, (rate, keys, norm, first, valid) in enumerate(rates):
            product = projected @ keys.T
            last = policy.history-1+(math.ceil(rate) if policy.g_align_future else 0)
            module.postprocess(distance, product, query_norm, norm.contiguous(),
                starts, query_rows, query_demos, trajectories, first.contiguous(),
                valid.contiguous(), allowed, policy.history, last, False, index == len(rates)-1)
        excluded = (torch.zeros(batch, device=device, dtype=torch.long)
                    if rows is None and demos is None else
                    policy._hankel_exclusion(policy.hankel_starts, rows, demos).sum(1))
        values, columns = torch.topk(distance, count, largest=False)
        self.last.update(route='fused_exhaustive', reason=None)
        return columns, values.sqrt(), excluded
