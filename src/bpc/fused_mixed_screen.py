"""Optional fused FP32 temporal-distance screening, never a final selector."""
import math

import torch

from .fast_cuda import extension


@torch.no_grad()
def screen(policy, queries32, screens, rates, rows=None, demos=None, allowed=None):
    """Return coarse squared distances and per-row arithmetic-valid flags.

    The caller must apply conservative error bounds and FP64 refinement before
    accepting neighbors. CUDA unavailability leaves the PyTorch screen intact.
    """
    if not queries32.is_cuda:
        return None
    module = extension(queries32.device.index, kernel='mixed_density')
    if module is None:
        return None
    empty_index = torch.empty(0, device=queries32.device, dtype=torch.long)
    empty_mask = torch.empty(0, device=queries32.device, dtype=torch.bool)
    query_rows = empty_index if rows is None else rows.contiguous()
    query_demos = empty_index if demos is None else demos.contiguous()
    permitted = empty_mask if allowed is None else allowed.contiguous()
    starts = policy.hankel_starts.contiguous()
    trajectories = policy.hankel_trajectory.contiguous()
    query_norm = queries32.square().sum(1)
    invalid = torch.zeros(len(queries32), device=queries32.device, dtype=torch.int32)
    distance = None
    for index, cached in enumerate(screens):
        product = queries32 @ cached.bank32.T
        if not index:
            distance = product
            first, valid, last = empty_index, empty_mask, 0
        else:
            rate, _, _, first, valid = rates[index-1]
            first, valid = first.contiguous(), valid.contiguous()
            last = policy.history-1+(math.ceil(rate) if policy.g_align_future else 0)
        module.postprocess(distance, product, query_norm, cached.norm32.contiguous(),
            starts, query_rows, query_demos, trajectories, first, valid, permitted,
            invalid, policy.history, last, index == 0, index == len(screens)-1)
    return distance, invalid == 0
