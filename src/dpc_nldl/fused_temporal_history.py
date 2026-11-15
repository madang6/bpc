"""Optional exact gather for per-support temporal reconstruction histories.

This module is deliberately separate from Policy. Callers use the original
PyTorch path when ``reconstruction_columns`` returns None. No rate selection,
support selection, or reconstruction solve is performed here.
"""
import math

import torch

from .fast_cuda import extension
from .temporal_metadata import gathered_result, temporal_constants


@torch.no_grad()
def reconstruction_columns(policy, supports, rates, dtype=None, *, _errors=None):
    """Gather the same action-first histories as Policy._reconstruction_columns.

    FP32 output uses the policy's already-rounded FP32 source banks, exactly as
    _past_columns does. Other outputs interpolate in the original source dtype
    and cast only afterward. Unsupported devices/layouts return None. The kernel
    detects the original interpolated-history boundary violation before reading
    memory; other invalid storage indices fall back to the original index path.
    """
    if (not supports.is_cuda or supports.dtype != torch.int64 or rates is None
            or rates.device != supports.device or rates.shape != supports.shape
            or rates.dtype not in (torch.float32, torch.float64)
            or dtype not in (None, torch.float32, torch.float64)
            or not policy.g_align_history or len(policy.g_lookup_rates) <= 1
            or policy.history < 1 or policy.g_state_offset < 0):
        return None
    configured = tuple(float(rate) for rate in policy.g_lookup_rates if rate != 1)
    if not configured or any(not math.isfinite(rate) or rate <= 0 for rate in configured):
        return None
    actions, states = (policy._float_sources() if dtype == torch.float32
                       else (policy.action, policy.state))
    if (states.device != supports.device or actions.device != supports.device
            or states.ndim != 2 or actions.ndim != 2
            or states.dtype not in (torch.float32, torch.float64) or actions.dtype != states.dtype
            or not states.is_contiguous() or not actions.is_contiguous()
            or policy.action_mean.device != supports.device
            or policy.action_scale.device != supports.device):
        return None
    output_dtype = states.dtype if dtype is None else dtype
    width = policy.history*(states.shape[1]+actions.shape[1])
    if not supports.numel():
        return torch.empty((*supports.shape, width), device=states.device, dtype=output_dtype)
    module = extension(supports.device.index, kernel='fused_temporal_history')
    if module is None:
        return None
    columns = supports.reshape(-1)
    starts = policy.hankel_starts[columns].contiguous()
    lower_bounds = policy.trajectory_starts[policy.hankel_trajectory[columns]].contiguous()
    configured, mean, scale, scaled = temporal_constants(policy, actions, configured)
    result, error = module.temporal_history(states, actions, starts,
        rates.reshape(-1).contiguous(), configured, lower_bounds, mean, scale, scaled,
        policy.history, policy.g_state_offset, output_dtype == torch.float64)
    return gathered_result(result, error, (*supports.shape, width), 'history', _errors)
