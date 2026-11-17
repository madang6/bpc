"""Optional exact mixed-rate future gathering; preserves cached native actions."""
import math

import torch

from .fast_cuda import extension
from .temporal_metadata import gathered_result, temporal_constants, trajectory_stops


@torch.no_grad()
def future_columns(policy, supports, rates, states=False, *, _errors=None):
    """Return the original future tensor, or None for the unchanged fallback.

    Rate selection stays outside this function. Native action futures are read
    from uf with its existing strides and dtype. Retimed actions interpolate the
    original action source before scaling/casting; state futures interpolate the
    proprio columns at their separately shifted timestamps.
    """
    if (not supports.is_cuda or supports.dtype != torch.int64 or rates is None
            or rates.device != supports.device or rates.shape != supports.shape
            or rates.dtype not in (torch.float32, torch.float64)
            or not policy.g_align_future or len(policy.g_lookup_rates) <= 1
            or policy.horizon < 1 or policy.g_state_offset < 0):
        return None
    configured = tuple(float(rate) for rate in policy.g_lookup_rates if rate != 1)
    if not configured or any(not math.isfinite(rate) or rate <= 0 for rate in configured):
        return None
    source = policy.state if states else policy.action
    native = source if states else policy.uf
    channels = policy.proprio_dim if states else source.shape[1]
    if (source.ndim != 2 or native.ndim != 2 or not 0 < channels <= source.shape[1]
            or source.device != supports.device or native.device != supports.device
            or source.dtype not in (torch.float32, torch.float64)
            or native.dtype not in (torch.float32, torch.float64)
            or native.dtype != source.dtype
            or not source.is_contiguous()
            or (not states and (policy.action_mean.device != supports.device
                                or policy.action_scale.device != supports.device))
            or (not states and native.shape[0] != policy.horizon*channels)):
        return None
    shape = ((*supports.shape, policy.horizon, channels) if states else
             (*supports.shape, policy.horizon*channels))
    if not supports.numel():
        return torch.empty(shape, device=source.device, dtype=native.dtype)
    module = extension(supports.device.index, kernel='fused_temporal_future')
    if module is None:
        return None
    columns = supports.reshape(-1).contiguous()
    starts = policy.hankel_starts[columns].contiguous()
    stops = trajectory_stops(policy)[policy.hankel_trajectory[columns]].contiguous()
    configured, mean, scale, scaled = temporal_constants(policy, source, configured, actions=not states)
    result, error = module.temporal_future(source, native, columns, starts,
        rates.reshape(-1).contiguous(), configured, stops, mean, scale, scaled,
        policy.history, policy.horizon, policy.g_state_offset if states else 0, channels, states)
    return gathered_result(result, error, shape, 'future', _errors)
