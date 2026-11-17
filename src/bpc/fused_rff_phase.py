"""Optional pointwise RFF epilogue; GEMMs and pooling remain with the caller."""

import torch

from .fast_cuda import extension


def phase_cosine_(phase, bias, projected):
    """Apply ``phase.add_(bias).sub_(projected).cos_()``, or return None.

    This only fuses pointwise operations. Their add/subtract rounding boundaries
    remain explicit, and CUDA cos/cosf uses the normal accuracy path. The caller
    retains the original reduction and every GEMM shape, stride and row slot.
    Unsupported layouts, devices, dtypes and gradient inputs are unchanged.
    """
    if (not all(isinstance(value, torch.Tensor) for value in (phase, bias, projected))
            or any(value.is_neg() or value.is_conj() for value in (phase, bias, projected))
            or not phase.is_cuda or phase.layout != torch.strided
            or phase.dtype not in (torch.float32, torch.float64)
            or phase.ndim != 3 or not phase.is_contiguous()
            or bias.layout != torch.strided or projected.layout != torch.strided
            or bias.device != phase.device or projected.device != phase.device
            or bias.dtype != phase.dtype or projected.dtype != phase.dtype
            or bias.shape != (phase.shape[2],)
            or projected.shape != (phase.shape[0], 1, phase.shape[2])
            or any(value.requires_grad for value in (phase, bias, projected))):
        return None
    if torch._C._overlaps(phase, bias) or torch._C._overlaps(phase, projected):
        return None
    if not phase.numel():
        return phase
    module = extension(phase.device.index, kernel='fused_rff_phase')
    if module is None:
        return None
    module.phase_cosine_(phase, bias, projected)
    return phase
