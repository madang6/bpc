"""Shared, optional CUDA loading for fitting and evaluation."""

import shutil
import warnings
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=None)
def extension(device, kernel="fast_cuda"):
    kernel = "density" if kernel in {"exact_density", "mixed_density"} else kernel
    root = Path(__file__).parent
    sources = [root / "csrc" / f"{kernel}{suffix}"
               for suffix in (".cpp", "_kernel.cu")]
    from torch.utils.cpp_extension import CUDA_HOME, load
    nvcc = Path(CUDA_HOME or "") / "bin" / "nvcc"
    if not CUDA_HOME or not (shutil.which("nvcc") or nvcc.is_file()):
        return None
    try:
        return load(
            name=f"dpc_nldl_{kernel}_v1", verbose=False, with_cuda=True,
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--fmad=true"],
            sources=list(map(str, sources)))
    except Exception as error:
        warnings.warn(f"{kernel} build failed; using PyTorch: {error}")
        return None


def call(name, reference, *arguments):
    if not reference.is_cuda:
        return None
    if name == "history_accumulators":
        candidates, states = arguments[2:4]
        shared = (candidates.shape[1] + states.shape[1]) * states.element_size()
        if shared > torch.cuda.get_device_properties(reference.device).shared_memory_per_block:
            return None
    module = extension(reference.device.index, "fast_cuda")
    return None if module is None else getattr(module, name)(*arguments)
