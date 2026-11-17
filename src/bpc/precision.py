"""Scoped full-FP32 products with the existing backend restoration order."""
from contextlib import contextmanager

import torch


def _set_matmul(precision, tf32, tf32_first):
    # These APIs are coupled: setting allow_tf32=True changes medium to high.
    if tf32_first:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.set_float32_matmul_precision(precision)
    if not tf32_first:
        torch.backends.cuda.matmul.allow_tf32 = tf32


@contextmanager
def ieee_fp32(*, tf32_first=False, restore_tf32_first=False):
    """Preserve each caller's setter order, including its exception cleanup."""
    if tf32_first:
        tf32 = torch.backends.cuda.matmul.allow_tf32
        precision = torch.get_float32_matmul_precision()
    else:
        precision = torch.get_float32_matmul_precision()
        tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        _set_matmul("highest", False, tf32_first)
        yield
    finally:
        _set_matmul(precision, tf32, restore_tf32_first)
