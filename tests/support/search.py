"""Shared temporal density bank and FP64 product instrumentation."""
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from dpc_nldl.bpc import Policy


class MatmulShapes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten.mm.default and args[0].dtype == torch.float64:
            self.shapes.append(tuple(tuple(x.shape) for x in args))
        return function(*args, **(kwargs or {}))


def density_fixture(device='cpu', width=2049, *, ties=False):
    generator = torch.Generator().manual_seed(713)

    def random(*shape):
        return torch.randn(shape, dtype=torch.float64, generator=generator).to(device)

    p = Policy.__new__(Policy)
    p.device, p.dtype = device, torch.float64
    p.history, p.g_supports, p.g_align_future, p.g_lookup = 5, 64, True, 'past'
    p.pf = random(width, 70)
    if ties:
        p.pf[:] = p.pf[0].clone()
    p.pf_norm = torch.einsum('ij,ij->i', p.pf, p.pf)
    p.hankel_starts = torch.arange(len(p.pf), device=device)
    p.hankel_trajectory = p.hankel_starts//128
    p._rate_keys = []
    if not ties:
        for rate in (.5, .75, 1.5, 2.):
            bank = random(*p.pf.shape)
            first = p.hankel_starts+p.history-1-math.ceil(rate*(p.history-1))
            valid = p.hankel_starts % 13 != 0
            p._rate_keys.append((rate, bank, bank.square().sum(1), first, valid))
    q = random(7, 70)
    rows = torch.arange(7, device=device)*17+4
    return p, q, rows
