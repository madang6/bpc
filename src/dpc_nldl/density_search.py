"""Shared input validation and cache bounds for active density-search backends."""
import math

import torch


def bank_tensors(policy, rates):
    """Inputs retained by every density-search cache and its mutation signature."""
    tensors = [policy.pf, policy.pf_norm, policy.hankel_starts,
               policy.hankel_trajectory]
    tensors += [value for _, keys, norm, first, valid in rates
                for value in (keys, norm, first, valid)]
    return tensors


def valid_inputs(policy, queries, count, rows, demos, allowed, rates):
    bank = policy.pf
    batch, width, device = len(queries), len(bank), bank.device
    return (bank.ndim == 2 and bank.dtype == torch.float64
        and queries.ndim == 2 and queries.dtype == torch.float64
        and queries.device == device and batch > 0 and 0 < count <= width
        and all(x.device == device and x.dtype == torch.long
                and x.shape == (width,) for x in
                (policy.hankel_starts, policy.hankel_trajectory))
        and all(x is None or (x.device == device and x.dtype == torch.long
                and x.shape == (batch,)) for x in (rows, demos))
        and (allowed is None or (allowed.device == device
                and allowed.dtype == torch.bool and allowed.shape == (batch, width)))
        and policy.pf_norm.shape == (width,)
        and policy.pf_norm.dtype == torch.float64 and policy.pf_norm.device == device
        and all(keys.shape == bank.shape and keys.dtype == torch.float64
                and keys.device == device and norm.shape == (width,)
                and norm.dtype == torch.float64 and norm.device == device
                and first.shape == (width,) and first.dtype == torch.long
                and first.device == device and valid.shape == (width,)
                and valid.dtype == torch.bool and valid.device == device
                and math.isfinite(rate) and rate > 0
                for rate, keys, norm, first, valid in rates))


def _signature(policy, rates):
    try:
        versions = tuple((id(value), value._version) for value in bank_tensors(policy, rates))
    except RuntimeError:  # Inference-mode tensors have no mutation counter.
        return None
    return (policy.history, policy.g_align_future,
            tuple(rate for rate, *_ in rates), versions)


def _stored_norm_error(banks):
    """Convert arbitrary stored-norm error into the screen's distance bound.

    The existing target bound covers the ordinary square/sum norm recipe. For
    a stored norm differing by at most delta, choose e with 2*B*e >= delta,
    where B=max(abs(bank)) is a lower bound on the screen's bank norm radius.
    Its perturbation term e*(2*radius+e) therefore covers this extra error too.
    Each scalar operation rounds outwards; no cached norm is assumed correct.
    """
    error = 0.
    for bank, norm in banks:
        delta = float((norm-bank.square().sum(1)).abs().max())
        bound = float(bank.abs().max())
        if not math.isfinite(delta) or not math.isfinite(bound):
            return None
        if not delta:
            continue
        delta = math.nextafter(delta, math.inf)
        if bound:
            # Dividing before halving avoids overflow in 2*bound.
            value = math.nextafter(delta/bound, math.inf)
            value = math.nextafter(value/2, math.inf)
        else:
            value = math.nextafter(math.sqrt(delta), math.inf)
        if not math.isfinite(value):
            return None
        error = max(error, value)
    return error
