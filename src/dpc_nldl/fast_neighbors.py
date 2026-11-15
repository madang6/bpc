"""Certified FP32 screening for the exact ordered FP64 G-support table."""

import math
from dataclasses import dataclass

import torch

from .precision import ieee_fp32


U32, U64 = 2.0**-24, 2.0**-53
MIN32, MIN64 = 2.0**-126, 2.0**-1022


def _gamma(operations, unit):
    product = operations * unit
    return math.inf if product >= 1 else product / (1 - product)


def _norm_upper(rows):
    width = rows.shape[1]
    gamma = math.nextafter(_gamma(2 * width, U64), math.inf)
    denominator = math.nextafter(1 - gamma, 0)
    square = (rows * rows).sum(1)
    square = torch.where(
        torch.isfinite(square) & (square >= 0), square,
        torch.full_like(square, math.inf))
    numerator = torch.nextafter(
        square + (4 * width + 16) * MIN64,
        torch.full_like(square, math.inf))
    upper = torch.nextafter(
        numerator / denominator, torch.full_like(numerator, math.inf))
    upper = torch.nextafter(upper.sqrt(), torch.full_like(upper, math.inf))
    return torch.where(
        torch.isfinite(upper) & (upper >= 0), upper,
        torch.full_like(upper, math.inf))


def _maximum_norm(rows):
    return math.nextafter(float(_norm_upper(rows).max()), math.inf)


@dataclass(frozen=True)
class _BankBounds:
    original: float
    cast: float
    cast_error: float


def _error_bounds(queries, queries32, bounds, reference_error=None):
    width = queries.shape[1]
    cast64 = queries32.double()
    query = _norm_upper(queries)
    query32 = _norm_upper(cast64)
    query_error = _norm_upper(cast64 - queries)
    radius = query + bounds.original
    radius32 = query32 + bounds.cast
    delta = query_error + bounds.cast_error
    cast_error = delta * (2 * radius + delta)
    gamma32 = _gamma(2 * width + 4, U32)
    coarse = ((gamma32 + 4 * U32 * (1 + gamma32)) * radius32.square()
              + (8 * width + 32) * MIN32)
    perturbation = (torch.zeros_like(radius) if reference_error is None else
                    reference_error * (2 * radius + reference_error))
    target_radius = radius if reference_error is None else radius + reference_error
    gamma64 = _gamma(2 * width + 4, U64)
    target = ((gamma64 + 4 * U64 * (1 + gamma64)) * target_radius.square()
              + (8 * width + 32) * MIN64)
    literal = (_gamma(2 * width + 3, U64) * radius.square()
               + (4 * width + 16) * MIN64)
    infinity = torch.full_like(radius, math.inf)
    return (
        torch.nextafter((cast_error + coarse + perturbation + target)
                        * (1 + 64 * U64), infinity),
        torch.nextafter((literal + perturbation + target)
                        * (1 + 64 * U64), infinity),
    )


def projection_error_bounds(inputs, projection):
    """Bound native FP64 batched projection against canonical rowwise GEMV."""
    width, output = projection.shape
    operations = 2 * width + 4
    gamma = math.nextafter(_gamma(operations, U64), math.inf)
    minimum = torch.finfo(torch.float64).tiny
    if bool((~torch.isfinite(inputs)).any() | (~torch.isfinite(projection)).any()
            | (((inputs.abs() < minimum) & (inputs != 0)).any())
            | (((projection.abs() < minimum) & (projection != 0)).any())):
        raise ValueError("projection inputs must be finite and non-subnormal")
    infinity = torch.full((len(inputs),), math.inf, device=inputs.device,
                          dtype=torch.float64)
    magnitude = torch.nextafter(
        _norm_upper(inputs) * _maximum_norm(projection.reshape(1, -1)), infinity)
    underflow = math.nextafter(
        math.sqrt(output) * operations * MIN64 / (1 - operations * U64), math.inf)
    radius = torch.nextafter(gamma * magnitude + underflow, infinity)
    return torch.nextafter(2 * radius, infinity)


class CertifiedTopK:
    """Reuse one FP32 bank to certify ordered top-k prefixes or fail closed."""

    def __init__(self, bank):
        if bank.ndim != 2 or bank.dtype != torch.float64:
            raise ValueError("certified support bank must be rank-2 float64")
        self.bank = bank
        self.bank32 = bank.float()
        self.norm32 = self.bank32.square().sum(1)
        cast64 = self.bank32.double()
        self.bounds = _BankBounds(
            _maximum_norm(bank), _maximum_norm(cast64),
            _maximum_norm(cast64 - bank))

    def select(self, queries, count, target_rows, demos, starts, trajectories, history,
               prefixes=(256, 384, 512, 1024), reference_error=None,
               ordered_contract=True):
        if (queries.ndim != 2 or queries.dtype != torch.float64
                or queries.device != self.bank.device
                or target_rows.shape != (len(queries),)
                or target_rows.dtype != torch.long
                or starts.shape != (len(self.bank),) or starts.dtype != torch.long
                or trajectories.shape != (len(self.bank),)
                or trajectories.dtype != torch.long
                or demos is not None and (demos.shape != (len(queries),)
                                          or demos.dtype != torch.long)
                or any(value.device != queries.device for value in
                       (target_rows, starts, trajectories)
                       + (() if demos is None else (demos,)))):
            raise ValueError("invalid certified support query or exclusion mask")
        prefixes = tuple(value for value in prefixes
                         if count < value < len(self.bank))
        output = torch.full(
            (len(queries), count), -1, device=queries.device, dtype=torch.long)
        certified = torch.zeros(len(queries), device=queries.device, dtype=torch.bool)
        used = torch.zeros(len(queries), device=queries.device, dtype=torch.long)
        if not prefixes:
            excluded = ((target_rows[:, None] - history <= starts)
                        & (starts <= target_rows[:, None]))
            if demos is not None:
                excluded |= trajectories == demos[:, None]
            return output, certified, used, excluded.sum(1)

        with ieee_fp32(tf32_first=True, restore_tf32_first=True):
            queries32 = queries.float()
            query_norm = queries32.square().sum(1)
            coarse = torch.addmm(
                self.norm32, queries32, self.bank32.T, beta=1, alpha=-2)
            coarse.add_(query_norm[:, None]).clamp_min_(0)
            finite = torch.isfinite(coarse).all(1)
            if demos is None:
                left = torch.searchsorted(starts, target_rows - history)
                right = torch.searchsorted(starts, target_rows, right=True)
                positions = left[:, None] + torch.arange(
                    history + 1, device=queries.device)
                valid = positions < right[:, None]
                batches = torch.arange(len(queries), device=queries.device)[:, None]
                coarse[batches.expand_as(positions)[valid], positions[valid]] = math.inf
                removed = right - left
            else:
                excluded = ((target_rows[:, None] - history <= starts)
                            & (starts <= target_rows[:, None]))
                excluded |= trajectories == demos[:, None]
                coarse.masked_fill_(excluded, math.inf)
                removed = excluded.sum(1)
            allowed = len(self.bank) - removed
            error, refinement = _error_bounds(
                queries, queries32, self.bounds, reference_error)
            blocked = (~finite | ~torch.isfinite(error)
                       | ~torch.isfinite(refinement))

            for prefix in prefixes:
                active = ~certified & ~blocked & (allowed > prefix)
                rows = torch.nonzero(active, as_tuple=False).flatten()
                if not len(rows):
                    continue
                values32, candidates = torch.topk(
                    coarse[rows], prefix + 1, largest=False, sorted=True)
                selected = candidates[:, :prefix]
                distance = (self.bank[selected]
                            - queries[rows, None]).square().sum(2)
                values, positions = torch.topk(
                    distance, count + 1, largest=False, sorted=True)
                lower = torch.nextafter(
                    values - refinement[rows, None],
                    torch.full_like(values, -math.inf))
                upper = torch.nextafter(
                    values + refinement[rows, None],
                    torch.full_like(values, math.inf))
                ordered = (upper[:, :count - 1] < lower[:, 1:count]).all(1)
                boundary = upper[:, count - 1] < lower[:, count]
                outside = torch.nextafter(
                    values32[:, prefix].double() - error[rows],
                    torch.full_like(error[rows], -math.inf))
                accepted = (torch.isfinite(distance).all(1)
                            & (ordered if ordered_contract else torch.ones_like(ordered))
                            & boundary
                            & (upper[:, count - 1] < outside))
                accepted_rows = rows[accepted]
                output[accepted_rows] = torch.gather(
                    selected[accepted], 1, positions[accepted, :count])
                certified[accepted_rows] = True
                used[accepted_rows] = prefix
                blocked[rows[~torch.isfinite(distance).all(1)]] = True
        return output, certified, used, removed
