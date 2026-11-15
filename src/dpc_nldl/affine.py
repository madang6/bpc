"""Affine-coordinate Cholesky solves shared by policy and graph probes.

The caller owns singular fallback. Keeping that host decision outside the core
allows CUDA capture without changing any reduction shape or stopping rule.
"""
import math

import torch


def normal_equations(design, target):
    """Keep full-precision products and return the FP64 Gram and right-hand side."""
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return ((design.transpose(1, 2) @ design).double(),
                (design.transpose(1, 2) @ target[:, :, None]).double())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def expand_coordinates(coordinates, count, root):
    shift = coordinates.sum(1, keepdim=True) / (count + root)
    result = torch.nn.functional.pad(coordinates, (0, 0, 0, 1)) - shift
    result[:, -1:] -= root * shift
    return result + 1 / count


def solve_coordinates(design, query, *, fixed=False):
    """Return coefficients and the per-row guard/refinement certificate."""
    design = design.to(query.dtype)
    batch, _, count = design.shape
    if count == 1:
        return query.new_ones((batch, 1)), torch.ones(batch, device=query.device, dtype=torch.bool)
    root = math.sqrt(count)
    source = design.float()
    reduced = source[:, :, :-1] - (
        (source.sum(-1) + root * source[:, :, -1]) / (count + root))[:, :, None]
    target = query.to(source.dtype) - source.mean(-1)
    del source
    gram, rhs = normal_equations(reduced, target)
    del reduced, target
    guard = gram.clone()
    guard.diagonal(dim1=-2, dim2=-1).sub_(
        (8 * torch.finfo(torch.float32).eps * gram.abs().sum(-1).amax(-1))[:, None])
    good = torch.linalg.cholesky_ex(guard).info == 0
    del guard
    rows = slice(None) if fixed else good
    output = query.new_empty((batch, count))
    if fixed or good.any():
        factor = (torch.linalg.cholesky_ex(gram).L if fixed
                  else torch.linalg.cholesky(gram[good]))
        def project(vector):
            return vector[:, :-1] - (vector.sum(1, keepdim=True)
                + root * vector[:, -1:]) / (count + root)
        coordinates = torch.cholesky_solve(rhs[rows], factor)
        if query.dtype == torch.float64:
            matrix, target = ((design, query) if fixed or good.all()
                              else (design[good], query[good]))
            for refinement in range(4):
                residual = target[:, :, None] - matrix @ expand_coordinates(coordinates, count, root)
                correction = torch.cholesky_solve(project(matrix.transpose(1, 2) @ residual), factor)
                coordinates += correction
                # Fixed readout always performs all four corrections; only
                # the final convergence mask participates in its fallback.
                if not fixed or refinement == 3:
                    converged = correction.abs().amax((1, 2)) <= (
                        1e-10 * coordinates.abs().amax((1, 2)).clamp_min(1))
                    if not fixed and converged.all():
                        break
        output[rows] = expand_coordinates(coordinates, count, root).squeeze(2).to(query.dtype)
        if query.dtype == torch.float64:
            if fixed:
                good &= converged
            else:
                good[good.clone()] = converged
    return output, good
