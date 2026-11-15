"""Literal 5eee261 readout equivalence and avoided CUDA synchronization."""
import math
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch

from tests.support.readout import affine_problems as fixture


def literal_affine(self, design, query):
    """Direct affine solve in configured precision; refine FP64 queries.

    With ``fixed_shape_readout`` no row's arithmetic depends on the others:
    the Cholesky path runs on every row (``cholesky_ex``, so failures do not
    raise) for exactly four refinement steps, the fallback runs on every row
    whenever any row needs it, and ``good`` selects per row afterwards.
    """
    design = design.to(query.dtype)
    batch, _, count = design.shape
    if count == 1:
        return query.new_ones((batch, 1))
    root = math.sqrt(count)
    source = design.float()
    reduced = source[:, :, :-1] - (
        (source.sum(-1) + root * source[:, :, -1]) / (count + root))[:, :, None]
    target = query.to(source.dtype) - source.mean(-1)
    del source
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        gram = (reduced.transpose(1, 2) @ reduced).double()
        rhs = (reduced.transpose(1, 2) @ target[:, :, None]).double()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    del reduced, target
    def expand(coordinates):
        shift = coordinates.sum(1, keepdim=True) / (count + root)
        result = torch.nn.functional.pad(coordinates, (0, 0, 0, 1)) - shift
        result[:, -1:] -= root * shift
        return result + 1 / count
    guard = gram.clone()
    guard.diagonal(dim1=-2, dim2=-1).sub_(
        (8 * torch.finfo(torch.float32).eps * gram.abs().sum(-1).amax(-1))[:, None])
    good = torch.linalg.cholesky_ex(guard).info == 0
    del guard
    fixed = self.fixed_shape_readout
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
            for _ in range(4):
                residual = target[:, :, None] - matrix @ expand(coordinates)
                correction = torch.cholesky_solve(project(matrix.transpose(1, 2) @ residual), factor)
                coordinates += correction
                converged = correction.abs().amax((1, 2)) <= (
                    1e-10 * coordinates.abs().amax((1, 2)).clamp_min(1))
                if converged.all() and not fixed:
                    break
        output[rows] = expand(coordinates).squeeze(2).to(query.dtype)
        if query.dtype == torch.float64:
            if fixed:
                good &= converged
            else:
                good[good.clone()] = converged
    if not good.all():
        rows = slice(None) if fixed else ~good
        fallback = design[rows].double()
        transpose = fallback.transpose(1, 2)
        solved = self._solve_g_normal(
            transpose @ fallback,
            (transpose @ query[rows, :, None].double()).squeeze(2)).to(query.dtype)
        output[~good] = solved[~good] if fixed else solved
    return output


class FixedShapeAffineSpeedTest(unittest.TestCase):
    def check_reference(self, device):
        for dtype in (torch.float32, torch.float64):
            for fixed in (True, False):
                with self.subTest(device=device, dtype=dtype, fixed=fixed):
                    policy, problems = fixture(device, dtype, fixed)
                    for design, query in problems:
                        with patch.object(policy, '_solve_g_normal', wraps=policy._solve_g_normal) as fallback:
                            expected = literal_affine(policy, design, query)
                            old_fallbacks = fallback.call_count
                        with patch.object(policy, '_solve_g_normal', wraps=policy._solve_g_normal) as fallback:
                            actual = policy._solve_affine(design, query)
                            self.assertEqual(fallback.call_count, old_fallbacks)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cpu_literal_coefficients_and_fallback_match(self):
        self.check_reference('cpu')

    @cuda
    def test_cuda_literal_coefficients_and_fallback_match(self):
        self.check_reference('cuda')

    @cuda
    def test_cuda_fixed_readout_removes_only_unused_host_reads_and_reductions(self):
        policy, problems = fixture('cuda', torch.float64, fixed=True)
        design, query = problems[1]
        # Profile one good row so neither implementation executes the fallback.
        design, query = design[1:2], query[1:2]
        counts = []
        for solve in (lambda: literal_affine(policy, design, query),
                      lambda: policy._solve_affine(design, query)):
            solve()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                solve()
                torch.cuda.synchronize()
            counts.append({item.key: item.count for item in profile.key_averages()})
        old, new = counts
        self.assertEqual(old['aten::_local_scalar_dense']-new['aten::_local_scalar_dense'], 4)
        self.assertEqual(old['aten::amax']-new['aten::amax'], 6)
        self.assertEqual(old['aten::cholesky_solve'], 5)
        self.assertEqual(new['aten::cholesky_solve'], 5)
        self.assertEqual(old['aten::linalg_cholesky_ex'], new['aten::linalg_cholesky_ex'])


if __name__ == '__main__':
    unittest.main()
