import unittest
from tests.suites import cuda

import torch

from dpc_nldl.bpc import Policy


class AffineExactRankTest(unittest.TestCase):


    def check_rank_cases(self, device):
        policy = Policy.__new__(Policy)
        policy.g_solver = "exact"
        policy.history, policy.state_dim, policy.action_dim, policy.horizon = 1, 10, 1, 1
        generator = torch.Generator().manual_seed(4)
        design = torch.randn(5, 11, 7, generator=generator, dtype=torch.float64)
        query = torch.randn(5, 11, generator=generator, dtype=torch.float64)
        design[0, :, -1] = design[0, :, 0]
        design[1].zero_()
        design[2] = design[2, :, :1].expand(-1, 7)
        design[3, :, -1] = design[3, :, 0] + 1e-10 * design[3, :, -1]
        for dtype in (torch.float32, torch.float64):
            z, q = design.to(device, dtype), query.to(device, dtype)
            z64, q64 = z.double(), q.double()
            mean = z64.mean(2, keepdim=True)
            centered = z64 - mean
            centered[2].zero_()  # Ignore mean-rounding noise for identical columns.
            expected = 1 / 7 + (torch.linalg.pinv(centered, rtol=1e-7) @
                                (q64 - mean.squeeze(2))[:, :, None]).squeeze(2)
            gram = z64.transpose(1, 2) @ z64
            rhs = (z64.transpose(1, 2) @ q64[:, :, None]).squeeze(2)
            for solver, actual in (
                    ("normal", policy._solve_g_normal(gram, rhs)),
                    ("design", policy._solve_affine(z, q))):
                with self.subTest(device=device, dtype=dtype, solver=solver):
                    tolerance = 2e-6 if dtype == torch.float32 else 2e-8
                    self.assertEqual(actual.dtype, dtype if solver == "design" else torch.float64)
                    torch.testing.assert_close(actual.double(), expected,
                                               rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(actual.sum(1), actual.new_ones(5), rtol=0,
                                               atol=2e-6 if dtype == torch.float32 else 2e-12)
                    residual = z64 @ actual.double()[:, :, None] - q64[:, :, None]
                    optimum = z64 @ expected[:, :, None] - q64[:, :, None]
                    torch.testing.assert_close(residual.square().sum((1, 2)),
                                               optimum.square().sum((1, 2)),
                                               rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(policy._solve_affine(z[:, :, :1], q), q.new_ones((5, 1)))

    def test_rank_deficient_and_full_rank_batch(self):
        self.check_rank_cases("cpu")

    @cuda
    def test_rank_deficient_and_full_rank_cuda_batch(self):
        self.check_rank_cases("cuda")


if __name__ == "__main__":
    unittest.main()
