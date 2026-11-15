"""Retained Fisher selector fixture and exact temporal numerical dependencies."""
import unittest
from tests.suites import cuda
import torch

from dpc_nldl.bpc import Policy
from dpc_nldl.lda_selector import fixed_projection
from tests.support.policies import data_config, fisher_recipe, trajectories


class FisherSelectorTest(unittest.TestCase):
    def check_fixed_projection(self, device, width):
        generator = torch.Generator().manual_seed(23)
        rows = torch.randn(65, width, generator=generator, dtype=torch.float64).to(device)
        matrix = torch.randn(128, width, generator=generator, dtype=torch.float64).to(device).T
        expected = fixed_projection(rows, matrix)
        split = torch.cat([fixed_projection(batch, matrix) for batch in rows.split([1, 7, 23, 32, 2])])
        torch.testing.assert_close(split, expected, rtol=0, atol=0)
        permutation = torch.randperm(len(rows), generator=generator).to(device)
        torch.testing.assert_close(fixed_projection(rows[permutation], matrix),
                                   expected[permutation], rtol=0, atol=0)
        torch.testing.assert_close(torch.cat([fixed_projection(row[None], matrix) for row in rows]),
                                   expected, rtol=0, atol=0)
        torch.testing.assert_close(fixed_projection(rows[:64].reshape(2, 32, -1), matrix).flatten(0, 1),
                                   expected[:64], rtol=0, atol=0)
        precision, tf32 = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
        try:
            torch.set_float32_matmul_precision("medium")
            torch.testing.assert_close(fixed_projection(rows, matrix), expected, rtol=0, atol=0)
            self.assertEqual(torch.get_float32_matmul_precision(), "medium")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.set_float32_matmul_precision(precision)

    def test_fixed_projection_group_order_and_padding_cpu(self):
        # The established implementation's single-thread CPU BLAS path
        # changes rounding for partially zero-padded batches; use the stable
        # two-thread backend for this exact-shape invariant (CUDA is unchanged).
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            for width in (5200, 10400):
                with self.subTest(width=width):
                    self.check_fixed_projection("cpu", width)
        finally:
            torch.set_num_threads(threads)

    @cuda
    def test_fixed_projection_group_order_and_padding_cuda(self):
        for width in (5200, 10400):
            with self.subTest(width=width):
                self.check_fixed_projection("cuda", width)

    def fixed_policy(self):
        method = fisher_recipe()
        p = Policy(trajectories(demos=4, length=16), data_config(), method,
                   fit_residuals=False)
        width = p.history * (p.action_dim + p.state_dim)
        p.g_lookup_matrix = torch.eye(width, dtype=p.dtype) / 10
        selector = p.support_selector
        selector.omega = p.g_lookup_matrix
        selector.phase = torch.linspace(-1, 1, width)
        p._project_hankel()
        return p, selector

    def test_cancellation_refines_to_the_same_temporal_rate_as_readout(self):
        p, selector = self.fixed_policy()
        p.pf = torch.full_like(p.pf, 1000.) + .01
        alternate = torch.full_like(p.pf, 1000.) + .005
        p._rate_keys = [(.5, alternate, alternate.square().sum(1), p.hankel_starts,
                         torch.ones(len(p.hankel_starts), dtype=torch.bool))]
        selector.release()
        keys = torch.full((1, p.pf.shape[1]), 1000., dtype=torch.float32)
        # The fast GEMM proposals tie, despite the second rate being closer.
        proposal = [torch.cdist(keys, bank.float()).square() for bank in (p.pf, alternate)]
        self.assertTrue(torch.equal(proposal[0], proposal[1]))
        distance, rates = selector._distance(keys)
        supports = torch.arange(len(p.hankel_starts))[None]
        direct, selected_rates = selector._distance(keys, supports=supports)
        torch.testing.assert_close(rates, selected_rates, rtol=0, atol=0)
        self.assertTrue(bool((rates == 1).all()))
        torch.testing.assert_close(distance.double(), direct, rtol=1e-7, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
