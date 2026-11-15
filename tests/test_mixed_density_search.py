"""Mixed retrieval removes full-bank FP64 work without capping support."""
import math
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch

from dpc_nldl.mixed_density_search import MixedDensitySearch
from tests.support.search import MatmulShapes, density_fixture as fixture


class MixedDensitySearchTest(unittest.TestCase):
    device = 'cpu'

    def assert_matches(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        torch.testing.assert_close(actual[1], expected[1], rtol=2e-13, atol=2e-13)
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_inference_training_demo_and_recovery_masks(self):
        p, q, rows = fixture(self.device)
        allowed = torch.ones(len(q), len(p.pf), dtype=torch.bool, device=self.device)
        allowed[:, ::3] = False
        search = MixedDensitySearch()
        for row_mask, demos, mask in ((None, None, None), (rows, None, None),
                (None, rows//128, allowed), (rows, rows//128, allowed)):
            expected = p._hankel_topk(q, 256, row_mask, demos, mask)
            with MatmulShapes() as products:
                actual = search.select(p, q, 256, row_mask, demos, mask)
            self.assert_matches(actual, expected)
            self.assertEqual(search.last['route'], 'mixed_certified')
            self.assertEqual(products.shapes, [])
            self.assertEqual(search.last['fp32_products'], 5)
        self.assertEqual(search.totals['certified_rows'], 4*len(q))

    def test_support_growth_is_not_capped_by_density_or_speculation_budget(self):
        p, q, rows = fixture(self.device)
        search = MixedDensitySearch(max_prefix=1024)
        for count in (1, 64, 256, 257, 512, 1024, 1500, len(p.pf)):
            with self.subTest(count=count):
                self.assert_matches(search.select(p, q, count, rows),
                                    p._hankel_topk(q, count, rows))
                self.assertEqual(search.last['certified'], len(q) if count < 1024 else 0)

    def test_bank_norm_and_metadata_mutations_invalidate_cache(self):
        p, q, rows = fixture(self.device)
        search = MixedDensitySearch()
        search.select(p, q, 256, rows)
        for change in (lambda: p.pf.add_(.1), lambda: p.pf_norm.add_(.0001),
                       lambda: p._rate_keys[-1][-1].fill_(False)):
            old = search.screens
            change()
            self.assert_matches(search.select(p, q, 256, rows), p._hankel_topk(q, 256, rows))
            self.assertIsNot(old, search.screens)

    def test_ambiguous_rows_fall_back_independently_and_restore_precision(self):
        p, q, rows = fixture(self.device)
        p._rate_keys = []
        p.pf[1:].copy_(p.pf[0])
        p.pf_norm.copy_(p.pf.square().sum(1))
        search = MixedDensitySearch(max_prefix=512)
        expected = p._hankel_topk(q, 256, rows)
        prior_precision = torch.get_float32_matmul_precision()
        prior_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.assert_matches(search.select(p, q, 256, rows), expected)
        self.assertEqual(search.last['route'], 'exhaustive')
        self.assertEqual(search.last['prefixes'], [384, 512])
        self.assertEqual(torch.get_float32_matmul_precision(), prior_precision)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, prior_tf32)

    def test_partial_fallback_and_insufficient_eligibility(self):
        p, q, rows = fixture(self.device)
        allowed = torch.ones(len(q), len(p.pf), dtype=torch.bool, device=self.device)
        allowed[2] = False
        allowed[2, :200] = True
        search = MixedDensitySearch(max_prefix=512)
        with patch.object(p, '_hankel_topk', wraps=p._hankel_topk) as original:
            actual = search.select(p, q, 256, rows, allowed=allowed)
        self.assert_matches(actual, p._hankel_topk(q, 256, rows, allowed=allowed))
        self.assertEqual(search.last['fallback_rows'], 1)
        self.assertEqual(original.call_count, 1)
        self.assertEqual(len(original.call_args.args[0]), 1)

    def test_projection_and_low_eligibility_preserve_finite_neighbors(self):
        p, projected, rows = fixture(self.device)
        p.g_lookup = 'fisher'
        p.g_lookup_matrix = torch.eye(70, dtype=torch.float64, device=self.device)
        p._g_lookup_projection = lambda value: value[:, :70] * 2
        queries = torch.cat([projected / 2, projected[:, :3]], dim=1)
        allowed = torch.zeros(len(queries), len(p.pf), dtype=torch.bool, device=self.device)
        allowed[:, :300] = True
        search = MixedDensitySearch()
        actual = search.select(p, queries, 64, allowed=allowed)
        self.assert_matches(actual, p._hankel_topk(queries, 64, allowed=allowed))
        self.assertEqual(search.last['certified'], len(queries))
        self.assertTrue((actual[0] < 300).all())

    def test_nonfinite_query_falls_back_only_for_affected_row(self):
        p, q, rows = fixture(self.device)
        q[2, 0] = math.nan
        search = MixedDensitySearch()
        with patch.object(p, '_hankel_topk', wraps=p._hankel_topk) as original:
            actual = search.select(p, q, 64)
        expected = p._hankel_topk(q, 64)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        torch.testing.assert_close(actual[1], expected[1], rtol=2e-13, atol=2e-13,
                                   equal_nan=True)
        self.assertEqual(search.last['fallback_rows'], 1)
        self.assertEqual(original.call_count, 1)
        self.assertEqual(len(original.call_args.args[0]), 1)

    def test_error_bound_inflation_cannot_drop_hidden_neighbor(self):
        p, q, rows = fixture(self.device)
        p._rate_keys = []
        p.pf = torch.arange(2049, dtype=torch.float64, device=self.device)[:, None]*10
        p.pf_norm = p.pf.square().sum(1)
        p.pf_norm[-1] = 0.
        q = p.pf.new_tensor([[200.5]])
        search = MixedDensitySearch(max_prefix=512, min_batch=1)
        actual = search.select(p, q, 3)
        self.assert_matches(actual, p._hankel_topk(q, 3))
        self.assertEqual(int(actual[0][0, 0]), 2048)
        self.assertEqual(search.last['route'], 'exhaustive')

    def test_optional_fused_screen_failure_retains_pytorch_screen(self):
        p, q, rows = fixture(self.device)
        search = MixedDensitySearch()
        with patch('dpc_nldl.fused_mixed_screen.screen', return_value=None):
            actual = search.select(p, q, 64, rows)
        self.assert_matches(actual, p._hankel_topk(q, 64, rows))
        self.assertFalse(search.last['fused_screen'])
        self.assertEqual(search.last['route'], 'mixed_certified')

    def test_close_fp32_boundary_expands_before_certifying(self):
        p, _, _ = fixture(self.device)
        p._rate_keys = []
        p.pf = (2e8 + torch.arange(len(p.pf), dtype=torch.float64,
                                  device=self.device))[:, None].sqrt()
        p.pf_norm = p.pf.square().sum(1)
        q = p.pf.new_zeros((2, 1))
        search = MixedDensitySearch(max_prefix=1024)
        actual = search.select(p, q, 64)
        self.assert_matches(actual, p._hankel_topk(q, 64))
        self.assertEqual(search.last['route'], 'mixed_certified')
        self.assertGreater(len(search.last['prefixes']), 1)

    def test_no_eligible_candidates_never_receive_a_certificate(self):
        p, q, rows = fixture(self.device)
        allowed = torch.zeros(len(q), len(p.pf), dtype=torch.bool, device=self.device)
        search = MixedDensitySearch(max_prefix=128)
        self.assert_matches(search.select(p, q, 64, allowed=allowed),
                            p._hankel_topk(q, 64, allowed=allowed))
        self.assertEqual(search.last['route'], 'exhaustive')
        self.assertEqual(search.last['certified'], 0)

    def test_unsupported_inputs_and_exception_restore_fp32_mode(self):
        p, q, rows = fixture(self.device)
        search = MixedDensitySearch()
        self.assertIsNone(search.select(p, q.float(), 64))
        self.assertIsNone(search.select(p, q, 0))
        self.assertIsNone(search.select(p, q, 64, rows.float()))
        with torch.inference_mode():
            p.pf = p.pf.clone()
            self.assertIsNone(search.select(p, q, 64))
        p, q, rows = fixture(self.device)
        precision = torch.get_float32_matmul_precision()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        with patch.object(search, '_refine', side_effect=RuntimeError('test')):
            with self.assertRaisesRegex(RuntimeError, 'test'):
                search.select(p, q, 64)
        self.assertEqual(torch.get_float32_matmul_precision(), precision)
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, tf32)


@cuda
class CudaMixedDensitySearchTest(MixedDensitySearchTest):
    device = 'cuda'

    def test_singleton_dispatch_keeps_mixed_cache_and_separate_statistics(self):
        p, q, rows = fixture(self.device)
        search = MixedDensitySearch()
        search.select(p, q, 64, rows)
        cached = search.screens
        expected = p._hankel_topk(q[:1], 64, rows[:1])
        self.assert_matches(search.select(p, q[:1], 64, rows[:1]), expected)
        self.assertIs(search.screens, cached)
        self.assertIn(search.last['route'], ('fused_small_batch', 'reference_small_batch'))
        self.assertEqual(search.last['small_batch_rows'], 1)
        self.assertEqual(search.last['fallback_rows'], 0)
        self.assertEqual(search.last['fp32_products'], 0)
        self.assertEqual(search.totals['small_batch_rows'], 1)
        self.assertEqual(search.totals['certified_rows'], len(q))


if __name__ == '__main__':
    unittest.main()
