"""Active fused density search preserves independent reference arithmetic."""
import math
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch

from bpc.fused_density_search import FusedDensitySearch
from bpc.mixed_density_search import MixedDensitySearch
from bpc.bpc import Policy
from bpc.lda_selector import Selector
from tests.support.search import MatmulShapes, density_fixture as fixture


class FusedDensitySearchTest(unittest.TestCase):
    device = 'cpu'

    def assert_equal(self, actual, expected):
        self.assertIsNotNone(actual)
        for a, b in zip(actual, expected):
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a.contiguous().view(torch.uint8),
                                        b.contiguous().view(torch.uint8)))

    def test_selector_preserves_custom_exclusion_and_overlap_overrides(self):
        for mode, search in (('fused', FusedDensitySearch), ('mixed', MixedDensitySearch)):
            for name in ('_rate_target_overlap', '_hankel_exclusion', '_exclude_hankel'):
                with self.subTest(mode=mode, method=name):
                    p, q, rows = fixture(self.device)
                    original = getattr(p, name)
                    calls = []

                    def custom(*args, **kwargs):
                        calls.append(True)
                        result = original(*args, **kwargs)
                        return ~result if result.dtype == torch.bool else result

                    setattr(p, name, custom)
                    expected = p._hankel_topk(q, 256, rows)
                    calls.clear()
                    selector = Selector.__new__(Selector)
                    selector.policy, selector._density_search_mode = p, mode
                    with patch.object(search, 'select',
                                      side_effect=AssertionError('override bypassed')):
                        actual = selector._coverage_density_topk(q, 256, rows, None, None)
                    self.assert_equal(actual, expected)
                    self.assertTrue(calls)


@cuda
class CudaFusedDensitySearchTest(FusedDensitySearchTest):
    device = 'cuda'

    def test_fused_exhaustive_calibration_cache_and_counts_are_bitwise(self):
        p, q, rows = fixture(self.device)
        selector = Selector.__new__(Selector)
        selector.policy = p
        selector.config = dict(fisher_rank=70, adaptive_support='coverage')
        selector.payload = dict(support_scale=.1)
        kwargs = dict(target_rows=rows, target_demos=rows//128, cache=True)
        selector._density_search_mode = 'reference'
        reference = selector._adaptive_topk(q, **kwargs)
        selector._density_search_mode = 'fused'
        candidate = selector._adaptive_topk(q, **kwargs)
        self.assertEqual(selector._density_search.last['route'], 'fused_exhaustive')
        for count in (64, 256, 257, 512):
            for subset in (None, torch.tensor([1, 4], device=self.device)):
                self.assert_equal(candidate(count, subset), reference(count, subset))
        for scale in (.25, .1, .01):
            selector.payload['support_scale'] = scale
            expected, actual = selector._adaptive_select(reference), selector._adaptive_select(candidate)
            for a, b in zip(actual, expected):
                self.assert_equal(a, b)
        generator = torch.Generator().manual_seed(25)
        future = torch.randn(len(p.pf), 8, dtype=torch.float64,
                             generator=generator).to(self.device)
        p._action_neighborhood_features = lambda: future
        p.trajectory_of = p.hankel_trajectory
        p._queries = lambda query_rows: p.pf[query_rows]+.03
        anchors = torch.arange(0, 64*17, 17, device=self.device)
        selector._density_search_mode = 'reference'
        expected = selector._calibrate_coverage(anchors, anchors)
        selector._density_search_mode = 'fused'
        actual = selector._calibrate_coverage(anchors, anchors)
        self.assertEqual(actual, expected)

    def test_fused_exhaustive_preserves_gemm_shapes_masks_and_ties(self):
        for tied in (False, True):
            p, q, rows = fixture(self.device, ties=tied)
            search = FusedDensitySearch()
            allowed = torch.ones(len(q), len(p.pf), dtype=torch.bool, device=self.device)
            allowed[:, ::3] = False
            for row_mask, demos, mask in ((None, None, None), (rows, None, None),
                    (None, rows//128, allowed), (rows, rows//128, allowed)):
                with MatmulShapes() as before:
                    expected = p._hankel_topk(q, 256, row_mask, demos, mask)
                with MatmulShapes() as after:
                    actual = search.select(p, q, 256, row_mask, demos, mask)
                self.assert_equal(actual, expected)
                self.assertEqual(before.shapes, after.shapes)
                self.assertEqual(search.last['route'], 'fused_exhaustive')
            p.pf_norm[-1] = 0.  # Literal cached norms, including stale values.
            self.assert_equal(search.select(p, q, 256, rows), p._hankel_topk(q, 256, rows))

    def test_fused_signed_zero_and_nonfinite_fallback(self):
        from bpc.fast_cuda import extension
        module = extension(0, kernel='exact_density')
        self.assertIsNotNone(module)
        product = torch.tensor([[0., -0., 0., -0.], [-0., 0., -0., 0.]],
                               dtype=torch.float64, device=self.device)
        qnorm = product.new_tensor([0., -0.])
        bnorm = product.new_tensor([0., 0., -0., -0.])
        starts = torch.arange(4, device=self.device)
        empty_index = starts[:0]
        empty_mask = torch.empty(0, device=self.device, dtype=torch.bool)
        expected = product.clone().mul_(-2).add_(qnorm[:, None]).add_(bnorm).clamp_min_(0)
        actual = product.clone()
        module.postprocess(actual, actual, qnorm, bnorm, starts, empty_index,
            empty_index, starts, empty_index, empty_mask, empty_mask, 5, 0, True, True)
        self.assert_equal((actual,), (expected,))
        second = product.clone().neg_()
        expected = torch.minimum(expected, second.clone().mul_(-2).add_(qnorm[:, None]).add_(bnorm).clamp_min_(0))
        module.postprocess(actual, second, qnorm, bnorm, starts, empty_index,
            empty_index, starts, empty_index, empty_mask, empty_mask, 5, 0, False, True)
        self.assert_equal((actual,), (expected,))
        p, q, rows = fixture(self.device)
        search = FusedDensitySearch()
        for value in (math.nan, math.inf, -math.inf, 1e200):
            q[0, 0] = value
            self.assertIsNone(search.select(p, q, 256, rows))
        q[0, 0] = 0.
        p.pf_norm[0] = math.nan
        self.assertIsNone(search.select(p, q, 256, rows))

    def test_fused_extreme_timestamps_follow_torch_int64_wrap(self):
        from bpc.fast_cuda import extension
        module = extension(0, kernel='exact_density')
        self.assertIsNotNone(module)
        limits = torch.iinfo(torch.int64)
        starts = torch.tensor([limits.min, limits.min+3, -1, 0,
                               limits.max-3, limits.max], device=self.device)
        rows = starts.clone()
        first = starts-7
        valid = torch.ones_like(starts, dtype=torch.bool)
        product = torch.zeros(len(rows), len(starts), dtype=torch.float64, device=self.device)
        query_norm = product.new_ones(len(rows))
        bank_norm = product.new_ones(len(starts))
        empty_index = starts[:0]
        empty_mask = valid[:0]
        history, last_offset = 5, 7
        expected = product.clone().mul_(-2).add_(query_norm[:, None]).add_(bank_norm).clamp_min_(0)
        overlap = ((first[None] <= rows[:, None])
                   & (rows[:, None] <= (starts+last_offset)[None]))
        excluded = ((rows[:, None]-history <= starts[None])
                    & (starts[None] <= rows[:, None]))
        expected.masked_fill_(overlap | excluded, math.inf)
        actual = product.clone()
        module.postprocess(actual, actual, query_norm, bank_norm, starts, rows,
            empty_index, starts, first, valid, empty_mask, history, last_offset, True, True)
        self.assert_equal((actual,), (expected,))

    def test_unavailable_extension_replays_the_original_batch(self):
        p, q, rows = fixture(self.device)
        selector = Selector.__new__(Selector)
        selector.policy, selector._density_search_mode = p, 'fused'
        expected = p._hankel_topk(q, 256, rows)
        with patch('bpc.fused_density_search.extension', return_value=None), patch.object(
                p, '_hankel_topk', wraps=p._hankel_topk) as baseline:
            baseline.__func__ = Policy._hankel_topk  # Transparent instrumentation.
            actual = selector._coverage_density_topk(q, 256, rows, None, None)
        self.assert_equal(actual, expected)
        self.assertEqual(baseline.call_count, 1)
        self.assertIs(baseline.call_args.args[0], q)
        self.assertIs(baseline.call_args.args[2], rows)
        self.assertEqual(selector._density_search.last['reason'], 'extension_unavailable')


if __name__ == '__main__':
    unittest.main()
