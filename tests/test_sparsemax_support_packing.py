"""Batched packing retains literal sparsemax prefixes, values, and support order."""
import math
import unittest
from tests.suites import cuda

import torch

from bpc.sparse_qp import sparsemax, sparsemax_support


def literal_support(topk, width, scale):
    """Independent literal implementation from checkpoint 1344044."""
    if width < 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('positive sparsemax scale and bank width required')
    count = min(32, width)
    rows = None
    while True:
        columns, scores = topk(count, rows)
        if rows is None:
            rows = torch.arange(len(scores), device=scores.device)
            output = [None] * len(rows)
        probability = sparsemax(scores * scale)
        done = torch.ones_like(rows, dtype=torch.bool) if count == width else probability[:, -1] == 0
        for row, col, score, prob in zip(rows[done].tolist(), columns[done], scores[done], probability[done]):
            output[row] = (col[prob > 0], score[prob > 0])
        rows = rows[~done]
        if not len(rows):
            return output
        count = min(2 * count, width)


def evaluate(implementation, scores, scale, *, strided=False):
    calls = []
    def topk(count, rows):
        calls.append((count, None if rows is None else rows.tolist()))
        values, columns = (scores if rows is None else scores[rows]).topk(count, dim=1)
        if strided:
            # Large integer IDs and non-contiguous views forbid a mixed-dtype
            # packing shortcut that would silently round columns through FP32.
            columns = columns + 2**60
            padded_columns = columns.new_empty(len(values), 2*count)
            padded_scores = values.new_empty(len(values), 2*count)
            padded_columns[:, ::2], padded_scores[:, ::2] = columns, values
            columns, values = padded_columns[:, ::2], padded_scores[:, ::2]
        return columns, values
    return implementation(topk, scores.shape[1], scale), calls


class SparsemaxSupportPackingTest(unittest.TestCase):
    def assert_packed_equal(self, scores, scale, *, strided=False):
        expected, expected_calls = evaluate(literal_support, scores, scale, strided=strided)
        actual, calls = evaluate(sparsemax_support, scores, scale, strided=strided)
        self.assertEqual(calls, expected_calls)
        self.assertEqual(len(actual), len(expected))
        for current, original in zip(actual, expected):
            for left, right in zip(current, original):
                self.assertEqual(left.dtype, right.dtype)
                self.assertEqual(left.shape, right.shape)
                self.assertTrue(torch.equal(left.contiguous().view(torch.uint8),
                                            right.contiguous().view(torch.uint8)))
        return actual, calls

    def test_masks_ties_scales_and_small_or_large_banks(self):
        rng = torch.Generator().manual_seed(518)
        for dtype in (torch.float32, torch.float64):
            for width in (1, 17, 33, 95, 257, 513):
                scores = torch.randn(9, width, generator=rng, dtype=dtype)
                scores[0] = 0  # Full-bank ties force every expansion.
                scores[1, 1:] = -torch.inf  # Single eligible column.
                scores[2, max(1, width//3):] = -torch.inf
                scores[3] = scores[3].round()  # Repeated finite scores.
                for scale in (.001, 1., 1000.):
                    with self.subTest(dtype=dtype, width=width, scale=scale):
                        self.assert_packed_equal(scores, scale)

    def test_expansion_keeps_unresolved_row_ids_and_original_tie_order(self):
        scores = torch.zeros(5, 257, dtype=torch.float64)
        scores[0, 1:] = scores[3, 1:] = -100
        scores[1, 40:] = -100
        scores[4, 100:] = -100
        selected, calls = self.assert_packed_equal(scores, 1.)
        self.assertEqual([len(columns) for columns, _ in selected], [1, 40, 257, 1, 100])
        # The 256-column call completes no rows; the final tied row must expand.
        self.assertEqual(calls, [(32, None), (64, [1, 2, 4]), (128, [2, 4]),
                                 (256, [2]), (257, [2])])

    @cuda
    def test_cuda_literal_contents_order_and_expansion(self):
        rng = torch.Generator().manual_seed(851)
        for dtype in (torch.float32, torch.float64):
            for width in (1, 33, 257, 513):
                scores = torch.randn(48, width, generator=rng, dtype=dtype).cuda()
                scores[0] = 0
                scores[1, 1:] = -torch.inf
                scores[2, max(1, width//3):] = -torch.inf
                for scale in (.001, 1., 1000.):
                    self.assert_packed_equal(scores, scale, strided=True)
                self.assert_packed_equal(scores[:0], 1.)

    def test_strided_outputs_large_column_ids_and_signed_zero_scores(self):
        scores = torch.tensor([[0., -0., -.1, -.2], [0., -1., -2., -torch.inf]], dtype=torch.float64)
        self.assert_packed_equal(scores, 1., strided=True)

    def test_empty_query_batch_retains_empty_list_and_prefix_request(self):
        for width in (1, 33, 257):
            selected, calls = self.assert_packed_equal(torch.empty(0, width), 1.)
            self.assertEqual(selected, [])
            self.assertEqual(calls, [(min(32, width), None)])

    def test_invalid_configuration_and_nonfinite_rows_preserve_errors(self):
        for width, scale in ((0, 1.), (-1, 1.), (3, 0.), (3, -1.), (3, math.nan), (3, math.inf)):
            def unused(*args):
                raise AssertionError('invalid arguments must fail before retrieval')
            errors = []
            for implementation in (literal_support, sparsemax_support):
                with self.assertRaises(ValueError) as caught:
                    implementation(unused, width, scale)
                errors.append(str(caught.exception))
            self.assertEqual(*errors)
        for scores in (torch.tensor([[-torch.inf, -torch.inf]]), torch.tensor([[torch.nan, 0.]]),
                       torch.tensor([[torch.inf, 0.]]), torch.tensor([[1e30, 0.]])):
            errors = []
            for implementation in (literal_support, sparsemax_support):
                with self.assertRaises(ValueError) as caught:
                    evaluate(implementation, scores, 1e30)
                errors.append(str(caught.exception))
            self.assertEqual(*errors)

    def test_batched_packing_removes_per_query_nonzero_scans(self):
        scores = torch.arange(17, dtype=torch.float64).expand(12, -1)*.01
        counts = []
        for implementation in (literal_support, sparsemax_support):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                evaluate(implementation, scores, 1.)
            counts.append(sum(event.count for event in profiler.key_averages() if event.key == 'aten::nonzero'))
        self.assertGreater(counts[0], len(scores))
        self.assertLessEqual(counts[1], 2)


if __name__ == '__main__':
    unittest.main()
