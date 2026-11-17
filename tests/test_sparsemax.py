import unittest
import torch
from bpc.sparse_qp import sparsemax, sparsemax_support


class SparsemaxTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)

    def test_prefix_support_matches_full_bank(self):
        for width in (1, 17, 95, 257):
            scores = torch.randn(4, width, dtype=torch.float64)
            scores[0] = 0
            scores[1, 1:] = -torch.inf
            for scale in (.001, 1., 1000.):
                calls = []
                def topk(count, rows):
                    calls.append(count)
                    values, columns = (scores if rows is None else scores[rows]).topk(count, dim=1)
                    return columns, values
                actual = sparsemax_support(topk, width, scale)
                expected = sparsemax(scores*scale)
                for i, (columns, values) in enumerate(actual):
                    torch.testing.assert_close(columns.sort().values,
                        (expected[i] > 0).nonzero().flatten(), rtol=0, atol=0)
                    torch.testing.assert_close(values, scores[i, columns], rtol=0, atol=0)
                self.assertLessEqual(max(calls), width)
        with self.assertRaises(ValueError):
            sparsemax_support(topk, width, 0)

    def test_prefix_expands_only_unresolved_histories(self):
        scores = torch.zeros(3, 257, dtype=torch.float64)
        scores[0, 1:] = -100
        scores[1, 40:] = -100
        calls = []
        def topk(count, rows):
            calls.append(None if rows is None else rows.tolist())
            values, columns = (scores if rows is None else scores[rows]).topk(count, dim=1)
            return columns, values
        result = sparsemax_support(topk, 257, 1.)
        self.assertEqual([len(columns) for columns, _ in result], [1, 40, 257])
        self.assertEqual(calls[:3], [None, [1, 2], [2]])

    def test_projection_ties_exclusions_and_adaptive_support(self):
        for count in (1, 17, 95, 257):
            scores = torch.randn(4, count, dtype=torch.float64) * .001
            scores[0] = 0
            if count > 1:
                scores[1, count // 2:] = -torch.inf
            values = scores.sort(descending=True).values
            sums = values.cumsum(-1)
            ranks = torch.arange(1, count + 1)
            active = (1 + ranks * values > sums).sum(-1, keepdim=True)
            threshold = (sums.gather(-1, active - 1) - 1) / active
            actual = sparsemax(scores)
            torch.testing.assert_close(actual, (scores - threshold).clamp_min(0))
            torch.testing.assert_close(actual.sum(1), torch.ones(4, dtype=scores.dtype))
            self.assertEqual(actual[~torch.isfinite(scores)].sum().item(), 0)
        for scores in [torch.tensor([[-torch.inf]]), torch.tensor([[torch.inf, 0.]]),
                       torch.tensor([[torch.nan, 0.]])]:
            with self.assertRaises(ValueError):
                sparsemax(scores)
