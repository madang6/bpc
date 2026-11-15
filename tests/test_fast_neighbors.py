import unittest
from tests.suites import cuda

import torch


from dpc_nldl.fast_neighbors import CertifiedTopK


class CertifiedTopKTest(unittest.TestCase):
    def test_compact_and_demo_exclusions_match_literal_at_boundary(self):
        starts = torch.cat([torch.arange(20 * demo, 20 * demo + 16)
                            for demo in range(4)])
        trajectories = torch.arange(4).repeat_interleave(16)
        bank = (10 * starts).to(torch.float64)[:, None]
        queries = torch.tensor([[232.25]], dtype=torch.float64)
        row = torch.tensor([23])
        for demos, expected_removed in ((None, 4), (torch.tensor([1]), 16)):
            with self.subTest(leave_demo_out=demos is not None):
                actual, certified, _, removed = CertifiedTopK(bank).select(
                    queries, 3, row, demos, starts, trajectories, 3,
                    prefixes=(8, 16, 32))
                excluded = ((row[:, None] - 3 <= starts)
                            & (starts <= row[:, None]))
                if demos is not None:
                    excluded |= trajectories == demos[:, None]
                distance = (bank.T - queries).square()
                distance.masked_fill_(excluded, torch.inf)
                expected = torch.topk(
                    distance, 3, largest=False, sorted=True).indices
                self.assertTrue(bool(certified.all()))
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(removed.tolist(), [expected_removed])

    def test_certified_prefix_matches_ordered_fp64_topk(self):
        bank = torch.arange(64, dtype=torch.float64)[:, None].mul(10)
        queries = torch.tensor([[132.25], [417.75]], dtype=torch.float64)
        starts = torch.arange(len(bank))
        rows = torch.tensor([13, -1])
        actual, certified, _, removed = CertifiedTopK(bank).select(
            queries, 3, rows, None, starts, torch.zeros_like(starts), 0,
            prefixes=(8, 16, 32))
        excluded = ((rows[:, None] <= starts) & (starts <= rows[:, None]))
        distance = (queries.square().sum(1)[:, None] + bank.square().sum(1)
                    - 2 * (queries @ bank.T)).clamp_min(0)
        distance.masked_fill_(excluded, torch.inf)
        expected = torch.topk(distance, 3, largest=False, sorted=True).indices
        self.assertTrue(bool(certified.all()))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(removed.tolist(), [1, 0])

    def test_query_perturbation_certifies_reference_topk(self):
        bank = torch.arange(64, dtype=torch.float64)[:, None].mul(10)
        reference = torch.tensor([[132.25]], dtype=torch.float64)
        query = reference + 1e-9
        starts = torch.arange(len(bank))
        actual, certified, _, _ = CertifiedTopK(bank).select(
            query, 3, torch.tensor([-1]), None, starts, torch.zeros_like(starts), 0,
            prefixes=(8, 16, 32),
            reference_error=torch.linalg.vector_norm(query - reference, dim=1))
        expected = torch.topk(
            (bank.T - reference).square(), 3, largest=False).indices
        self.assertTrue(bool(certified.all()))
        self.assertTrue(torch.equal(actual, expected))

    def test_tied_order_fails_closed(self):
        bank = torch.tensor([[-1.0], [1.0], [3.0], [6.0], [10.0], [15.0]],
                            dtype=torch.float64)
        actual, certified, _, _ = CertifiedTopK(bank).select(
            torch.zeros(1, 1, dtype=torch.float64), 2,
            torch.tensor([-1]), None, torch.arange(len(bank)),
            torch.zeros(len(bank), dtype=torch.long), 0, prefixes=(4,))
        self.assertFalse(bool(certified.any()))
        self.assertTrue(bool((actual == -1).all()))

        actual, certified, _, _ = CertifiedTopK(bank).select(
            torch.zeros(1, 1, dtype=torch.float64), 2,
            torch.tensor([-1]), None, torch.arange(len(bank)),
            torch.zeros(len(bank), dtype=torch.long), 0, prefixes=(4,),
            ordered_contract=False)
        self.assertTrue(bool(certified.all()))
        self.assertEqual(set(actual[0].tolist()), {0, 1})

    def test_nonfinite_bank_fails_closed(self):
        bank = torch.arange(12, dtype=torch.float64)[:, None]
        bank[-1] = torch.inf
        actual, certified, _, _ = CertifiedTopK(bank).select(
            torch.tensor([[2.25]], dtype=torch.float64), 2,
            torch.tensor([-1]), None, torch.arange(len(bank)),
            torch.zeros(len(bank), dtype=torch.long), 0, prefixes=(4, 8))
        self.assertFalse(bool(certified.any()))
        self.assertTrue(bool((actual == -1).all()))

    @cuda
    def test_cuda_k200_matches_ordered_fp64_topk(self):
        bank = torch.arange(2048, device="cuda", dtype=torch.float64)[:, None]
        bank = torch.cat([10 * bank, bank.remainder(7)], 1)
        queries = torch.tensor(
            [[-7.25, 0.5], [-13.5, 1.25]], device="cuda", dtype=torch.float64)
        starts = torch.arange(len(bank), device="cuda")
        rows = torch.tensor([10, -1], device="cuda")
        actual, certified, _, removed = CertifiedTopK(bank).select(
            queries, 200, rows, None, starts, torch.zeros_like(starts), 10,
            prefixes=(256, 384, 512, 1024))
        excluded = ((rows[:, None] - 10 <= starts) & (starts <= rows[:, None]))
        distance = (queries.square().sum(1)[:, None] + bank.square().sum(1)
                    - 2 * (queries @ bank.T)).clamp_min_(0)
        distance.masked_fill_(excluded, torch.inf)
        expected = torch.topk(distance, 200, largest=False, sorted=True).indices
        self.assertTrue(bool(certified.all()))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(removed.tolist(), [11, 0])


if __name__ == "__main__":
    unittest.main()
