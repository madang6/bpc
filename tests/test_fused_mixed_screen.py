"""Fused FP32 screening respects eligibility and its conservative error bound."""
import math
import unittest
from tests.suites import cuda

import torch

from dpc_nldl.fast_cuda import extension
from dpc_nldl.fast_neighbors import CertifiedTopK, _BankBounds, _error_bounds
from dpc_nldl.fused_mixed_screen import screen
from tests.support.search import density_fixture as fixture


@cuda
class FusedMixedScreenTest(unittest.TestCase):
    def test_temporal_masks_and_distances_are_bounded(self):
        p, q, rows = fixture('cuda')
        rates = p._interpolated_keys()
        screens = [CertifiedTopK(bank) for bank in [p.pf]+[x[1] for x in rates]]
        bounds = _BankBounds(*(max(getattr(item.bounds, name) for item in screens)
                              for name in ('original', 'cast', 'cast_error')))
        error, _ = _error_bounds(q, q.float(), bounds)
        allowed = torch.ones(len(q), len(p.pf), device='cuda', dtype=torch.bool)
        allowed[:, ::3] = False
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for query_rows, demos, permitted in ((None, None, None), (rows, None, None),
                    (None, rows//128, allowed), (rows, rows//128, allowed)):
                coarse, finite = screen(p, q.float(), screens, rates, query_rows, demos, permitted)
                self.assertTrue(finite.all())
                expected = None
                for index, cached in enumerate(screens):
                    value = (q.float() @ cached.bank32.T).mul_(-2)
                    value.add_(cached.norm32).add_(q.float().square().sum(1)[:, None]).clamp_min_(0)
                    if index:
                        rate, _, _, first, valid = rates[index-1]
                        mask = ~valid
                        if query_rows is not None:
                            mask = mask | p._rate_target_overlap(first, p.hankel_starts, rate, query_rows)
                        value.masked_fill_(mask, math.inf)
                    expected = value if expected is None else torch.minimum(expected, value)
                if query_rows is not None or demos is not None:
                    expected.masked_fill_(p._hankel_exclusion(p.hankel_starts, query_rows, demos), math.inf)
                if permitted is not None:
                    expected.masked_fill_(~permitted, math.inf)
                self.assertTrue(torch.equal(coarse, expected))
                columns, distances, _ = p._hankel_topk(q, len(p.pf), query_rows, demos, permitted)
                selected = coarse.gather(1, columns).double()
                difference = (selected-distances.square()).abs()
                eligible = torch.isfinite(distances)
                self.assertTrue((difference[eligible] <= error[:, None].expand_as(difference)[eligible]).all())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    def test_nonfinite_arithmetic_flag_survives_eligibility_masks(self):
        p, q, _ = fixture('cuda')
        rates = p._interpolated_keys()
        screens = [CertifiedTopK(bank) for bank in [p.pf]+[x[1] for x in rates]]
        q[3, 4] = math.nan
        allowed = torch.zeros(len(q), len(p.pf), device='cuda', dtype=torch.bool)
        coarse, finite = screen(p, q.float(), screens, rates, allowed=allowed)
        self.assertTrue(torch.isinf(coarse).all())
        expected = torch.ones_like(finite)
        expected[3] = False
        self.assertTrue(torch.equal(finite, expected))

    def test_kernel_rejects_wrong_precision(self):
        module = extension(0, kernel='mixed_density')
        self.assertIsNotNone(module)
        product = torch.zeros(2, 8, device='cuda', dtype=torch.float64)
        starts = torch.arange(8, device='cuda')
        empty = starts[:0]
        mask = torch.empty(0, device='cuda', dtype=torch.bool)
        invalid = torch.zeros(2, device='cuda', dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, 'FP32'):
            module.postprocess(product, product, product[:, 0].contiguous(), product[0],
                starts, empty, empty, starts, empty, mask, mask, invalid, 5, 0, True, True)


if __name__ == '__main__':
    unittest.main()
