"""Same-shape rate reuse against the literal pre-optimization target path."""
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch

from tests.support.readout import offline_fixture as fixture


def literal_fixed_g(p, supports, rows):
    """Checkpoint87c1e94 fixed-support solve, without any shared rate cache."""
    batch, count = supports.shape
    tile = p._g_microbatch(count)
    grams, rhs, coefficients = [], [], []
    for start in range(0, batch, tile):
        stop = min(start+tile, batch)
        selected = rows[start:stop]
        query = p._queries(selected)
        design = p._reconstruction_columns(
            supports[start:stop], query, torch.float32, selected).transpose(1, 2)
        query = query.float()
        if p.g_solver == 'exact':
            coefficients.append(p._solve_affine(design, query))
            continue
        transpose = design.transpose(1, 2)
        grams.append(transpose@design)
        rhs.append((transpose@query.unsqueeze(2)).squeeze(2))
    return (torch.cat(coefficients) if coefficients else p._solve_g_normal(
        torch.cat(grams), torch.cat(rhs)))


def literal_prior(p, plane, supports, queries, rows=None, rates=None):
    """Checkpoint87c1e94 prior and fixed solve, including explicit future rates."""
    futures = p._future_columns(
        supports, p._queries(rows) if p.g_align_future and rows is not None else queries, rows, rates)
    if p.g_weights == 'uniform':
        coefficients = queries.new_full(supports.shape, 1/supports.shape[1])
        prior = futures[:, 0].clone()
        prior[:, :p.action_dim] = futures[:, :, :p.action_dim].mean(1)
        return prior, coefficients
    if plane is None:
        coefficients = literal_fixed_g(p, supports, rows)
        return torch.einsum('bkh,bk->bh', futures.float(), coefficients).to(p.dtype), coefficients
    futures = futures.to(plane.dtype)
    design = plane.transpose(1, 2)
    if p.g_solver == 'exact':
        coefficients = p._solve_affine(design, queries)
    else:
        queries = queries.to(design.dtype)
        coefficients = p._solve_g_normal(design.transpose(1, 2)@design,
            (design.transpose(1, 2)@queries[:, :, None]).squeeze(2)).to(futures.dtype)
    return torch.einsum('bkh,bk->bh', futures, coefficients).to(p.dtype), coefficients


def capture_rates(policy, callback):
    original, seen = policy._rate_trace, []
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(result.get('hankel_rates', torch.empty(0)).clone())
        return result
    with patch.object(policy, '_rate_trace', side_effect=record):
        result = callback()
    return result, seen


class OfflineRateReuseTests(unittest.TestCase):
    def check_single_microbatch(self, device='cpu'):
        policy, rows, query, supports = fixture(device)
        self.assertGreaterEqual(policy._g_microbatch(supports.shape[1]), len(rows))
        expected, legacy_rates = capture_rates(policy, lambda: literal_prior(
            policy, None, supports, query.float(), rows=rows))
        actual, shared_rates = capture_rates(policy, lambda: policy._g_prior(
            None, supports, query.float(), rows=rows))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(len(legacy_rates), 2)
        self.assertEqual(len(shared_rates), 1)
        self.assertTrue(bool((shared_rates[0] != 1).any()))
        for rates in legacy_rates:
            torch.testing.assert_close(rates, shared_rates[0], rtol=0, atol=0)

    def test_single_microbatch_targets_coefficients_and_rates_are_bitwise_identical(self):
        self.check_single_microbatch()

    @cuda
    def test_cuda_single_microbatch_is_bitwise_identical(self):
        self.check_single_microbatch(device='cuda')

    def test_split_microbatches_keep_each_original_rate_projection(self):
        policy, rows, query, supports = fixture()
        with patch.object(policy, '_g_microbatch', return_value=1):
            expected, legacy_rates = capture_rates(policy, lambda: literal_prior(
                policy, None, supports, query.float(), rows=rows))
            actual, rates = capture_rates(policy, lambda: policy._g_prior(
                None, supports, query.float(), rows=rows))
        self.assertEqual(len(rates), 4)
        for left, right in zip(actual+tuple(rates), expected+tuple(legacy_rates)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_explicit_future_rates_do_not_override_original_history_rates(self):
        policy, rows, query, supports = fixture()
        supplied = torch.ones(supports.shape, dtype=query.dtype)
        expected, old_rates = capture_rates(policy, lambda: literal_prior(
            policy, None, supports, query.float(), rows=rows, rates=supplied))
        actual, rates = capture_rates(policy, lambda: policy._g_prior(
            None, supports, query.float(), rows=rows, rates=supplied))
        self.assertEqual(len(rates), 1)
        for left, right in zip(actual+tuple(rates), expected+tuple(old_rates)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_native_only_and_uniform_priors_add_no_unused_rate_projections(self):
        for rates, weights, calls in (((1.,), 'optimized', 0), ((.5, 1., 2.), 'uniform', 1)):
            policy, rows, query, supports = fixture(rates=rates)
            policy.g_weights = weights
            expected = literal_prior(policy, None, supports, query.float(), rows=rows)
            actual, recorded = capture_rates(policy, lambda: policy._g_prior(
                None, supports, query.float(), rows=rows))
            self.assertEqual(len(recorded), calls)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_materialized_offline_targets_share_rates_and_preserve_exclusions(self):
        policy, rows, query, supports = fixture()
        excluded = torch.searchsorted(policy.hankel_starts, rows, right=True)-torch.searchsorted(
            policy.hankel_starts, rows-policy.history)
        def reference():
            plane = policy._reconstruction_columns(supports, query, policy.g_fit_dtype, rows)
            return literal_prior(policy, plane, supports, query.float(), rows=rows)[0]
        expected, old_rates = capture_rates(policy, reference)
        actual, rates = capture_rates(policy, lambda: policy._offline_g(
            rows, atlas=(supports, excluded)))
        torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], excluded, rtol=0, atol=0)
        torch.testing.assert_close(actual[3], policy._support_endpoints(supports), rtol=0, atol=0)
        self.assertEqual(len(old_rates), 2)
        self.assertEqual(len(rates), 1)
        for old in old_rates:
            torch.testing.assert_close(old, rates[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
