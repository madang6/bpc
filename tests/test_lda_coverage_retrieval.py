"""Exact support and request-reuse checks for density-normalized retrieval."""
import unittest
from tests.suites import cuda, integration
from types import SimpleNamespace
from unittest.mock import patch

import torch

from bpc.lda_selector import Selector
from bpc.sparse_qp import sparsemax


def reference_topk(selector, queries, target_rows=None, target_demos=None, allowed=None):
    """The checkpoint implementation, including each original query batch."""
    policy = selector.policy
    _, values, _ = policy._hankel_topk(queries,
        min(4*policy.g_supports, len(policy.hankel_starts)),
        target_rows, target_demos, allowed)
    farthest = values.masked_fill(~torch.isfinite(values), 0).amax(1)
    spread = (farthest.square()-values[:, 0].square()).clamp_min(1e-12)

    def topk(count, rows):
        batch = [x if rows is None or x is None else x[rows]
                 for x in (queries, target_rows, target_demos, allowed)]
        columns, values, _ = policy._hankel_topk(batch[0], count, *batch[1:])
        return columns, -values.square()/(spread if rows is None else spread[rows])[:, None]

    return topk


def reference_calibration(selector, anchors, rows):
    """Checkpoint calibration: independent reference and all nine searches."""
    policy = selector.policy
    future = policy._action_neighborhood_features().double()
    scales = [.25*2**(-i/4) for i in range(9)]
    mass, reference = {scale: 0. for scale in scales}, 0.
    for batch, query_rows in zip(anchors.split(16), rows.split(16)):
        distance = torch.cdist(future[batch], future).square()
        distance.masked_fill_(policy.hankel_trajectory[batch, None]
                             == policy.hankel_trajectory[None], torch.inf)
        nearest = distance.topk(min(4*policy.g_supports, distance.shape[1]), largest=False).values
        radius = nearest.masked_fill(~torch.isfinite(nearest), 0).amax(1).clamp_min(1e-12)
        teacher = sparsemax(-distance/radius[:, None])
        queries, demos = policy._queries(query_rows), policy.trajectory_of[query_rows]
        columns, _, _ = policy._hankel_topk(queries, policy.g_supports, query_rows, demos)
        reference += float(teacher.gather(1, columns).sum())/len(rows)
        for scale in scales:
            selector.payload['support_scale'] = scale
            selected = selector._adaptive_select(reference_topk(selector, queries, query_rows, demos))
            mass[scale] += sum(float(teacher[i, col].sum())
                              for i, (col, _) in enumerate(selected))/len(rows)
    feasible = [scale for scale in scales if mass[scale] >= reference]
    return dict(support_scale=max(feasible), support_calibration_histories=len(rows),
                support_calibration_mass=mass, support_reference_mass=reference)


class CoverageRetrievalTest(unittest.TestCase):
    def selector(self, tied=False, width=1027, device='cpu'):
        rank = torch.arange(1, width+1, dtype=torch.float64, device=device)
        distance = torch.stack([rank**power for power in (.5, 1., 2.)])
        if tied:
            distance = distance.floor()
        generator = torch.Generator(device=device).manual_seed(72)
        distance = distance[:, torch.randperm(width, generator=generator, device=device)]
        calls = []

        def topk(queries, count, target_rows=None, target_demos=None, allowed=None):
            index = queries[:, 0].long()
            calls.append((count, tuple(index.tolist())))
            local = distance[index].clone()
            if allowed is not None:
                local.masked_fill_(~allowed, torch.inf)
            # Distinct exclusion masks check that no cached result crosses
            # query rows or leaks a held-out column back into the support.
            if target_rows is not None:
                local.scatter_(1, target_rows[:, None], torch.inf)
            if target_demos is not None:
                local.scatter_(1, (target_demos+3)[:, None], torch.inf)
            values, columns = local.topk(count, largest=False)
            return columns, values, torch.zeros(len(index), dtype=torch.long, device=device)

        selector = Selector.__new__(Selector)
        selector.config = dict(fisher_rank=70, adaptive_support='coverage')
        selector.payload = dict(support_scale=.1)
        selector.policy = SimpleNamespace(g_supports=64,
            hankel_starts=torch.arange(width, device=device), pf=True, _hankel_topk=topk)
        return selector, torch.arange(3, dtype=torch.float64, device=device)[:, None], calls

    def assert_matches(self, actual, expected, *, atol=0):
        self.assertEqual(len(actual), len(expected))
        for (columns, scores), (reference, values) in zip(actual, expected):
            torch.testing.assert_close(columns, reference, rtol=0, atol=0)
            torch.testing.assert_close(scores, values, rtol=0, atol=atol)

    def test_unique_density_prefix_replaces_repeated_full_bank_searches(self):
        selector, queries, calls = self.selector()
        expected = selector._adaptive_select(reference_topk(selector, queries))
        reference_calls = list(calls)
        calls.clear()
        actual = selector.select(queries)
        self.assert_matches(actual, expected)
        self.assertGreater(len(reference_calls), 2)
        self.assertEqual(calls, [(256, (0, 1, 2))])

    def test_tied_prefix_preserves_original_topk_calls_and_order(self):
        for limited in (False, True):
            with self.subTest(limited=limited):
                selector, queries, calls = self.selector(tied=True)
                allowed = None
                if limited:
                    allowed = torch.zeros(3, 1027, dtype=torch.bool)
                    allowed[:, :21] = True  # Repeated excluded infinities.
                kwargs = dict(target_rows=torch.tensor([7, 9, 8]),
                              target_demos=torch.tensor([0, 1, 2]), allowed=allowed)
                expected = selector._adaptive_select(reference_topk(selector, queries, **kwargs))
                reference_calls = list(calls)
                calls.clear()
                actual = selector.select(queries, **kwargs)
                self.assert_matches(actual, expected)
                self.assertEqual(calls, reference_calls)

    def test_calibration_reuses_only_identical_requests_across_scales(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                selector, queries, calls = self.selector(tied=tied)
                kwargs = dict(target_rows=torch.tensor([7, 9, 8]),
                              target_demos=torch.tensor([0, 1, 2]))
                scales = (.1, .01, .001, .0001)
                expected = []
                for scale in scales:
                    selector.payload['support_scale'] = scale
                    expected.append(selector._adaptive_select(reference_topk(selector, queries, **kwargs)))
                reference_calls = list(calls)
                calls.clear()
                provider = selector._adaptive_topk(queries, **kwargs, cache=True)
                for scale, selected in zip(scales, expected):
                    selector.payload['support_scale'] = scale
                    self.assert_matches(selector._adaptive_select(provider), selected)
                # The separate density prefix is deliberately not merged with
                # a tied count=256 request: its tie order may differ by count.
                self.assertEqual(len(calls[1:]), len(set(calls[1:])))
                self.assertLess(len(calls), len(reference_calls))
                self.assertTrue(any(count > 256 for count, _ in calls))
                self.assertTrue(any(len(rows) < 3 for _, rows in reference_calls))

    @integration
    def test_real_temporal_bank_matches_checkpoint_and_full_sparsemax(self):
        self.check_real_temporal_bank('cpu')

    def check_real_temporal_bank(self, device):
        from bpc.bpc import Policy
        from tests.support.policies import fisher_recipe as recipe
        from tests.support.policies import data_config as _data_config, trajectories as _trajectories

        method = recipe()
        method['device'] = device
        method['g'].update(lookup='lda', lda=dict(features=24, anchors=32,
            shrinkage=1., sigma_scale=.5, seed=9, fisher_rank=6, teacher='sparsemax', adaptive_support='coverage'))
        policy = Policy(_trajectories(demos=4, length=32), _data_config(), method,
                        fit_residuals=False)
        policy.fit_g_lookup()
        policy._project_hankel()
        selector = policy.support_selector
        selector.config['adaptive_support'] = 'coverage'
        indices = torch.tensor([1, 17, 37], device=device)
        queries = policy._past_columns(indices)
        rows = policy.hankel_starts[indices]+policy.history
        allowed = torch.ones(len(rows), len(policy.hankel_starts), dtype=torch.bool, device=device)
        allowed[:, 3:9] = False
        for mask in (None, allowed):
            kwargs = dict(target_rows=rows, allowed=mask)
            columns, distances, _ = policy._hankel_topk(queries,
                len(policy.hankel_starts), rows, allowed=mask)
            local = distances[:, :min(4*policy.g_supports, len(policy.hankel_starts))]
            spread = (local.masked_fill(~torch.isfinite(local), 0).amax(1).square()
                      - local[:, 0].square()).clamp_min(1e-12)
            for scale in (.001, .1, 10.):
                selector.payload['support_scale'] = scale
                expected = selector._adaptive_select(reference_topk(selector, queries, **kwargs))
                actual = selector.select(queries, **kwargs)
                self.assert_matches(actual, expected, atol=1e-10)
                probability = sparsemax(-scale*distances.square()/spread[:, None])
                for i, (selected, _) in enumerate(actual):
                    torch.testing.assert_close(selected.sort().values,
                        columns[i, probability[i] > 0].sort().values, rtol=0, atol=0)

    @integration
    def test_coverage_calibration_preserves_mass_grid_and_scale(self):
        self.check_calibration('cpu')

    def check_calibration(self, device):
        from bpc.bpc import Policy
        from tests.support.policies import fisher_recipe as recipe
        from tests.support.policies import data_config as _data_config, trajectories as _trajectories

        method = recipe()
        method['device'] = device
        method['g'].update(lookup='lda', lda=dict(features=24, anchors=32,
            shrinkage=1., sigma_scale=.5, seed=9, fisher_rank=6, teacher='sparsemax', adaptive_support='coverage'))
        policy = Policy(_trajectories(demos=4, length=32), _data_config(), method,
                        fit_residuals=False)
        policy.fit_g_lookup()
        policy._project_hankel()
        selector = policy.support_selector
        selector.config['adaptive_support'] = 'coverage'
        anchors = torch.randperm(len(policy.hankel_starts),
            generator=torch.Generator(device=device).manual_seed(9), device=device)[:20]
        rows = policy.hankel_starts[anchors]+policy.history
        original = policy._hankel_topk
        with patch.object(policy, '_hankel_topk', wraps=original) as calls:
            actual = selector._calibrate_coverage(anchors, rows)
            actual_calls = calls.call_count
        with patch.object(policy, '_hankel_topk', wraps=original) as calls:
            expected = reference_calibration(selector, anchors, rows)
            reference_calls = calls.call_count
        self.assertEqual(actual, expected)
        self.assertLess(actual_calls, reference_calls)

    @cuda
    def test_cuda_prefixes_subsets_and_ties_match_checkpoint(self):
        for tied in (False, True):
            selector, queries, calls = self.selector(tied=tied, device='cuda')
            allowed = torch.ones(3, 1027, device='cuda', dtype=torch.bool)
            allowed[:, 3:9] = False
            kwargs = dict(target_rows=torch.tensor([7, 9, 8], device='cuda'),
                          target_demos=torch.tensor([0, 1, 2], device='cuda'), allowed=allowed)
            provider = selector._adaptive_topk(queries, **kwargs, cache=True)
            for scale in (.1, .01, .001, .0001):
                selector.payload['support_scale'] = scale
                expected = selector._adaptive_select(reference_topk(selector, queries, **kwargs))
                self.assert_matches(selector._adaptive_select(provider), expected)
                self.assert_matches(selector.select(queries, **kwargs), expected)

    @cuda
    def test_cuda_real_temporal_bank_matches_checkpoint_and_full_sparsemax(self):
        self.check_real_temporal_bank('cuda')

    @cuda
    def test_cuda_coverage_calibration_preserves_mass_grid_and_scale(self):
        self.check_calibration('cuda')


if __name__ == '__main__':
    unittest.main()
