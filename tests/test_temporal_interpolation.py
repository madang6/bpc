import copy
import unittest

import torch

from dpc_nldl.artifacts import _target_key
from dpc_nldl.bpc import Policy
from dpc_nldl.pipeline import fit_groups
from tests.support.policies import (
    data_config, interpolation_policy, method_config, toy_policy, trajectories,
)


class TemporalInterpolationTest(unittest.TestCase):
    def test_affine_offset_interpolation_preserves_causal_endpoint(self):
        p = interpolation_policy([.5, .75, 1., 1.5, 2.], offset=1)
        columns = torch.tensor([4, len(p.hankel_starts) - 1])
        endpoint = p.hankel_starts[columns] + p.history - 1
        for rate in p.g_lookup_rates:
            plane = p._past_columns(columns, rate=rate)
            positions = endpoint[:, None] + rate * (torch.arange(p.history).double() - p.history + 1)
            lo, hi = positions.floor().long(), positions.ceil().long()
            w = (positions - lo)[..., None]
            expected = torch.cat([
                (source[lo + shift] + w * (source[hi + shift] - source[lo + shift])).flatten(1)
                for source, shift in ((p.action, 0), (p.state, 1))], 1)
            torch.testing.assert_close(plane, expected, rtol=0, atol=0)
            torch.testing.assert_close(plane[:, -p.state_dim:], p.state[endpoint + 1])
        p.g_align_history = True
        query = p._past_columns(columns, rate=.75)
        torch.testing.assert_close(p._reconstruction_columns(columns[:, None], query)[:, 0], query)
        future_rows = endpoint[:, None] + 1 + torch.arange(p.horizon)
        torch.testing.assert_close(p.uf.T[columns], p.action[future_rows].flatten(1))

    def test_aligned_reconstruction_uses_selected_rate_and_keeps_futures(self):
        policy = interpolation_policy([.5, 1., 2.])
        columns = torch.tensor([[4], [16]])
        query = policy._past_columns(columns[:, 0], rate=.5)
        original = policy._reconstruction_columns(columns, query)
        futures = policy.uf.clone()
        policy.g_align_history = True
        aligned = policy._reconstruction_columns(columns, query)
        torch.testing.assert_close(aligned[:, 0], query)
        self.assertGreater(float((original[:, 0] - query).norm()), 0)
        self.assertTrue(torch.equal(futures, policy.uf))

    def test_alignment_excludes_leaky_rate_in_fitting(self):
        policy = interpolation_policy([1., 2.])
        policy.g_align_history = True
        columns = torch.tensor([[4]])
        query = policy._past_columns(columns[:, 0], rate=2)
        rows = policy.hankel_starts[columns[:, 0]] - 2
        torch.testing.assert_close(policy._reconstruction_columns(columns, query)[:, 0], query)
        torch.testing.assert_close(
            policy._reconstruction_columns(columns, query, target_rows=rows),
            policy._past_columns(columns))

    def test_fixed_affine_solve_uses_aligned_columns(self):
        policy = interpolation_policy([.5, 1., 2.])
        policy.g_align_history = True
        policy.g_solver = "exact"
        policy.g_constraint = "affine"
        policy.g_l1 = 0
        rows = torch.tensor([5, 19])
        queries = policy._queries(rows)
        supports, _, _ = policy._hankel_topk(queries, 4, target_rows=rows)
        plane = policy._reconstruction_columns(supports, queries, torch.float32, rows)
        expected = policy._solve_affine(plane.transpose(1, 2), queries.float())
        torch.testing.assert_close(policy._fixed_g(supports, rows), expected)

    def test_rate_one_is_exactly_the_existing_policy(self):
        original, explicit = interpolation_policy(), interpolation_policy([1.0])
        columns = torch.tensor([[2, 4], [12, 15]])
        self.assertTrue(torch.equal(original._past_columns(columns),
                                    explicit._past_columns(columns, rate=1)))
        queries = original._past_columns(columns[:, 0]) + .13
        for left, right in zip(original._hankel_topk(queries, 5),
                               explicit._hankel_topk(queries, 5)):
            self.assertTrue(torch.equal(left, right))

    def test_interpolation_preserves_endpoint_and_selects_slow_history(self):
        policy = interpolation_policy([.5, 1.0])
        column = torch.tensor([4])
        base = int(policy.hankel_starts[column])
        query = torch.cat([
            torch.stack([source[base + 1],
                         (source[base + 1] + source[base + 2]) / 2,
                         source[base + 2]]).flatten()
            for source in (policy.action, policy.state)])[None]
        actual = policy._past_columns(column, rate=.5)
        torch.testing.assert_close(actual, query, rtol=0, atol=0)
        supports, distances, _ = policy._hankel_topk(query, 4)
        self.assertEqual(int(supports[0, 0]), int(column[0]))
        self.assertEqual(float(distances[0, 0]), 0)
        self.assertEqual(float(policy._rate_trace(query, supports)["hankel_rates"][0, 0]), .5)
        self.assertGreater(float((policy._past_columns(column) - query).norm()), 0)
        self.assertEqual(len(supports[0].unique()), 4)
        for rate in (.5, 1., 2.):
            aligned = policy._past_columns(column, rate=rate)[0]
            torch.testing.assert_close(aligned[2:3], policy.action[base + 2])
            torch.testing.assert_close(aligned[-3:], policy.state[base + 2])

    def test_rate_search_never_uses_history_across_demo_boundaries(self):
        policy = interpolation_policy([.5, 1., 2.])
        columns = torch.arange(len(policy.hankel_starts))
        queries = policy._past_columns(torch.tensor([2, 10, 22]), rate=.5) + .17
        expected = torch.full((len(queries), len(columns)), float("inf"),
                              dtype=policy.dtype)
        for column in columns:
            end = int(policy.hankel_starts[column]) + policy.history - 1
            demo = policy.hankel_trajectory[column]
            for rate in (.5, 1., 2.):
                first = end - rate * (policy.history - 1)
                if first < int(policy.trajectory_starts[demo]):
                    continue
                sample = torch.linspace(first, end, policy.history,
                                        dtype=policy.dtype)
                lower, upper = sample.floor().long(), sample.ceil().long()
                weight = sample - lower
                history = torch.cat([
                    (source[lower] * (1 - weight[:, None])
                     + source[upper] * weight[:, None]).flatten()
                    for source in (policy.action, policy.state)])
                expected[:, column] = torch.minimum(
                    expected[:, column], (queries - history).square().sum(1))
        supports, distances, _ = policy._hankel_topk(queries, 6)
        reference, indices = expected.topk(6, largest=False)
        self.assertTrue(torch.equal(supports, indices))
        torch.testing.assert_close(distances.square(), reference,
                                   rtol=1e-9, atol=1e-9)
        self.assertTrue(all(len(row.unique()) == 6 for row in supports))
        with self.assertRaisesRegex(ValueError, "boundary"):
            policy._past_columns(torch.tensor([10]), rate=2)

    def test_fit_excludes_target_in_expanded_interpolation_history(self):
        policy = interpolation_policy([1., 2.])
        column = torch.tensor([4])
        query = policy._past_columns(column, rate=2)
        row = policy.hankel_starts[column] - 2
        self.assertFalse(bool(policy._hankel_exclusion(
            policy.hankel_starts[column], target_rows=row)[0, 0]))
        supports, distances, _ = policy._hankel_topk(
            query, len(policy.hankel_starts), target_rows=row)
        actual = distances[0, supports[0] == column[0]]
        expected = (query - policy._past_columns(column)).norm(dim=1)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(float(actual[0]), 0)

    def test_scalar_and_batched_retrieval_agree_with_demo_exclusion(self):
        policy = interpolation_policy([.5, 1., 2.])
        query = policy._past_columns(torch.tensor([2, 13, 24]), rate=.5) + .19
        demos = torch.tensor([0, 1, 2])
        batched = policy._hankel_topk(query, 5, target_demos=demos)
        scalar = [policy._hankel_topk(q[None], 5, target_demos=d[None])
                  for q, d in zip(query, demos)]
        for index, result in enumerate(batched):
            torch.testing.assert_close(result, torch.cat([r[index] for r in scalar]))
        self.assertTrue(bool((policy.hankel_trajectory[batched[0]]
                              != demos[:, None]).all()))

    def test_batched_direct_actions_match_scalar_with_interpolation(self):
        policy = toy_policy(5)
        policy.g_constraint, policy.g_solver, policy.g_l1 = "affine", "exact", 0
        policy.g_align_history = True
        policy.g_lookup_rates = (.5, 1., 2.)
        policy._rate_keys = None
        policy.trajectory_starts = torch.tensor([0])
        policy.hankel_trajectory = torch.zeros_like(policy.hankel_starts)
        generator = torch.Generator().manual_seed(41)
        states = torch.randn(4, 5, 4, generator=generator, dtype=policy.dtype)
        actions = torch.randn(4, 5, 2, generator=generator, dtype=policy.dtype)
        output, trace = policy.direct_actions(actions, states)
        for row in range(4):
            scalar, scalar_trace = policy.direct_actions(
                actions[row:row + 1], states[row:row + 1])
            torch.testing.assert_close(output[row], scalar[0], rtol=1e-5, atol=1e-5)
            for key in ("hankel_supports", "hankel_rates", "g_coefficients"):
                torch.testing.assert_close(trace[key][row], scalar_trace[key][0])

    def test_invalid_rates_are_rejected(self):
        for rates in ([], [.5], [1, 1], [0, 1], [-1, 1], [1, float("inf")]):
            with self.subTest(rates=rates), self.assertRaisesRegex(ValueError, "lookup_rates"):
                method = method_config()
                method["g"]["lookup_rates"] = rates
                Policy(trajectories(), data_config(), method)

    def test_interpolation_has_distinct_fit_and_target_cache_keys(self):
        config = {"data": data_config(), "method": method_config(),
                  "methods": {"metric": {"base": "gip_direct",
                                          "g_lookup": "action_ridge"}}}
        modified = copy.deepcopy(config)
        modified["method"]["g"].update(lookup_rates=[.5, 1., 2.], state_offset=1, future_consistency=0)
        modified["method"]["residual"]["lookup"] = "g_supports"
        self.assertNotEqual(fit_groups(config)[0]["key"],
                            fit_groups(modified)[0]["key"])
        self.assertNotEqual(_target_key(config, config["method"]),
                            _target_key(modified, modified["method"]))


if __name__ == "__main__":
    unittest.main()
