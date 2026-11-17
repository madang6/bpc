import tempfile
import unittest
from tests.suites import cuda, integration
from pathlib import Path
from unittest.mock import patch

import torch

from bpc.bpc import Policy
from tests.support.policies import (
    action_ridge_config, data_config, small_ridge_fit, toy_policy, trajectories,
)
from tests.support.numerics import reference_prox


class AffineGTest(unittest.TestCase):

    def test_constraint_validation_and_default(self):
        method = action_ridge_config()
        policy = Policy(trajectories(), data_config(), method, fit_residuals=False)
        self.assertEqual(policy.g_constraint, "affine")
        method["g"]["constraint"] = "invalid"
        with self.assertRaisesRegex(ValueError, "constraint"):
            Policy(trajectories(), data_config(), method, fit_residuals=False)

    def test_prox_matches_reference_including_left_boundary_and_ties(self):
        generator = torch.Generator().manual_seed(91)
        for dtype in (torch.float32, torch.float64):
            for count in (1, 2, 7, 200):
                proposal = torch.randn(4, count, generator=generator, dtype=dtype)
                proposal[0].zero_()  # root lies before the first breakpoint
                proposal[1].fill_(.25)  # tied breakpoints
                threshold = torch.tensor([2., .03, 0., .7], dtype=dtype)
                actual = Policy._affine_l1_prox(proposal, threshold)
                expected = reference_prox(proposal, threshold).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0,
                                           atol=2e-5 if dtype == torch.float32 else 1e-12)

    def test_solver_signed_mass_kkt_and_single_batch_equivalence(self):
        p = toy_policy(5)
        p.g_constraint = "affine"
        rhs = torch.tensor([[2., -2., 0., 0.], [0., 0., 0., 0.]], dtype=p.dtype)
        gram = torch.eye(4, dtype=p.dtype)[None].expand(2, -1, -1)
        actual = p._solve_g_normal(gram, rhs)
        expected = reference_prox(rhs, rhs.new_full((2,), p.g_l1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(actual.sum(1), torch.ones(2, dtype=p.dtype))
        self.assertLess(float(actual[0].min()), 0)
        self.assertLess(float(p._g_kkt(actual, actual - rhs).max()), 1e-12)
        for i in range(2):
            torch.testing.assert_close(actual[i:i + 1], p._solve_g_normal(gram[i:i + 1], rhs[i:i + 1]))

    def test_fixed_iterations_match_independent_reference(self):
        p = toy_policy(5)
        p.g_constraint = "affine"
        generator = torch.Generator().manual_seed(14)
        plane = torch.randn(3, 7, 15, generator=generator, dtype=p.dtype)
        gram = plane @ plane.transpose(1, 2)
        rhs = torch.randn(3, 7, generator=generator, dtype=p.dtype)
        lipschitz = torch.linalg.eigvalsh(gram)[:, -1]
        expected = torch.full_like(rhs, 1 / rhs.shape[1])
        for _ in range(p.g_iterations):
            proposal = expected - ((gram @ expected[:, :, None]).squeeze(2) - rhs) / lipschitz[:, None]
            expected = reference_prox(proposal, p.g_l1 / lipschitz)
        torch.testing.assert_close(p._solve_g_normal(gram, rhs), expected, rtol=0, atol=1e-12)

    def test_batched_shared_rff_readout_matches_single_and_legacy_paths(self):
        p = toy_policy(5)
        p.g_constraint = "affine"
        states = p.state[:15].reshape(3, 5, 4)
        actions = p.action[:15].reshape(3, 5, 2)
        output, trace = p.direct_actions(actions, states)
        torch.testing.assert_close(trace["g_sum"], torch.ones(3, dtype=p.dtype))
        self.assertTrue(bool(torch.isfinite(trace["g_kkt"]).all()))
        for i in range(3):
            single, detail = p.direct_actions(actions[i:i + 1], states[i:i + 1])
            torch.testing.assert_close(output[i:i + 1], single, rtol=1e-5, atol=1e-5)
            for name in trace:
                torch.testing.assert_close(trace[name][i:i + 1], detail[name], rtol=1e-5, atol=1e-5)
        # Force the generic (non-batched) action_for entry, sharing the same solver.
        raw_actions = actions[0] * p.action_scale + p.action_mean
        raw_state = states[0, -1] * p.state_scale + p.state_mean
        p._state_history = [torch.zeros(4), *states[0, :-1]]
        p._action_history = [torch.zeros(2), *raw_actions[:-1]]
        with patch.object(Policy, "batched_direct", False):
            actual = p.action_for(raw_state.numpy(), raw_actions[-1].numpy())
        torch.testing.assert_close(torch.from_numpy(actual), output[0], rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual(float(p.last_trace["g_sum"]), 1., places=12)

    @integration
    def test_shared_support_fit_cache_round_trip(self):
        for temporal in (False, True):
            with self.subTest(temporal=temporal):
                method = action_ridge_config()
                method["g"].update(constraint="affine", solver="exact", l1=0)
                if temporal:
                    method["g"].update(state_offset=1, lookup_rates=[.5, .75, 1, 1.5, 2])
                bank = trajectories(demos=8, length=12)
                p = Policy(bank, data_config(), method)
                with small_ridge_fit():
                    p.fit_g_lookup()
                p._project_hankel()
                with tempfile.TemporaryDirectory() as folder:
                    p.fit_targets(folder, None)
                    for path in Path(folder).glob("*.pt"):
                        part = torch.load(path, weights_only=True)
                        rows = part["rows"]
                        base, _, _, neighbors = p._offline_g(rows)
                        torch.testing.assert_close(part["targets"], p.action[rows] - base[:, :p.action_dim])
                        torch.testing.assert_close(part["g_neighbors"], neighbors)
                    with patch.object(p, "_offline_g", side_effect=AssertionError("cached G targets rebuilt")):
                        p.fit(target_folder=folder)
                restored = Policy(bank, data_config(), method, fitted=p.fitted_payload({}))
                self.assertEqual(restored.g_constraint, "affine")
                self.assertEqual(restored.g_solver, "exact")
                torch.testing.assert_close(restored.residual_weights, p.residual_weights)
                self.assertTrue(p.fit_stats["g_support_reused_for_rff"])

    @cuda
    def test_packed_cuda_prior_matches_materialized(self):
        method = action_ridge_config()
        method["device"] = "cuda"
        method["g"].update(constraint="affine", fit_dtype="float32", iterations=200)
        p = Policy(trajectories(demos=8, length=12), data_config(), method,
                   fit_residuals=False)
        with small_ridge_fit():
            p.fit_g_lookup()
        p._project_hankel()
        rows = p.trajectory_starts[1:] + p.history
        query = p._queries(rows)
        supports, _, _ = p._hankel_topk(query, p.g_supports, target_rows=rows)
        actual = p._g_prior(None, supports, query.float(), rows)
        expected = p._g_prior(p._past_columns(supports).float(), supports, query.float())
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=5e-5, atol=5e-5)


if __name__ == "__main__":
    unittest.main()
