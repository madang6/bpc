import math
import unittest
from tests.suites import cuda, integration
from unittest.mock import patch

import torch


from dpc_nldl import fitting
from dpc_nldl.bpc import Policy
from tests.support.policies import (
    action_ridge_config, data_config, method_config, selector_policy, small_ridge_fit,
    trajectories,
)


class GSelectorTest(unittest.TestCase):
    def test_action_neighborhood_teacher_excludes_overlapping_windows(self):
        policy = Policy(
            trajectories(demos=3, length=12), data_config(), method_config(),
            fit_residuals=False)
        columns = torch.arange(len(policy.hankel_starts))
        features = policy.hankel_starts[:, None].to(policy.dtype)
        anchors = torch.tensor([3])
        neighbors, _ = policy._metric_topk(anchors, columns, features, 2, 1)
        overlap = ((policy.hankel_trajectory[neighbors[0]]
                    == policy.hankel_trajectory[anchors[0]])
                   & ((policy.hankel_starts[neighbors[0]]
                       - policy.hankel_starts[anchors[0]]).abs()
                      < policy.history + policy.horizon))
        self.assertFalse(bool(overlap.any()))


    def test_action_ridge_lookup_uses_projected_hankel_keys(self):
        policy = Policy(
            trajectories(), data_config(), action_ridge_config(),
            fit_residuals=False)
        width = policy.history * (policy.action_dim + policy.state_dim)
        policy.g_lookup_matrix = torch.zeros(width, 1, dtype=policy.dtype)
        policy.g_lookup_matrix[-1, 0] = 1
        policy._project_hankel()
        query = policy._past_columns(torch.tensor([3]))
        supports, distances, excluded = policy._hankel_topk(query, 2)
        expected = torch.topk(
            (policy.pf - query @ policy.g_lookup_matrix).square().sum(1),
            2, largest=False).indices
        self.assertTrue(torch.equal(supports[0], expected))
        self.assertEqual(float(distances[0, 0]), 0)
        self.assertEqual(int(excluded[0]), 0)

    @integration
    def test_action_ridge_fit_round_trip_and_exclusion(self):
        method = action_ridge_config()
        with small_ridge_fit():
            policy = Policy(
                trajectories(), data_config(), method, fit_residuals=False).fit()
        columns = torch.arange(len(policy.hankel_starts))
        past = policy._past_columns(columns).double()
        future = policy.uf.T.double()
        past -= past.mean(0)
        future -= future.mean(0)
        ridge = policy.fit_stats["g_lookup"]["ridge"]
        self.assertGreater(policy.fit_stats["g_lookup"]["matched_scale"], 0)
        residual = ((past.T @ past + ridge * torch.eye(past.shape[1]))
                    @ policy.g_lookup_matrix.double() - past.T @ future)
        self.assertLess(float(residual.norm() / (past.T @ future).norm()), 1e-6)

        payload = policy.fitted_payload({"test": True})
        with patch.object(
                Policy, "_project_hankel",
                side_effect=AssertionError("redundant projected-key pass")):
            restored = Policy(
                trajectories(), data_config(), method,
                fitted=payload, fit_residuals=False)
        self.assertTrue(torch.equal(policy.pf, restored.pf))
        demo = torch.tensor([1])
        query = restored._past_columns(torch.tensor([0]))
        supports, _, excluded = restored._hankel_topk(
            query, 2, target_demos=demo)
        expected = int((restored.hankel_trajectory == demo).sum())
        self.assertEqual(int(excluded[0]), expected)
        self.assertTrue(torch.all(restored.hankel_trajectory[supports[0]] != demo))


    def test_chunked_hankel_lookup_matches_materialized_plane(self):
        policy = selector_policy()
        row = int(policy.trajectory_starts[1]) + policy.history
        query = policy._queries(torch.tensor([row]))
        expected = policy._hankel_topk(query, policy.g_supports)
        policy.pf = policy.pf_norm = None
        actual = policy._hankel_topk(query, policy.g_supports)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

    def test_chunked_g_planes_match_dense_solve(self):
        policy = selector_policy()
        rows = policy.trajectory_starts[1:4] + policy.history
        expected = policy._offline_g(rows)
        lookup, sizes = policy._hankel_topk, []

        def capture(queries, *args, **kwargs):
            sizes.append(len(queries))
            return lookup(queries, *args, **kwargs)

        with (patch("dpc_nldl.bpc.G_PLANE_BYTES", 1),
              patch.object(policy, "_hankel_topk", side_effect=capture)):
            actual = policy._offline_g(rows)
        self.assertEqual(sizes, [1] * len(rows))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)

    def test_float32_fit_uses_materialized_float64_lookup(self):
        method = method_config()
        method["g"]["fit_dtype"] = "float32"
        policy = Policy(
            trajectories(), data_config(), method,
            fit_residuals=False)
        rows = policy.trajectory_starts[1:3] + policy.history
        prior, excluded, same_demo, _ = policy._offline_g(rows)
        self.assertEqual(prior.dtype, policy.dtype)
        self.assertEqual((len(prior), len(excluded), len(same_demo)), (2, 2, 2))

    def test_fixed_g_matches_materialized_history(self):
        policy = selector_policy()
        rows = policy.trajectory_starts[1:4] + policy.history
        queries = policy._queries(rows).float()
        supports, _, _ = policy._hankel_topk(
            policy._queries(rows), policy.g_supports, target_rows=rows)
        plane = policy._past_columns(supports).float()
        expected = (plane @ plane.transpose(1, 2),
                    torch.bmm(plane, queries.unsqueeze(2)).squeeze(2))
        actual = policy._fixed_g(supports, rows)
        torch.testing.assert_close(actual, policy._solve_g_normal(*expected),
                                   rtol=2e-5, atol=1e-5)

    def test_uniform_g_averages_only_the_first_supported_actions(self):
        method = method_config()
        method["g"]["weights"] = "uniform"
        policy = Policy(
            trajectories(), data_config(), method,
            fit_residuals=False)
        row = int(policy.trajectory_starts[1]) + policy.history
        query = policy._queries(torch.tensor([row]))
        supports, _, _ = policy._hankel_topk(query, policy.g_supports)
        futures = policy.uf.T[supports]
        with patch.object(
                policy, "_solve_affine",
                side_effect=AssertionError("optimized G solve invoked")):
            prior, coefficients = policy._g_prior(
                policy._past_columns(supports), supports, query)
        torch.testing.assert_close(
            prior[:, :policy.action_dim],
            futures[:, :, :policy.action_dim].mean(1))
        torch.testing.assert_close(
            prior[:, policy.action_dim:],
            futures[:, 0, policy.action_dim:])
        torch.testing.assert_close(
            coefficients, torch.full_like(coefficients, 1 / policy.g_supports))
        with patch.object(
                policy, "_past_columns",
                side_effect=AssertionError("uniform G constructed a support plane")):
            policy._offline_g(torch.tensor([row]))


class RFFSelectorTest(unittest.TestCase):
    def test_rff_storage_validation(self):
        for storage in ("invalid", "host_paged"):
            method = method_config()
            method["rff"].update({"dtype": "float32", "storage": storage})
            with self.assertRaisesRegex(ValueError, "RFF|host_paged"):
                Policy(trajectories(), data_config(), method)

    @cuda
    def test_host_paged_fit_matches_resident_with_page_tail_and_sparse_targets(self):
        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype):
                self._check_paged_fit(dtype)

    def _check_paged_fit(self, dtype):
        method = method_config()
        method["device"] = "cuda"
        method["dtype"] = dtype
        method["scaling"]["enabled"] = True
        method["rff"].update({"dtype": "float32", "features": 32})
        results = []
        for storage in ("resident", "host_paged"):
            method["rff"]["storage"] = storage
            policy = Policy(trajectories(demos=2, length=4120), data_config(),
                            method)
            rows = torch.arange(policy.samples, device="cuda")
            neighbors = torch.stack([(rows + 3) % len(rows), (rows + 9) % len(rows)], 1)
            eligible = rows[policy.row_in_trajectory >= policy.history]
            targets = .3 * policy.action[eligible] + .1
            build, captured = fitting.feature_atlas, {}

            def capture(*args):
                atlas = build(*args)
                captured["features"] = torch.empty(len(rows), policy.rff_dim)
                fitting.replay_rff(policy, atlas, lambda page, start, stop:
                    captured["features"][start:stop].copy_(page.cpu()))
                return atlas

            with (patch.object(fitting, "residual_training_neighbors",
                               return_value=(rows, neighbors)),
                  patch.object(fitting, "feature_atlas", side_effect=capture)):
                fitting.fit_rff(policy, eligible, targets)
            self.assertEqual(policy.rff_storage_used, storage)
            self.assertLessEqual(policy.rff_refinement["deployed_relative_residual"], 5e-6)
            results.append((captured["features"], policy.W_gip.cpu(), policy.W_knn.cpu(),
                            torch.tensor([policy.s2_gip, policy.s2_knn])))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        for left, right in zip(results[0][1:], results[1][1:]):
            torch.testing.assert_close(left, right, rtol=5e-4, atol=5e-5)

    @staticmethod
    def _direct_features(policy, rows, neighbors):
        elements = fitting.neighbor_elements(policy, rows, neighbors)
        return (policy.rff_scale * torch.cos(
            elements @ policy.omega + policy.phase)).mean(1)

    def test_preprojected_atlas_is_the_same_float64_feature_map(self):
        method = method_config()
        method["rff"].update({"features": 515, "dtype": "float64"})
        policy = Policy(
            trajectories(), data_config(), method,
            fit_residuals=True)
        generator = torch.Generator().manual_seed(91)
        policy.omega = torch.randn(
            policy.input_dim, policy.rff_dim,
            generator=generator, dtype=policy.dtype)
        policy.phase = torch.rand(
            policy.rff_dim, generator=generator, dtype=policy.dtype) * math.tau
        policy._factor_rff()
        rows = torch.tensor([2, 11, 21])
        neighbors = torch.tensor([[1, 1, 4], [8, 13, 13], [18, 23, 27]])
        expected = self._direct_features(policy, rows, neighbors)
        actual = fitting.feature_atlas(policy, rows, neighbors)
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
        torch.testing.assert_close(
            policy._query_features(policy.state[rows], neighbors), expected,
            rtol=2e-13, atol=2e-13)

    def test_fast_fit_uses_one_atlas_and_matches_float64_ridge(self):
        method = method_config()
        method["rff"].update({"features": 32, "dtype": "float32"})
        policy = Policy(
            trajectories(), data_config(), method,
            fit_residuals=True)
        rows = torch.arange(policy.samples)
        neighbors = torch.stack([(rows + 3) % len(rows), (rows + 9) % len(rows)], 1)
        eligible = torch.nonzero(policy.row_in_trajectory >= policy.history).ravel()
        gip_targets = .3 * policy.action[eligible] + .1
        captured = {}
        build = fitting.feature_atlas

        def capture(*args):
            captured["features"] = build(*args)
            return captured["features"]

        with (patch.object(
                fitting, "residual_training_neighbors",
                return_value=(rows, neighbors)),
              patch.object(fitting, "feature_atlas", side_effect=capture) as atlas):
            fitting.fit_rff(policy, eligible, gip_targets)
        self.assertEqual(atlas.call_count, 1)
        self.assertLessEqual(policy.rff_refinement["deployed_relative_residual"], 5e-6)

        features = self._direct_features(policy, rows, neighbors)
        gram = features.T @ features
        gram.diagonal().add_(policy.rff_ridge)
        knn_targets = policy.action[rows] - policy.action[neighbors].mean(1)
        expected_knn = torch.linalg.solve(gram, features.T @ knn_targets)
        expected_gip = torch.linalg.solve(gram, features[eligible].T @ gip_targets)
        torch.testing.assert_close(
            captured["features"].double(), features, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            policy._query_features(policy.state[rows], neighbors),
            captured["features"], rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(
            policy.W_knn.double(), expected_knn, rtol=5e-4, atol=5e-5)
        torch.testing.assert_close(
            policy.W_gip.double(), expected_gip, rtol=5e-4, atol=5e-5)
        for weight, index, target, variance in (
                (policy.W_knn, rows, knn_targets, policy.s2_knn),
                (policy.W_gip, eligible, gip_targets, policy.s2_gip)):
            error = (captured["features"][index] @ weight).double() - target
            torch.testing.assert_close(
                torch.tensor(variance), torch.tensor(max(float(error.square().mean()),
                    policy.residual_variance_floor)), rtol=2e-6, atol=1e-7)

    def test_float64_fit_reuses_the_joined_ridge_without_an_atlas(self):
        method = method_config()
        method["rff"]["features"] = 32
        policy = Policy(
            trajectories(), data_config(), method,
            fit_residuals=True)
        rows = torch.arange(policy.samples)
        neighbors = torch.stack([(rows + 3) % len(rows), (rows + 9) % len(rows)], 1)
        eligible = torch.nonzero(policy.row_in_trajectory >= policy.history).ravel()
        gip_targets = .3 * policy.action[eligible] + .1
        policy.feature_batch = 3
        feature_batches, moment_batches = [], []
        query_features, addmm = policy._query_features, torch.Tensor.addmm_

        def bounded_features(queries, candidates):
            feature_batches.append(len(queries))
            self.assertLessEqual(len(queries), policy.feature_batch)
            return query_features(queries, candidates)

        def moments(target, left, right, *args, **kwargs):
            if right.shape[1] == policy.rff_dim:
                moment_batches.append(len(right))
            return addmm(target, left, right, *args, **kwargs)

        with (patch("dpc_nldl.fitting.FIT_ROWS", 7),
              patch.object(policy, "_query_features", side_effect=bounded_features),
              patch.object(torch.Tensor, "addmm_", moments),
              patch.object(
                fitting, "residual_training_neighbors",
                return_value=(rows, neighbors)),
              patch.object(
                  fitting, "feature_atlas",
                  side_effect=AssertionError("legacy fit materialized an atlas"))):
            fitting.fit_rff(policy, eligible, gip_targets)
        self.assertEqual(sum(feature_batches), 2 * len(rows))  # moments and diagnostics
        self.assertEqual(moment_batches, [7, 7, 7, 7, 4])
        features = self._direct_features(policy, rows, neighbors)
        gram = features.T @ features
        gram.diagonal().add_(policy.rff_ridge)
        knn_targets = policy.action[rows] - policy.action[neighbors].mean(1)
        torch.testing.assert_close(
            policy.W_knn, torch.linalg.solve(gram, features.T @ knn_targets))
        torch.testing.assert_close(
            policy.W_gip,
            torch.linalg.solve(gram, features[eligible].T @ gip_targets))
        self.assertIsNone(policy.rff_refinement)

    @integration
    def test_g_support_residual_lookup_reuses_hankel_supports(self):
        method = action_ridge_config()
        method["residual"]["lookup"] = "g_supports"
        policy = Policy(
            trajectories(demos=8, length=12), data_config(), method,
            fit_residuals=True)
        width = policy.history * (policy.action_dim + policy.state_dim)
        policy.g_lookup_matrix = torch.randn(width, 2, dtype=policy.dtype)
        policy._project_hankel()
        rows = torch.tensor([policy.history, policy.history + 1])
        _, _, _, retained = policy._offline_g(rows)
        neighbors = torch.empty(
            policy.samples, policy.g_supports, dtype=torch.long)
        neighbors[rows] = retained
        with patch.object(
                policy, "_hankel_topk",
                side_effect=AssertionError("G supports searched twice")):
            selected_rows, reused = fitting.residual_training_neighbors(policy, neighbors)
        self.assertTrue(torch.equal(reused[selected_rows[:2]], retained))


if __name__ == "__main__":
    unittest.main()
