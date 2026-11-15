import tempfile
import unittest
from tests.suites import integration
from unittest.mock import patch

import torch

from tests.support.policies import data_config, method_config, trajectories
from tests.support.readout import VariableSelector, ragged_policy as policy
from dpc_nldl import fitting
from dpc_nldl.bpc import Policy
from dpc_nldl.ragged_supports import RaggedColumns, pooled_mean


class RaggedSupportTests(unittest.TestCase):
    def test_storage_round_trip_and_true_count_pooling(self):
        rows = [torch.tensor([4]), torch.tensor([2, 8, 6]), torch.tensor([1, 3])]
        packed = RaggedColumns.from_rows(rows)
        restored = RaggedColumns(**packed.payload()).align(torch.tensor([1, 3, 4]), 6)
        dense = restored[torch.tensor([4, 1, 3])]
        self.assertEqual(dense.tolist(), [[1, 3, -1], [4, -1, -1], [2, 8, 6]])
        values = torch.arange(10.).reshape(-1, 1)[dense.clamp_min(0)]
        torch.testing.assert_close(pooled_mean(values, dense >= 0),
                                   torch.tensor([[2.], [4.], [16 / 3]]))
        chosen = restored.take(torch.tensor([4, 1]))
        self.assertEqual(chosen.values.tolist(), [1, 3, 4])
        self.assertEqual(chosen.counts.tolist(), [2, 1])

    def test_offline_g_uses_exact_selected_sets_and_exclusions(self):
        model = policy()
        rows = torch.tensor([3, 5, 11, 19])
        choices = model.support_selector.select(model._queries(rows), target_rows=rows)
        actual, _, _, neighbors = model._offline_g(rows)
        for index, (columns, _) in enumerate(choices):
            selected = rows[index:index + 1]
            expected = model._offline_g(selected, atlas=(columns[None], selected.new_zeros(1)))[0]
            torch.testing.assert_close(actual[index:index + 1], expected, atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(neighbors[index][0], model._support_endpoints(columns))
            starts = model.hankel_starts[columns]
            self.assertFalse(bool(((selected - model.history <= starts) & (starts <= selected)).any()))

    def test_rff_atlas_matches_each_unpadded_neighborhood(self):
        model = policy()
        rows = torch.tensor([3, 5, 11])
        neighbors = RaggedColumns.from_rows([torch.tensor([6]), torch.tensor([9, 10, 14]),
                                             torch.tensor([17, 22])]).align(rows, model.samples)
        fitting.make_basis(model, rows, neighbors, 0, .5)
        atlas = fitting.feature_atlas(model, rows, neighbors.take(rows))
        expected = torch.cat([model._query_features(model.state[row:row + 1], neighbors[row])
                              for row in rows.tolist()])
        torch.testing.assert_close(atlas, expected, atol=3e-6, rtol=3e-6)
        padded = neighbors[rows]
        torch.testing.assert_close(model._query_features(
            model.state[rows], padded, mask=padded >= 0), expected, atol=3e-6, rtol=3e-6)

    @integration
    def test_cached_fit_and_online_readout_reuse_variable_supports(self):
        expected_fit = policy().fit()
        model = policy()
        with tempfile.TemporaryDirectory() as folder, patch.object(
                model, "_hankel_topk", side_effect=AssertionError("old lookup used")):
            model.fit_targets(folder, None)
            with patch.object(model, "_offline_g", side_effect=AssertionError("cached targets rebuilt")):
                model.fit(folder)
        for name in ("W_gip", "W_knn", "omega", "phase"):
            torch.testing.assert_close(getattr(model, name), getattr(expected_fit, name), rtol=0, atol=0)
        rows = torch.tensor([3, 11, 19])
        queries = model._queries(rows)
        split = model.history * model.action_dim
        actions = queries[:, :split].reshape(len(rows), model.history, model.action_dim)
        states = queries[:, split:].reshape(len(rows), model.history, model.state_dim)
        chosen = model.support_selector.select(queries)
        actual, trace = model.direct_actions(actions, states)
        for index, (columns, distances) in enumerate(chosen):
            expected, _ = model.direct_actions(actions[index:index + 1], states[index:index + 1],
                                               selected=(columns[None], distances[None]))
            torch.testing.assert_close(actual[index:index + 1], expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(trace["selector_count"].tolist(), [1, 2, 3])


    def test_nnc_variable_supports_generate_actions_without_residual_features(self):
        model = policy().select_method('deepc_g_direct')
        rows = torch.tensor([3, 11, 19])
        queries = model._queries(rows)
        split = model.history * model.action_dim
        actions = queries[:, :split].reshape(len(rows), model.history, model.action_dim)
        states = queries[:, split:].reshape(len(rows), model.history, model.state_dim)
        selected = model.support_selector.select(queries)
        with patch.object(model, '_query_features', side_effect=AssertionError('NNC used RFF')):
            actual, trace = model.direct_actions(actions, states)
            for index, (columns, distances) in enumerate(selected):
                expected, _ = model.direct_actions(actions[index:index+1], states[index:index+1],
                    selected=(columns[None], distances[None]))
                torch.testing.assert_close(actual[index:index+1], expected, rtol=0, atol=0)
        self.assertEqual(trace['selector_count'].tolist(), [1, 2, 3])
        self.assertTrue(torch.equal(trace['residual_z'], torch.zeros_like(actual)))
        torch.testing.assert_close(actual, trace['base_z'] * model.action_scale + model.action_mean)

    def test_retimed_ragged_atlas_matches_explicit_ridge(self):
        method = method_config()
        method["g"].update(lookup="action_ridge", state_offset=1, align_history=True,
                           align_future=True, future_consistency=0, lookup_rates=[1.])
        method["residual"].update(lookup="g_supports", action_source="retimed_first")
        method["rff"].update(dtype="float32", features=16, refine_steps=4, refine_tolerance=1e-5)
        model = Policy(trajectories(length=12), data_config(), method)
        width = model.history * (model.action_dim + model.state_dim)
        model.g_lookup_matrix = torch.eye(width, dtype=model.dtype)
        model._project_hankel()
        model.support_selector = VariableSelector(model)
        rows = torch.nonzero(model.row_in_trajectory >= model.history).flatten()
        base, _, _, endpoints = model._offline_g(rows)
        neighbors = endpoints.align(rows, model.samples)
        target = model.action[rows] - base[:, :model.action_dim]
        fitting.make_basis(model, rows, neighbors, 0, method["rff"]["sigma_scale"])
        features = []
        for row in rows.tolist():
            indices = neighbors[row]
            actions = model.retimed_neighbor_actions(model._queries(torch.tensor([row])), indices,
                                                     torch.tensor([row]))
            features.append(model._query_features(model.state[row:row + 1], indices, actions))
        features = torch.cat(features).double()
        expected = torch.linalg.solve(features.T @ features + model.rff_ridge * torch.eye(
            model.rff_dim, dtype=torch.float64), features.T @ target)
        fitting.fit_rff(model, rows, target, neighbors)
        torch.testing.assert_close(model.W_gip.double(), expected, rtol=3e-5, atol=3e-5)

if __name__ == "__main__":
    unittest.main()
