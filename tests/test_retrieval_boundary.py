"""Retrieval extension and ownership contracts, independent of policy dispatch."""
import copy
import unittest
from tests.suites import integration
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from dpc_nldl.bpc import Policy
from dpc_nldl.retrieval import (
    ActionRidgeLookup, FisherLookup, RawLookup, configure_search_precision,
    is_builtin_retrieval, make_retrieval,
)
from tests.support.policies import (
    action_ridge_config as _action_ridge_config, data_config as _data_config, trajectories
    as _trajectories,
)


class RetrievalBoundaryTest(unittest.TestCase):
    def test_config_factory_exposes_only_manuscript_lookups(self):
        for name, expected in (("past", RawLookup), ("action_ridge", ActionRidgeLookup),
                               ("lda", FisherLookup)):
            self.assertIsInstance(make_retrieval(name), expected)
        for name in ("action_neighborhood", "sparsemax_qp", "entropy", "prototype", "custom"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unknown G lookup"):
                make_retrieval(name)

    @integration
    def test_strategy_can_fit_and_project_without_policy_dispatch_changes(self):
        class PeriodicKey:
            def fit(self, policy):
                width = policy.history * (policy.action_dim + policy.state_dim)
                policy.g_lookup_matrix = torch.zeros(width, 1, dtype=policy.dtype)
                policy.g_lookup_stats = {"test_key": "periodic"}
                return policy.g_lookup_stats

            def project(self, policy, queries):
                return queries[..., -1:].sin()

        model = Policy(_trajectories(), _data_config(), _action_ridge_config(),
                       fit_residuals=False)
        model.retrieval = PeriodicKey()
        model.fit()
        self.assertEqual(model.fit_stats["g_lookup"], {"test_key": "periodic"})
        self.assertFalse(is_builtin_retrieval(model))
        query = model._queries(model.trajectory_starts[1:] + model.history)
        supports, distances, _ = model._hankel_topk(query, model.g_supports)
        bank = model._past_columns(torch.arange(len(model.hankel_starts)))
        expected = torch.cdist(query[:, -1:].sin(), bank[:, -1:].sin()).topk(
            model.g_supports, largest=False)
        torch.testing.assert_close(supports, expected.indices)
        torch.testing.assert_close(distances, expected.values, atol=1e-7, rtol=1e-7)

    def test_stateless_strategy_uses_shallow_copy_fields(self):
        model = SimpleNamespace(g_lookup="action_ridge", retrieval=ActionRidgeLookup(),
                                g_lookup_scale=2., g_lookup_matrix=torch.eye(3))
        shadow = copy.copy(model)
        shadow.g_lookup_matrix = 3 * model.g_lookup_matrix
        query = torch.tensor([[1., 2., 3.]])
        self.assertIs(shadow.retrieval, model.retrieval)
        torch.testing.assert_close(model.retrieval.project(model, query), 2 * query)
        torch.testing.assert_close(shadow.retrieval.project(shadow, query), 6 * query)

    def test_optimizations_accept_inheritance_but_reject_changed_projection(self):
        class Inherited(FisherLookup):
            pass

        class Changed(FisherLookup):
            def project(self, policy, queries):
                return super().project(policy, queries) * 2

        model = SimpleNamespace(g_lookup="lda", retrieval=Inherited())
        self.assertTrue(is_builtin_retrieval(model))
        model.retrieval = Changed()
        self.assertFalse(is_builtin_retrieval(model))
        model.retrieval = FisherLookup()
        model.retrieval.project = lambda policy, queries: queries
        self.assertFalse(is_builtin_retrieval(model))

    def test_precision_switch_retains_target_cache_but_resets_runtime_variants(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                selector = SimpleNamespace(config={"adaptive_support": "coverage", "fisher_rank": 8})
                model = SimpleNamespace(g_lookup="lda", support_selector=selector,
                                        device=torch.device(device))
                configure_search_precision(model, "mixed", for_targets=True)
                self.assertEqual(selector._density_search_mode,
                                 "reference" if device == "cpu" else "mixed")
                cache = selector._density_search = object()
                configure_search_precision(model, "mixed", for_targets=True)
                self.assertIs(selector._density_search, cache)
                configure_search_precision(model, "mixed")
                self.assertEqual(selector._density_search_mode, "mixed")
                self.assertIsNone(selector._density_search)
                selector._density_search = object()
                configure_search_precision(model, "mixed")
                self.assertIsNone(selector._density_search)

    @integration
    def test_inherited_policy_keeps_accelerated_density_search(self):
        from dpc_nldl.mixed_density_search import MixedDensitySearch
        from dpc_nldl.lda_selector import Selector
        from tests.support.search import density_fixture as fixture

        class InstrumentedPolicy(Policy):
            def summary(self):
                return len(self.hankel_starts)

        model, query, rows = fixture()
        model.__class__ = InstrumentedPolicy
        selector = Selector.__new__(Selector)
        selector.policy, selector._density_search_mode = model, "mixed"
        expected = model._hankel_topk(query, 256, rows)
        select = MixedDensitySearch.select
        with patch.object(MixedDensitySearch, "select", autospec=True, side_effect=select) as search:
            actual = selector._coverage_density_topk(query, 256, rows, None, None)
        self.assertEqual(search.call_count, 1)
        self.assertEqual(selector._density_search.last["route"], "mixed_certified")
        for index, (result, reference) in enumerate(zip(actual, expected)):
            tolerance = 2e-13 if index == 1 else 0
            torch.testing.assert_close(result, reference, atol=tolerance, rtol=tolerance)

    def test_linear_cuda_screens_fall_back_for_custom_retrieval(self):
        strategy = ActionRidgeLookup()
        strategy.project = lambda policy, queries: queries.sin()
        query, rows = torch.ones(2, 1), torch.arange(2)
        expected = object()
        model = SimpleNamespace(
            g_lookup="action_ridge", retrieval=strategy, g_native=True,
            device=torch.device("cuda"), pf=torch.zeros(1025, 1, dtype=torch.float64),
            g_temporal_screen=True, _support_screen=None,
            _literal_g_supports=Mock(return_value=expected),
            _temporal_support_topk=Mock(side_effect=AssertionError("linear screen bypassed custom retrieval")),
        )
        for rates, count in (((1.,), 200), ((.5, 1., 2.), 64)):
            with self.subTest(rates=rates), patch(
                    "dpc_nldl.fast_neighbors.CertifiedTopK",
                    side_effect=AssertionError("linear screen bypassed custom retrieval")):
                model.g_lookup_rates = rates
                self.assertIs(Policy._g_support_topk(model, query, count, rows, None, 2), expected)
        self.assertEqual(model._literal_g_supports.call_count, 2)
        model._temporal_support_topk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
