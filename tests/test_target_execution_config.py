"""Fitting arithmetic changes must not reuse residuals or target caches."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from tests.suites import integration
from unittest.mock import Mock

import torch

from bpc.artifacts import _target_key
from bpc.loader import compatible_spec
from bpc.bpc import Policy
from bpc.pipeline import (
    _validate_methods, fit_groups, load_config, target_execution_options,
)


class TargetExecutionConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] /
            'configs/robosuite_fisher_sparsemax_local.json')
        self.config['method']['residual'].pop('target_retrieval_precision', None)
        self.config['method']['residual'].pop('target_execution', None)

    def test_explicit_historical_defaults_preserve_artifact_identity(self):
        before = deepcopy(self.config)
        historical = fit_groups(self.config)
        self.config['method']['residual'].update(
            target_retrieval_precision='fp64', target_execution='legacy')
        self.assertEqual(fit_groups(self.config), historical)
        self.assertEqual(_target_key(self.config, self.config['method']),
                         _target_key(before, before['method']))
        self.assertEqual(target_execution_options(before['method']), {
            'target_retrieval_precision': 'fp64', 'target_execution': 'legacy'})
        self.assertNotIn('target_execution', before['method']['residual'])

    def test_fitting_changes_isolate_both_artifacts_and_target_cache(self):
        historical = fit_groups(self.config)[0]
        historical_target = _target_key(self.config, historical['method'])
        identities, target_ids = {historical['key']}, {historical_target}
        for precision, execution in (
                ('mixed', 'legacy'), ('fp64', 'batched'), ('mixed', 'batched')):
            with self.subTest(precision=precision, execution=execution):
                changed = deepcopy(self.config)
                changed['method']['residual'].update(
                    target_retrieval_precision=precision, target_execution=execution)
                _validate_methods(changed, None)
                current = fit_groups(changed)[0]
                target = _target_key(changed, current['method'])
                self.assertNotIn(current['key'], identities)
                self.assertNotIn(target, target_ids)
                self.assertFalse(compatible_spec(historical['spec'], current['spec']))
                identities.add(current['key'])
                target_ids.add(target)

    def test_runtime_precision_remains_independent_of_fitting_options(self):
        self.config['method']['residual'].update(
            target_retrieval_precision='mixed', target_execution='batched')
        before = fit_groups(self.config)
        self.config['methods']['bpc_fisher']['retrieval_precision'] = 'mixed'
        self.assertEqual(fit_groups(self.config), before)

    def test_rejects_unknown_execution_and_incompatible_mixed_selector(self):
        for field in ('target_retrieval_precision', 'target_execution'):
            changed = deepcopy(self.config)
            changed['method']['residual'][field] = 'wrong'
            with self.assertRaisesRegex(ValueError, field):
                _validate_methods(changed, None)
            with self.assertRaisesRegex(ValueError, field):
                fit_groups(changed)
        self.config['method']['residual']['target_retrieval_precision'] = 'mixed'
        self.config['method']['g']['lda']['fisher_rank'] = None
        with self.assertRaisesRegex(ValueError, 'positive fisher_rank'):
            _validate_methods(self.config, None)

    @integration
    def test_target_sharding_applies_execution_precision_without_refitting_lookup(self):
        # Device labels exercise runtime dispatch without requiring a GPU; the
        # actual target sharding and file writes still run on CPU tensors.
        for device, precision, expected in (
                ('cpu', 'fp64', 'reference'), ('cpu', 'mixed', 'reference'),
                ('cuda', 'fp64', 'fused'), ('cuda', 'mixed', 'mixed')):
            with self.subTest(device=device, precision=precision), TemporaryDirectory() as folder:
                policy = object.__new__(Policy)
                policy.config = deepcopy(self.config['method'])
                policy.target_execution_options = dict(
                    target_retrieval_precision=precision, target_execution='batched')
                policy.device = torch.device(device)
                policy.g_lookup = 'lda'
                payload = object()
                policy.support_selector = SimpleNamespace(
                    config=dict(fisher_rank=70, adaptive_support='coverage'),
                    payload=payload, _density_search_mode='old', _density_search=object())
                policy.history, policy.target_batch, policy.action_dim = 2, 3, 2
                policy.row_in_trajectory = torch.arange(7)
                policy.action = torch.arange(14).reshape(7, 2).double()
                policy.residual_lookup = 'g_supports'
                policy.fit_g_lookup = Mock(side_effect=AssertionError('lookup refitted'))

                def offline(rows):
                    self.assertEqual(policy.support_selector._density_search_mode, expected)
                    self.assertIsNone(policy.support_selector._density_search)
                    return (torch.zeros(len(rows), 2), rows.new_zeros(len(rows)),
                            rows.new_zeros(len(rows)), rows[:, None])

                policy._offline_g = Mock(side_effect=offline)
                original = deepcopy(policy.config)
                policy.fit_targets(folder, None)
                policy.fit_g_lookup.assert_not_called()
                self.assertIs(policy.support_selector.payload, payload)
                self.assertEqual(policy.config, original)
                parts = [torch.load(path) for path in sorted(Path(folder).glob('*.pt'))]
                self.assertEqual(torch.cat([part['rows'] for part in parts]).tolist(), [2, 3, 4, 5, 6])
                torch.testing.assert_close(torch.cat([part['targets'] for part in parts]),
                                           policy.action[2:])

    def test_batched_guard_compares_resolved_tensor_device(self):
        from bpc.lda_selector import Selector
        from bpc.retrieval import make_retrieval
        from bpc.target_readout import _eligible
        policy = object.__new__(Policy)
        # cpu:0 versus cpu reproduces the cuda versus cuda:0 alias issue
        # without requiring a CUDA test runner.
        policy.device = torch.device('cpu:0')
        policy.hankel_starts = torch.arange(4)
        policy.support_selector = object.__new__(Selector)
        policy.support_selector.config = {'fisher_rank': 70}
        policy.g_lookup, policy.g_native, policy.g_fit_dtype = 'lda', True, torch.float32
        policy.retrieval = make_retrieval(policy.g_lookup)
        policy.g_solver, policy.g_constraint, policy.g_weights = 'exact', 'affine', 'optimized'
        policy.g_tolerance = None
        queries = torch.zeros(2, 3)
        self.assertNotEqual(queries.device, policy.device)
        self.assertTrue(_eligible(policy, queries))


if __name__ == '__main__':
    unittest.main()
