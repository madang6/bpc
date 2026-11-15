"""Exercise the actual Fisher selector and Policy._offline_g dispatch."""
import unittest
from tests.suites import cuda, integration
from unittest.mock import patch

import torch
import numpy as np

from dpc_nldl.bpc import Policy
from dpc_nldl.ragged_supports import RaggedColumns
from dpc_nldl.target_readout import _eligible, offline_prior
from tests.support.policies import data_config as _data_config, trajectories as _trajectories
from tests.support.policies import fisher_recipe as recipe


def fisher_policy(device='cpu'):
    method = recipe()
    method['device'] = device
    method['g'].update(lookup='lda', fit_dtype='float32', solver='exact',
        constraint='affine', l1=0, lda=dict(features=24, anchors=32,
        shrinkage=1., sigma_scale=.5, seed=9, fisher_rank=6, teacher='sparsemax', adaptive_support='coverage'))
    trajectories = _trajectories(demos=4, length=32)
    generator = np.random.default_rng(831)
    for trajectory in trajectories:
        for key, value in trajectory.items():
            trajectory[key] = value + generator.normal(0, 2, value.shape)
    policy = Policy(trajectories, _data_config(), method,
                    fit_residuals=True)
    policy.fit_g_lookup()
    policy._project_hankel()
    policy.support_selector.config['adaptive_support'] = 'coverage'
    policy.support_selector.payload['support_scale'] = 1.
    return policy


@integration
class TargetReadoutIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = fisher_policy()
        cls.rows = torch.tensor([3, 7, 15, 35, 39, 47, 65, 72, 105])

    def setUp(self):
        self.policy.target_execution_options = dict(
            target_execution='legacy', target_retrieval_precision='fp64')
        self.policy._target_readout_totals = {}

    def assert_result_close(self, actual, expected):
        torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=2e-5)
        for left, right in zip(actual[1:3], expected[1:3]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertIsInstance(actual[3], RaggedColumns)
        torch.testing.assert_close(actual[3].values, expected[3].values, rtol=0, atol=0)
        torch.testing.assert_close(actual[3].offsets, expected[3].offsets, rtol=0, atol=0)

    def test_native_policy_uses_batched_prior_and_keeps_every_neighbor(self):
        p, rows = self.policy, self.rows
        self.assertTrue(_eligible(p, p._queries(rows)))
        for leave_demo_out in (False, True):
            p.target_execution_options['target_execution'] = 'legacy'
            expected = p._offline_g(rows, leave_demo_out=leave_demo_out)
            p.target_execution_options['target_execution'] = 'batched'
            actual = p._offline_g(rows, leave_demo_out=leave_demo_out)
            self.assert_result_close(actual, expected)
            self.assertEqual(p._target_readout_last['projection_calls'], 1)
            self.assertEqual(p._target_readout_last['rows'], len(rows))
            self.assertEqual(p._target_readout_last['selected_columns'], len(actual[3].values))
            self.assertGreater(len(actual[3].counts.unique()), 1)

    @cuda
    def test_unindexed_cuda_device_enters_real_batched_route(self):
        p = fisher_policy('cuda')
        # Configurations use unindexed "cuda" while actual tensors resolve to
        # cuda:N. Dispatch must compare resolved devices, or it silently falls
        # back despite an explicitly selected target_execution='batched'.
        self.assertEqual(p.device, torch.device('cuda'))
        self.assertIsNone(p.device.index)
        rows = self.rows.to(p.hankel_starts.device)
        self.assertIsNotNone(rows.device.index)
        self.assertTrue(_eligible(p, p._queries(rows)))
        p._target_readout_totals = {}
        for leave_demo_out in (False, True):
            p.target_execution_options['target_execution'] = 'legacy'
            expected = p._offline_g(rows, leave_demo_out=leave_demo_out)
            p.target_execution_options['target_execution'] = 'batched'
            actual = p._offline_g(rows, leave_demo_out=leave_demo_out)
            self.assert_result_close(actual, expected)
            self.assertTrue(torch.isfinite(actual[0]).all())
            self.assertEqual(p._target_readout_last['rows'], len(rows))
            self.assertEqual(p._target_readout_last['projection_calls'], 1)
            self.assertGreater(p._target_readout_last['solve_batches'], 0)
            self.assertEqual(p._target_readout_last['selected_columns'], len(actual[3].values))
        self.assertEqual(p._target_readout_totals['rows'], 2 * len(rows))

    def test_bounded_tiles_reuse_projection_and_keep_all_supports(self):
        p, rows = self.policy, self.rows
        expected = p._offline_g(rows)
        p.target_execution_options['target_execution'] = 'batched'
        with patch('dpc_nldl.target_readout.WORKSPACE_BYTES', 1):
            actual = p._offline_g(rows)
        self.assert_result_close(actual, expected)
        self.assertEqual(p._target_readout_last['projection_calls'], 1)
        self.assertEqual(p._target_readout_last['solve_batches'], len(rows))

    def test_unsupported_math_and_private_overrides_keep_grouped_solver(self):
        p, rows = self.policy, self.rows
        queries = p._queries(rows)
        selected = p.support_selector.select(queries, target_rows=rows)
        for name, value in (('method', 'deepc_g_direct'),
                            ('g_fit_dtype', torch.float64), ('g_weights', 'uniform')):
            with self.subTest(setting=name), patch.object(p, name, value):
                self.assertIsNone(offline_prior(p, rows, selected, queries,
                                               supports_validated=True))
        expected = p._offline_g(rows)
        p.target_execution_options['target_execution'] = 'batched'
        with patch.object(p, '_solve_affine', wraps=p._solve_affine) as original:
            actual = p._offline_g(rows)
            self.assertGreater(original.call_count, 0)
        self.assert_result_close(actual, expected)
        self.assertEqual(p._target_readout_totals, {})

    def test_bad_support_exclusion_is_not_bypassed(self):
        p, rows = self.policy, self.rows
        invalid = [(torch.searchsorted(p.hankel_starts, row)[None], torch.zeros(1))
                   for row in rows]
        p.target_execution_options['target_execution'] = 'batched'
        with patch.object(p.support_selector, 'select', return_value=invalid):
            with self.assertRaisesRegex(RuntimeError, 'target action entered'):
                p._offline_g(rows)


if __name__ == '__main__':
    unittest.main()
