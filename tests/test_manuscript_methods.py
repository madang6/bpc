"""End-to-end contracts for the five retained manuscript choices."""
import copy
import unittest
from tests.suites import integration
from unittest.mock import patch

import torch

from dpc_nldl.bpc import Policy
from dpc_nldl.ragged_supports import RaggedColumns
from tests.support.policies import data_config, manuscript_fixture


@integration
class ManuscriptMethodsTest(unittest.TestCase):
    def test_retained_methods_fit_act_exclude_targets_and_reload(self):
        base, bank = manuscript_fixture()
        for name in ('fisher', 'action_ridge', 'nhc', 'nnc', 'nlr'):
            with self.subTest(method=name):
                method = copy.deepcopy(base)
                if name == 'action_ridge':
                    method['g']['lookup'] = 'action_ridge'
                    method['g'].pop('lda')
                if name == 'nhc':
                    method['g']['weights'] = 'uniform'
                if name == 'nlr':
                    method['g'].update(lookup='past', lookup_rates=[1.],
                                       align_history=False, align_future=False)
                    method['g'].pop('lda')
                    method['residual']['action_source'] = 'native'
                correct = name != 'nnc'
                model = Policy(bank, data_config(), method, fit_residuals=correct)
                with patch('torch.optim.Adam', side_effect=AssertionError('unexpected training method')):
                    model.fit()
                rows = model.trajectory_starts[:3] + 5
                query = model._queries(rows)
                split = model.history * model.action_dim
                actions = query[:, :split].reshape(len(rows), model.history, model.action_dim)
                states = query[:, split:].reshape(len(rows), model.history, model.state_dim)
                actual, trace = model.direct_actions(actions, states)
                self.assertEqual(actual.shape, (len(rows), model.action_dim))
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(trace['prior_z'], trace['base_z']+trace['residual_z'])

                # Held-out target windows and demonstrations remain excluded
                # under fixed top-K, adaptive supports and uniform readout.
                _, _, retained, neighbors = model._offline_g(rows, leave_demo_out=True)
                self.assertTrue((retained == 0).all())
                for index in range(len(rows)):
                    endpoints = (neighbors[index][0] if isinstance(neighbors, RaggedColumns)
                                 else neighbors[index])
                    self.assertTrue((model.trajectory_of[endpoints] != model.trajectory_of[rows[index]]).all())

                if name == 'nnc':
                    self.assertIsNone(model.W_gip)
                    self.assertTrue(torch.equal(trace['residual_z'], torch.zeros_like(actual)))
                    torch.testing.assert_close(actual, trace['base_z']*model.action_scale+model.action_mean)
                if name == 'nhc':
                    selected = model.support_selector.select(query)
                    for index, (columns, _) in enumerate(selected):
                        supports = columns[None]
                        rates = model._rate_trace(query[index:index+1], supports)['hankel_rates']
                        first = model._future_columns(supports, rates=rates)[..., :model.action_dim]
                        torch.testing.assert_close(trace['base_z'][index], first.mean(1)[0])
                if name == 'nlr':
                    self.assertIsNone(model.g_lookup_matrix)
                    expected = torch.cdist(query, model._past_columns(
                        torch.arange(len(model.hankel_starts)))).topk(model.g_supports, largest=False).indices
                    torch.testing.assert_close(trace['hankel_supports'], expected)

                restored = Policy(bank, data_config(), method,
                    fitted=copy.deepcopy(model.fitted_payload({'method': name})), fit_residuals=correct)
                reloaded, restored_trace = restored.direct_actions(actions, states)
                torch.testing.assert_close(reloaded, actual, rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(restored_trace['base_z'], trace['base_z'], rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(restored_trace['residual_z'], trace['residual_z'], rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
