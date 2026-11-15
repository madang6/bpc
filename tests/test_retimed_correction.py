"""Retimed correction retains G and endpoint state while exposing its action columns."""
import copy
import tempfile
import unittest
from tests.suites import cuda, integration
from pathlib import Path

import torch

from dpc_nldl import fitting
from dpc_nldl.bpc import Policy
from dpc_nldl.artifacts import _required_payload, _target_key, fit_policy, load_policy, prepare_lookups
from dpc_nldl.pipeline import fit_groups
from tests.support.policies import (
    action_ridge_config, data_config, interpolation_policy, method_config,
    small_ridge_fit, temporal_policy, trajectories,
)


class RetimedCorrectionTest(unittest.TestCase):
    @integration
    def test_configured_retimed_load_requires_matching_payload(self):
        method = method_config()
        method['g'].update(lookup='action_ridge', state_offset=1,
                           align_history=True, align_future=True, future_consistency=0,
                           lookup_rates=[.5, 1., 2.])
        method['residual'].update(lookup='g_supports', action_source='retimed_first')
        bank = trajectories(length=12)
        policy = Policy(bank, data_config(), method)
        width = policy.history*(policy.action_dim+policy.state_dim)
        policy.g_lookup_matrix = torch.eye(width, dtype=policy.dtype)
        policy._project_hankel()
        policy.fit()
        payload = policy.fitted_payload({})
        config = dict(data=data_config(), method=method,
                      methods=dict(metric=dict(base='gip_direct', g_lookup='action_ridge')))
        self.assertIn('residual_action_source', _required_payload(fit_groups(config)[0]))
        for value in (None, 'native'):
            with self.subTest(value=value):
                invalid = copy.deepcopy(payload)
                if value is None:
                    invalid.pop('residual_action_source')
                else:
                    invalid['residual_action_source'] = value
                with self.assertRaisesRegex(ValueError, 'matching residual action source payload'):
                    Policy(bank, data_config(), method, fitted=invalid)
                light = Policy(bank, data_config(), method, fitted=invalid,
                               fit_residuals=False, load_operators=False)
                self.assertIsNone(light.W_gip)
        native = copy.deepcopy(method)
        native['residual']['action_source'] = 'native'
        with self.assertRaisesRegex(ValueError, 'superseded'):
            Policy(bank, data_config(), native, fitted=payload)

    @integration
    def test_retimed_config_supports_upstream_preparation_and_reload(self):
        with tempfile.TemporaryDirectory(prefix='dpc-retimed-upstream-') as folder:
            root = Path(folder)
            method = action_ridge_config()
            method['g'].update(state_offset=1, align_history=True, align_future=True,
                               future_consistency=0, lookup_rates=[.5, 1., 2.])
            method['residual'].update(lookup='g_supports', action_source='retimed_first')
            config = dict(experiment=dict(name='retimed', cache_dir=str(root/'cache')),
                          analysis=dict(reference_method='metric'), data=data_config(), method=method,
                          methods=dict(metric=dict(base='gip_direct', g_lookup='action_ridge')))
            torch.save({'data': trajectories(demos=8, length=12)}, root/'bank.pt')
            with small_ridge_fit():
                prepare_lookups(config, root)
            fit_policy(config, root, shard='0/2')
            fit_policy(config, root, shard='1/2')
            fit_policy(config, root)
            full = load_policy(config, root, 'metric')
            self.assertEqual(full.residual_action_source, 'retimed_first')

    def test_native_rate_preserves_all_readout_arithmetic(self):
        p = temporal_policy((1.,))
        actions, states = p.action[:15].reshape(3, 5, 2), p.state[:15].reshape(3, 5, 4)
        expected, expected_trace = p.direct_actions(actions, states)
        p.set_residual_action_source('retimed_first')
        actual, trace = p.direct_actions(actions, states)
        self.assertTrue(torch.equal(actual, expected))
        for key in trace:
            self.assertTrue(torch.equal(trace[key], expected_trace[key]), key)

    def test_correction_uses_retimed_first_action_and_unchanged_g(self):
        p = temporal_policy()
        actions, states = p.action[10:25].reshape(3, 5, 2), p.state[10:25].reshape(3, 5, 4)
        _, native = p.direct_actions(actions, states)
        p.set_residual_action_source('retimed_first')
        output, trace = p.direct_actions(actions, states)
        query = torch.cat([actions.flatten(1), states.flatten(1)], 1)
        supports = trace['hankel_supports']
        neighbors = p._support_endpoints(supports)
        decoded = p.retimed_neighbor_actions(query, neighbors)
        endpoint = neighbors - 1
        positions = endpoint + trace['hankel_rates']
        expected = p._interpolate(p.action, positions)
        for rate in p.g_lookup_rates:
            mask = trace['hankel_rates'] == rate
            expected[mask] = p._retime_action_values(expected[mask], rate)
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
        residual = p._query_features(states[:, -1], neighbors, decoded) @ p.W_gip
        torch.testing.assert_close(trace['residual_z'], residual, rtol=0, atol=0)
        torch.testing.assert_close(output, (trace['base_z'] + residual) * p.action_scale + p.action_mean)
        for key in ('hankel_supports', 'hankel_rates', 'g_coefficients', 'base_z'):
            self.assertTrue(torch.equal(trace[key], native[key]), key)
        self.assertGreater(float((decoded - p.action[neighbors]).norm()), 0)

    def test_cached_endpoint_helper_replays_target_exclusion_and_boundary(self):
        p = interpolation_policy((1., 2.), offset=1)
        p.g_align_history = p.g_align_future = True
        p.g_rate_scaled_action_dims = (0,)
        col = torch.tensor([[3]])
        neighbors = p._support_endpoints(col)
        query = p._past_columns(col[:, 0], rate=2.)
        target = p.hankel_starts[col[:, 0]] + p.history + 1
        online = p.retimed_neighbor_actions(query, neighbors)
        offline = p.retimed_neighbor_actions(query, neighbors, target)
        self.assertGreater(float((online - offline).norm()), 0)
        self.assertTrue(torch.equal(offline, p.action[neighbors]))
        col = torch.tensor([[len(p.hankel_starts)-1]])
        query = p._past_columns(col[:, 0], rate=2.)
        self.assertTrue(torch.equal(
            p.retimed_neighbor_actions(query, p._support_endpoints(col)),
            p.action[p._support_endpoints(col)]))
        with self.assertRaisesRegex(ValueError, 'Hankel columns'):
            p.retimed_neighbor_actions(query, torch.tensor([[p.samples + 1]]))

    def test_delta_phase_atlas_matches_explicit_action_features(self):
        p = temporal_policy()
        rows = torch.arange(8, 15)
        queries = p._queries(rows)
        supports, _, _ = p._hankel_topk(queries, p.g_supports, target_rows=rows)
        neighbors = p._support_endpoints(supports)
        actions = p.retimed_neighbor_actions(queries, neighbors, rows)
        atlas = fitting.feature_atlas(p, rows, neighbors, neighbor_actions=actions)
        direct = p._query_features(p.state[rows], neighbors, actions)
        torch.testing.assert_close(atlas, direct, atol=2e-7, rtol=2e-5)

    @cuda
    def test_retimed_host_paging_preserves_resident_features(self):
        p = temporal_policy()
        p.device = torch.device('cuda')
        for key, value in vars(p).items():
            if torch.is_tensor(value):
                setattr(p, key, value.to(p.device))
        rows = torch.arange(8, 15, device=p.device)
        queries = p._queries(rows)
        supports, _, _ = p._hankel_topk(queries, p.g_supports, target_rows=rows)
        neighbors = p._support_endpoints(supports)
        actions = p.retimed_neighbor_actions(queries, neighbors, rows).float().cpu()
        expected = fitting.feature_atlas(p, rows, neighbors, neighbor_actions=actions)
        p.rff_storage = 'host_paged'
        atlas = fitting.feature_atlas(p, rows, neighbors, neighbor_actions=actions)
        self.addCleanup(atlas.close)
        actual = torch.empty_like(expected)
        atlas.replay(lambda values, start, stop: actual[start:stop].copy_(values))
        self.assertTrue(torch.equal(actual, expected))

    @integration
    def test_payload_roundtrip_after_full_retimed_fit(self):
        method = method_config()
        method['g'].update(lookup='action_ridge', state_offset=1,
                           align_history=True, align_future=True, future_consistency=0,
                           lookup_rates=[.5, 1., 2.])
        method['residual']['lookup'] = 'g_supports'
        method['residual']['action_source'] = 'retimed_first'
        p = Policy(trajectories(length=12), data_config(), method)
        width = p.history * (p.action_dim + p.state_dim)
        p.g_lookup_matrix = torch.eye(width, dtype=p.dtype)
        p._project_hankel()
        p.fit()
        payload = p.fitted_payload({})
        self.assertEqual(payload['residual_action_source'], 'retimed_first')
        restored = Policy(trajectories(length=12), data_config(), method, fitted=payload)
        self.assertEqual(restored.residual_action_source, 'retimed_first')
        self.assertGreater(payload['fit']['rff_fit_seconds']['retimed_actions'], 0)
        with self.assertRaisesRegex(ValueError, 'unknown method'):
            restored.select_method('gip_spc')
        with self.assertRaisesRegex(ValueError, 'unknown'):
            restored.set_residual_action_source('other')
        self.assertEqual(restored.residual_action_source, 'retimed_first')

    @integration
    def test_full_fit_uses_native_basis_and_retimed_ridge_features(self):
        # Exercise both the production atlas path and the direct-feature fallback.
        for atlas in (False, True):
            with self.subTest(atlas=atlas):
                method = method_config()
                method['g'].update(lookup='action_ridge', state_offset=1,
                                   align_history=True, align_future=True, future_consistency=0,
                                   lookup_rates=[.5, 1., 2.], rate_scaled_action_dims=[0])
                method['residual'].update(lookup='g_supports', action_source='retimed_first')
                if atlas:
                    method['rff']['dtype'] = 'float32'
                p = Policy(trajectories(length=16), data_config(), method)
                width = p.history * (p.action_dim + p.state_dim)
                p.g_lookup_matrix = torch.eye(width, dtype=p.dtype)
                p._project_hankel()
                rows = torch.nonzero(p.row_in_trajectory >= p.history).ravel()
                base, _, _, endpoints = p._offline_g(rows)
                target = p.action[rows] - base[:, :p.action_dim]
                neighbors = torch.empty(p.samples, p.g_supports, dtype=torch.long)
                neighbors[rows] = endpoints
                fitting.make_basis(p, rows, neighbors, method['rff']['seed'], method['rff']['sigma_scale'])
                native_omega, native_phase = p.omega.clone(), p.phase.clone()
                actions = p.retimed_neighbor_actions(p._queries(rows), endpoints, rows)
                features = p._query_features(p.state[rows], endpoints, actions)
                expected = torch.linalg.solve(
                    features.double().T @ features.double()
                    + p.rff_ridge * torch.eye(p.rff_dim, dtype=torch.float64),
                    features.double().T @ target.double())
                fitting.fit_rff(p, rows, target, neighbors)
                self.assertTrue(torch.equal(p.omega, native_omega))
                self.assertTrue(torch.equal(p.phase, native_phase))
                torch.testing.assert_close(p.W_gip.double(), expected, rtol=3e-5, atol=3e-6)
                self.assertGreater(p.rff_timings['retimed_actions'], 0)

    def test_feature_source_changes_fit_identity_but_reuses_exact_g_targets(self):
        config = {'data': data_config(), 'method': method_config(),
                  'methods': {'metric': {'base': 'gip_direct', 'g_lookup': 'action_ridge'}}}
        config['method']['residual']['lookup'] = 'g_supports'
        changed = copy.deepcopy(config)
        changed['method']['residual']['action_source'] = 'retimed_first'
        self.assertNotEqual(fit_groups(config)[0]['key'], fit_groups(changed)[0]['key'])
        self.assertEqual(_target_key(config, config['method']), _target_key(changed, changed['method']))


if __name__ == '__main__':
    unittest.main()
