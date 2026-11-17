"""Continuation, delta units, boundary/leakage and deployed-path checks."""
import copy
import unittest
from tests.suites import cuda

import torch

from bpc.artifacts import _target_key
from bpc.pipeline import fit_groups
from tests.support.policies import interpolation_policy as _policy
from tests.support.policies import toy_policy
from tests.support.policies import data_config as _data_config, method_config as _method_config


class TemporalFutureTest(unittest.TestCase):
    def policy(self, rates=(.5, .75, 1., 1.5, 2.)):
        p = _policy(rates, offset=1)
        p.g_align_history = p.g_align_future = True
        return p

    def test_past_time_map_continues_into_actions_and_shifted_states(self):
        p = self.policy()
        supports = torch.tensor([[4], [13]])
        endpoint = p.hankel_starts[supports] + p.history - 1
        for rate in p.g_lookup_rates:
            rates = torch.full(supports.shape, rate, dtype=p.dtype)
            for states in (False, True):
                source = p.state[:, :p.proprio_dim] if states else p.action
                pos = endpoint[..., None] + rate * torch.arange(1, p.horizon + 1).double()
                pos += p.g_state_offset if states else 0
                lo, hi = pos.floor().long(), pos.ceil().long()
                expected = source[lo] + (pos - lo)[..., None] * (source[hi] - source[lo])
                actual = p._future_columns(supports, rates=rates, states=states)
                torch.testing.assert_close(actual, expected if states else expected.flatten(-2), rtol=0, atol=0)
            if rate == 1:
                self.assertTrue(torch.equal(p._future_columns(supports, rates=rates), p.uf.T[supports]))

    def test_fast_future_boundary_is_excluded_from_search_and_rate_selection(self):
        p = self.policy((1., 2.))
        col = torch.tensor([len(p.hankel_starts) - 1])
        query = p._past_columns(col, rate=2.)
        trace = p._rate_trace(query, col[:, None])
        self.assertEqual(trace['hankel_rates'].item(), 1.)
        indices, distances, _ = p._hankel_topk(query, len(p.hankel_starts))
        observed = distances[0, indices[0] == col.item()]
        torch.testing.assert_close(observed, (query - p._past_columns(col)).norm(dim=1))
        with self.assertRaisesRegex(ValueError, 'boundary'):
            p._future_columns(col[:, None], rates=torch.tensor([[2.]]))

    def test_first_interpolated_future_cannot_contain_fitting_target(self):
        p = self.policy((1., 2.))
        col = torch.tensor([3])
        query = p._past_columns(col, rate=2.)
        target = p.hankel_starts[col] + p.history + 1
        self.assertEqual(p._rate_trace(query, col[:, None])['hankel_rates'].item(), 2.)
        self.assertEqual(p._rate_trace(query, col[:, None], target)['hankel_rates'].item(), 1.)
        indices, distances, _ = p._hankel_topk(query, len(p.hankel_starts), target_rows=target)
        torch.testing.assert_close(distances[0, indices[0] == col.item()],
                                   (query - p._past_columns(col)).norm(dim=1))

    def test_delta_scaling_uses_raw_units_and_leaves_other_controls_unchanged(self):
        p = toy_policy(5)
        p.g_rate_scaled_action_dims = (0,)
        raw = torch.tensor([[[.6, -.7], [.6, .8]]], dtype=p.dtype)
        values = (raw - p.action_mean) / p.action_scale
        for rate in (.5, 1., 2.):
            result = p._retime_action_values(values, rate) * p.action_scale + p.action_mean
            expected = raw.clone()
            expected[..., 0] *= rate
            torch.testing.assert_close(result, expected)
        # Constant delta commands must shrink under slowing; resampling alone cannot.
        self.assertEqual(float((p._retime_action_values(values, .5) * p.action_scale
                                + p.action_mean)[0, 0, 0]), .3)

    def test_past_and_future_share_delta_scaling(self):
        p = self.policy((.5, 1.))
        col = torch.tensor([[4]])
        past = p._past_columns(col, rate=.5)
        future = p._future_columns(col, rates=torch.tensor([[.5]]))
        p.g_rate_scaled_action_dims = (0,)
        scaled_past = p._past_columns(col, rate=.5)
        torch.testing.assert_close(scaled_past[..., :p.history * p.action_dim],
                                   p._retime_action_values(past[..., :p.history * p.action_dim]
                                                          .reshape(1, 1, p.history, p.action_dim), .5).flatten(-2))
        torch.testing.assert_close(scaled_past[..., p.history * p.action_dim:],
                                   past[..., p.history * p.action_dim:])
        torch.testing.assert_close(p._future_columns(col, rates=torch.tensor([[.5]])),
                                   p._retime_action_values(future.reshape(1, 1, p.horizon, p.action_dim), .5).flatten(-2))

    def test_prior_uses_same_rate_in_fit_and_online_readout(self):
        p = self.policy()
        p.g_constraint, p.g_solver, p.g_l1 = 'affine', 'exact', 0
        rows = torch.tensor([5, 19])
        query = p._queries(rows)
        supports, _, _ = p._hankel_topk(query, 4, target_rows=rows)
        plane = p._reconstruction_columns(supports, query, torch.float32, rows)
        prior, coefficients = p._g_prior(plane, supports, query.float(), rows=rows)
        future = p._future_columns(supports, query, rows).float()
        torch.testing.assert_close(prior, torch.einsum('bkh,bk->bh', future, coefficients).to(p.dtype))
        packed, packed_coefficients = p._g_prior(None, supports, query.float(), rows=rows)
        torch.testing.assert_close(packed, prior)
        torch.testing.assert_close(packed_coefficients, coefficients)

    def test_direct_scalar_batch_parity_and_rate_one_compatibility(self):
        p = toy_policy(5)
        p.g_constraint, p.g_solver, p.g_l1 = 'affine', 'exact', 0
        p.g_align_history = p.g_align_future = True
        p.g_lookup_rates = (.5, 1., 2.)
        p.samples = len(p.action)
        p.trajectory_starts = torch.tensor([0])
        p.hankel_trajectory = torch.zeros_like(p.hankel_starts)
        rng = torch.Generator().manual_seed(12)
        actions = torch.randn(3, 5, 2, generator=rng, dtype=p.dtype)
        states = torch.randn(3, 5, 4, generator=rng, dtype=p.dtype)
        for dims in ((), (0,)):
            p.g_rate_scaled_action_dims = dims
            p._rate_keys = None
            output, trace = p.direct_actions(actions, states)
            for row in range(3):
                scalar, st = p.direct_actions(actions[row:row+1], states[row:row+1])
                torch.testing.assert_close(output[row], scalar[0], rtol=1e-5, atol=1e-5)
                for key in ('hankel_rates', 'g_coefficients', 'predicted_action_z', 'predicted_proprio_z'):
                    torch.testing.assert_close(trace[key][row], st[key][0], rtol=1e-5, atol=1e-5)
        p.g_lookup_rates, p._rate_keys = (1.,), None
        aligned = p.direct_actions(actions, states)
        p.g_align_future = False
        original = p.direct_actions(actions, states)
        self.assertTrue(torch.equal(aligned[0], original[0]))
        for key in aligned[1]:
            self.assertTrue(torch.equal(aligned[1][key], original[1][key]), key)

    def test_compact_temporal_recipe_preserves_archived_cache_identity(self):
        from bpc.pipeline import temporal_method
        config = {'data': _data_config(), 'method': _method_config(),
                  'methods': {'metric': {'base': 'bpc', 'g_lookup': 'action_ridge'}}}
        config['method']['g'].update(lookup_rates=[.5, 1, 2], state_offset=1, future_consistency=0)
        config['method']['residual']['lookup'] = 'g_supports'
        expanded = copy.deepcopy(config)
        expanded['method'] = temporal_method(config['method'])
        self.assertEqual(fit_groups(config)[0]['key'], fit_groups(expanded)[0]['key'])
        self.assertEqual(_target_key(config, config['method']),
                         _target_key(expanded, expanded['method']))
        for section, key, value in [('g', 'align_history', False),
                                    ('g', 'align_future', False),
                                    ('residual', 'action_source', 'native')]:
            invalid = copy.deepcopy(config)
            invalid['method'][section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'superseded'):
                fit_groups(invalid)
        actuator = copy.deepcopy(config)
        actuator['method']['g']['rate_scaled_action_dims'] = []
        self.assertEqual(temporal_method(actuator['method'])['g']['rate_scaled_action_dims'], [])
        self.assertNotEqual(fit_groups(config)[0]['key'], fit_groups(actuator)[0]['key'])

    @cuda
    def test_cuda_temporal_readout_matches_cpu(self):
        p = toy_policy(5)
        p.g_constraint, p.g_solver, p.g_l1 = 'affine', 'exact', 0
        p.g_align_history = p.g_align_future = True
        p.g_lookup_rates = (.5, 1., 2.)
        p.g_rate_scaled_action_dims = (0,)
        p.samples = len(p.action)
        p.trajectory_starts = torch.tensor([0])
        p.hankel_trajectory = torch.zeros_like(p.hankel_starts)
        rng = torch.Generator().manual_seed(321)
        actions = torch.randn(2, 5, 2, generator=rng, dtype=p.dtype)
        states = torch.randn(2, 5, 4, generator=rng, dtype=p.dtype)
        expected, trace = p.direct_actions(actions, states)
        for key, value in vars(p).copy().items():
            if isinstance(value, torch.Tensor):
                setattr(p, key, value.cuda())
        p.device, p._rate_keys = torch.device('cuda'), None
        result, actual = p.direct_actions(actions.cuda(), states.cuda())
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-4, atol=1e-5)
        for key in ('hankel_supports', 'hankel_rates', 'g_coefficients', 'predicted_action_z'):
            torch.testing.assert_close(actual[key].cpu(), trace[key], rtol=1e-4, atol=1e-5)
