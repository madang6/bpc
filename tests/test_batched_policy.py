import unittest
from tests.suites import integration
from unittest.mock import patch

import numpy as np
import torch


from tests.support.policies import toy_policy
from tests.support.numerics import literal_readout


class BatchedPolicyTest(unittest.TestCase):
    def test_large_query_keeps_scalar_projection_arithmetic(self):
        p = toy_policy(10)
        generator = torch.Generator().manual_seed(53)
        queries = 25000 * torch.randn(8, p.state_dim, generator=generator, dtype=p.dtype)
        neighbors = torch.randint(len(p.state), (8, p.g_supports), generator=generator)
        actual = p._query_features(queries, neighbors) @ p.residual_weights
        expected = []
        for query, selected in zip(queries, neighbors):
            bank = torch.cat([p.state[selected], p.action[selected]], 1).to(p.rff_dtype)
            phase = bank @ p.rff_bank_weight + p.rff_bias
            phase.sub_(query[None].to(p.rff_dtype) @ p.rff_query_weight)
            expected.append(phase.cos().mean(0).mul_(p.rff_scale) @ p.residual_weights)
        torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-5)

    def test_batched_direct_matches_literal_scalar_readout(self):
        generator = torch.Generator().manual_seed(29)
        for history in (5, 10):
            p = toy_policy(history)
            for count in (1, 8, 24):
                states = torch.randn(count, history, 4, generator=generator, dtype=p.dtype)
                actions = torch.randn(count, history, 2, generator=generator, dtype=p.dtype)
                output, trace = p.direct_actions(actions, states)
                for i in range(count):
                    supports, prior, coefficients, residual = literal_readout(p, actions[i], states[i])
                    torch.testing.assert_close(trace["hankel_supports"][i], supports, rtol=0, atol=0)
                    torch.testing.assert_close(trace["g_coefficients"][i], coefficients,
                                               rtol=1e-11, atol=1e-11)
                    torch.testing.assert_close(trace["base_z"][i], prior[:2], rtol=1e-11, atol=1e-11)
                    torch.testing.assert_close(trace["residual_z"][i], residual, rtol=1e-5, atol=1e-5)
                    expected = (prior[:2] + residual) * p.action_scale + p.action_mean
                    torch.testing.assert_close(output[i], expected, rtol=1e-5, atol=1e-5)

    @integration
    def test_scalar_entry_matches_the_direct_readout(self):
        p = toy_policy(5)
        states, actions = p.state[:5].clone(), p.action[:5].clone()
        raw_state = states[-1] * p.state_scale + p.state_mean
        raw_actions = actions * p.action_scale + p.action_mean
        p._state_history = [torch.zeros(4), *states[:-1]]
        p._action_history = [torch.zeros(2), *raw_actions[:-1]]
        canonical_actions = (raw_actions - p.action_mean) / p.action_scale
        states[-1] = p.scale_state(raw_state)
        supports, prior, _, residual = literal_readout(p, canonical_actions, states)
        actual = p.action_for(raw_state.numpy(), raw_actions[-1].numpy())
        np.testing.assert_allclose(actual, ((prior[:2] + residual) * p.action_scale
                                            + p.action_mean).numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(p.last_trace["hankel_supports"], supports.numpy())
        np.testing.assert_allclose(p.last_trace["base_z"], prior[:2], rtol=1e-11, atol=1e-11)
        self.assertNotIn("g_future_consistency", p.last_trace)

    def test_nnc_omits_rff_computation_and_retains_the_same_base(self):
        p = toy_policy(5)
        actions, states = p.action[:15].reshape(3, 5, 2), p.state[:15].reshape(3, 5, 4)
        _, reference = p.direct_actions(actions, states)
        p.select_method('bpc_nnc')
        p.residual_weights = None
        with patch.object(p, '_query_features', side_effect=AssertionError('NNC computed RFF')):
            output, trace = p.direct_actions(actions, states)
        torch.testing.assert_close(trace['base_z'], reference['base_z'], rtol=0, atol=0)
        torch.testing.assert_close(trace['residual_z'], torch.zeros_like(trace['base_z']))
        torch.testing.assert_close(output, trace['base_z'] * p.action_scale + p.action_mean)
        torch.testing.assert_close(trace['rff_norm'], torch.zeros(3, dtype=p.dtype))

    @integration
    def test_disabling_trace_bounds_scalar_history_without_changing_actions(self):
        recorded, bounded = toy_policy(5), toy_policy(5)
        bounded._record_history = False
        for policy in (recorded, bounded):
            policy._state_history = list(policy.state[:policy.history])
            policy._action_history = list(policy.action[:policy.history]
                                          * policy.action_scale + policy.action_mean)
        for row in range(5, 15):
            state = (recorded.state[row] * recorded.state_scale + recorded.state_mean).numpy()
            previous = (recorded.action[row] * recorded.action_scale + recorded.action_mean).numpy()
            np.testing.assert_array_equal(recorded.action_for(state, previous),
                                          bounded.action_for(state, previous))
            self.assertEqual(len(bounded._state_history), bounded.history + 1)
            self.assertEqual(len(bounded._action_history), bounded.history + 1)
            self.assertIsNone(bounded.last_trace)
        self.assertEqual(len(recorded._state_history), 15)
        self.assertIsNotNone(recorded.last_trace)


if __name__ == "__main__":
    unittest.main()
