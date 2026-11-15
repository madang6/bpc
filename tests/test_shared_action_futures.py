"""One unchanged action-future tensor serves G, correction, and diagnostics."""
import unittest
from unittest.mock import patch

import torch

from dpc_nldl.bpc import Policy
from tests.support.policies import temporal_policy


class SharedActionFutureTests(unittest.TestCase):
    def test_readout_and_all_traces_match_separately_generated_retimed_futures(self):
        for source in ('native', 'retimed_first'):
            for weights in ('optimized', 'uniform'):
                with self.subTest(source=source, weights=weights):
                    policy = temporal_policy()
                    policy.g_weights = weights
                    policy.set_residual_action_source(source)
                    actions = policy.action[10:25].reshape(3, 5, 2)
                    states = policy.state[10:25].reshape(3, 5, 4)

                    def separate_futures(*args, **kwargs):
                        # Original behavior: G independently decodes its own
                        # futures; correction/spread use a separate decode.
                        kwargs.pop('futures')
                        return Policy._g_prior(policy, *args, **kwargs)

                    with (patch.object(policy, '_g_prior', side_effect=separate_futures),
                          patch.object(policy, '_future_columns', wraps=policy._future_columns) as reference_calls):
                        expected, expected_trace = policy.direct_actions(actions, states)
                    with patch.object(policy, '_future_columns', wraps=policy._future_columns) as shared_calls:
                        actual, trace = policy.direct_actions(actions, states)

                    reference_action = [c for c in reference_calls.call_args_list if not c.kwargs.get('states')]
                    shared_action = [c for c in shared_calls.call_args_list if not c.kwargs.get('states')]
                    shared_state = [c for c in shared_calls.call_args_list if c.kwargs.get('states')]
                    self.assertEqual(len(reference_action), 2)
                    self.assertEqual(len(shared_action), 1)
                    self.assertEqual(len(shared_state), 1)
                    self.assertTrue(bool((trace['hankel_rates'] != 1).any()))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(set(trace), set(expected_trace))
                    for name in trace:
                        torch.testing.assert_close(trace[name], expected_trace[name], rtol=0, atol=0,
                                                   msg=lambda message: f'{name}: {message}')

    def test_explicit_futures_remain_unmodified_and_default_prior_still_decodes(self):
        policy = temporal_policy()
        actions = policy.action[10:25].reshape(3, 5, 2)
        states = policy.state[10:25].reshape(3, 5, 4)
        query = torch.cat([actions.flatten(1), states.flatten(1)], 1)
        supports, _, _ = policy._hankel_topk(query, policy.g_supports)
        rates = policy._rate_trace(query, supports)['hankel_rates']
        plane = policy._reconstruction_columns(supports, query, rates=rates)
        futures = policy._future_columns(supports, query, rates=rates)
        original = futures.clone()
        for weights in ('optimized', 'uniform'):
            policy.g_weights = weights
            expected = policy._g_prior(plane, supports, query, rates=rates)
            with patch.object(policy, '_future_columns', side_effect=AssertionError('future decoded twice')):
                actual = policy._g_prior(plane, supports, query, rates=rates, futures=futures)
            torch.testing.assert_close(futures, original, rtol=0, atol=0)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
