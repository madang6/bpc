"""Frozen literal future gathering for independent fused-kernel comparisons."""
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch

from dpc_nldl.fused_temporal_future import future_columns
from tests.support.temporal_gather import future_fixture


def literal_future(policy, supports, rates, states=False):
    """Checkpoint5eee261 _future_columns, without any optimized dispatch."""
    offsets = torch.arange(1, policy.horizon+1, device=policy.device)
    endpoint = policy.hankel_starts[supports]+policy.history-1
    source = policy.state[:, :policy.proprio_dim] if states else policy.action
    shift = policy.g_state_offset if states else 0
    result = (source[endpoint[..., None]+offsets+shift] if states else
              policy.uf.T[supports].reshape(*supports.shape, policy.horizon, policy.action.shape[1]))
    result = result.clone()
    stops = policy._trajectory_stops()
    for rate in policy.g_lookup_rates:
        mask = rates == rate
        if rate == 1 or not mask.any():
            continue
        positions = endpoint[mask][:, None]+rate*offsets.double()+shift
        upper = positions.ceil().long()
        if bool((upper[:, -1] >= stops[policy.hankel_trajectory[supports[mask]]]).any()):
            raise ValueError('interpolated future crosses a demonstration boundary')
        values = policy._interpolate(source, positions)
        result[mask] = values if states else policy._retime_action_values(values, rate)
    return result if states else result.flatten(-2)


class FusedTemporalFutureCPUTests(unittest.TestCase):
    def test_cpu_falls_back_without_loading_extension(self):
        policy = future_fixture()
        supports = torch.tensor([[0]])
        with patch('dpc_nldl.fused_temporal_future.extension', side_effect=AssertionError('CUDA loaded')):
            self.assertIsNone(future_columns(policy, supports, torch.ones(1, 1)))

    def test_literal_native_actions_keep_cached_values_and_states_use_shift(self):
        policy = future_fixture(offset=1)
        supports = torch.tensor([[0, 3]])
        rates = torch.ones(supports.shape)
        actions = literal_future(policy, supports, rates)
        torch.testing.assert_close(actions, policy.uf.T[supports], rtol=0, atol=0)
        endpoint = policy.hankel_starts[supports]+policy.history-1
        positions = endpoint[..., None]+torch.arange(1, policy.horizon+1)+1
        torch.testing.assert_close(literal_future(policy, supports, rates, states=True),
                                   policy.state[positions, :policy.proprio_dim], rtol=0, atol=0)

    def test_literal_matches_policy_before_or_after_independent_dispatch(self):
        policy = future_fixture(dtype=torch.float32)
        supports = torch.tensor([[0, 2, 4, 5], [6, 3, 1, 0]])
        rates = torch.tensor([[.5, .75, 1., 1.1], [1.5, 2., .9, float('nan')]])
        for states in (False, True):
            torch.testing.assert_close(literal_future(policy, supports, rates, states),
                policy._future_columns(supports, rates=rates, states=states), rtol=0, atol=0)


@cuda
class FusedTemporalFutureCUDATests(unittest.TestCase):
    def compare(self, policy, supports, rates, states=False):
        expected = literal_future(policy, supports, rates, states)
        actual = future_columns(policy, supports, rates, states)
        self.assertIsNotNone(actual, 'CUDA extension unavailable or unexpected fallback')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    def test_mixed_rates_source_native_precisions_offsets_and_state_futures(self):
        supports = torch.tensor([[0, 2, 4, 5], [6, 3, 1, 0]], device='cuda')
        for dtype in (torch.float32, torch.float64):
            for native_dtype in (dtype,):
                for rate_dtype in (torch.float32, torch.float64):
                    for offset in (0, 1):
                        policy = future_fixture('cuda', dtype, native_dtype, offset)
                        rates = torch.tensor([[.5, .75, 1., 1.1], [1.5, 2., .9, float('nan')]],
                                             device='cuda', dtype=rate_dtype)
                        for states in (False, True):
                            with self.subTest(source=dtype, native=native_dtype, rate=rate_dtype,
                                              offset=offset, states=states):
                                self.compare(policy, supports, rates, states)

    def test_explicit_state_shift_boundary_does_not_change_action_times(self):
        policy = future_fixture('cuda', offset=1)
        policy.hankel_starts[2] = 40  # e49: action end63 is legal, state end64 is not.
        supports = torch.tensor([[2]], device='cuda')
        rates = torch.full((1, 1), 2., device='cuda')
        self.compare(policy, supports, rates, states=False)
        for decode in (lambda: literal_future(policy, supports, rates, states=True),
                       lambda: future_columns(policy, supports, rates, states=True)):
            with self.assertRaisesRegex(ValueError, 'interpolated future crosses'):
                decode()

    def test_rounded_rate_intermediate_boundary_remains_an_error(self):
        policy = future_fixture('cuda', offset=0)
        policy.g_lookup_rates = (1., 13/7+1e-9, 13/7-1e-9)
        policy.hankel_starts[1] = 41  # e50, first matched rate ends just above63.
        supports = torch.tensor([[1]], device='cuda')
        rates = torch.full((1, 1), 13/7, device='cuda', dtype=torch.float32)
        for decode in (lambda: literal_future(policy, supports, rates),
                       lambda: future_columns(policy, supports, rates)):
            with self.assertRaisesRegex(ValueError, 'interpolated future crosses'):
                decode()

    def test_noncontiguous_rate_tensor_empty_batch_and_clean_fallback(self):
        policy = future_fixture('cuda')
        supports = torch.tensor([[[0, 1], [3, 4]]], device='cuda')
        rates = torch.tensor([[[.5, 1.5], [.75, 2.]]], device='cuda').transpose(-1, -2)
        self.compare(policy, supports, rates)
        empty = future_columns(policy, supports[:, :0], rates[:, :0], states=True)
        self.assertEqual(empty.shape, (1, 0, 2, policy.horizon, policy.proprio_dim))
        with patch('dpc_nldl.fused_temporal_future.extension', return_value=None):
            self.assertIsNone(future_columns(policy, supports, rates))
        self.assertIsNone(future_columns(policy, supports, None))
        policy.action_mean = policy.action_mean.cpu()
        self.assertIsNone(future_columns(policy, supports, rates))

    def test_mismatched_native_precision_preserves_original_assignment_error(self):
        policy = future_fixture('cuda', dtype=torch.float32, native_dtype=torch.float64)
        supports = torch.tensor([[0]], device='cuda')
        rates = torch.full((1, 1), .5, device='cuda')
        self.assertIsNone(future_columns(policy, supports, rates))
        with self.assertRaisesRegex(RuntimeError, 'dtypes match'):
            literal_future(policy, supports, rates)

    def test_initial_native_access_failure_is_not_hidden_by_valid_slow_rate(self):
        policy = future_fixture('cuda', offset=0)
        supports = torch.tensor([[6]], device='cuda')
        rates = torch.full((1, 1), .5, device='cuda')
        policy.hankel_starts[6] = 114  # e123: native reaches130, slow reaches126.5.
        self.assertIsNone(future_columns(policy, supports, rates, states=True))
        policy.uf = policy.uf[:, :6]  # Column6 native access fails even if retimed.
        self.assertIsNone(future_columns(policy, supports, rates, states=False))
        # Native failure across the batch precedes another row's rate guard.
        policy.hankel_starts[2] = 41
        supports = torch.tensor([[6, 2]], device='cuda')
        rates = torch.tensor([[.5, 2.]], device='cuda')
        self.assertIsNone(future_columns(policy, supports, rates, states=True))
        self.assertIsNone(future_columns(policy, supports, rates, states=False))


if __name__ == '__main__':
    unittest.main()
