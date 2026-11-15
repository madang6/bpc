"""Exact temporal history gather, including source rounding and rate decoding."""
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch

from dpc_nldl.fused_temporal_history import reconstruction_columns
from tests.support.temporal_gather import policy_fixture


def literal_per_support(policy, supports, rates, dtype):
    """Decode each support's last matching configured rate, preserving ordering."""
    rows = []
    for support, chosen in zip(supports.reshape(-1), rates.reshape(-1)):
        effective = 1.
        for rate in policy.g_lookup_rates:
            if rate != 1 and bool(chosen == rate):
                effective = rate
        rows.append(policy._past_columns(support.reshape(1), dtype, rate=effective))
    return torch.cat(rows).reshape(*supports.shape, -1)


def literal_reconstruction(policy, supports, rates, dtype=None):
    """Original native gather followed by one masked overwrite per time rate.

    Keep this independent of Policy integration so adding the optimized call to
    _reconstruction_columns cannot turn the CUDA comparison into a self-test.
    """
    plane = policy._past_columns(supports, dtype)
    for rate in policy.g_lookup_rates:
        if rate != 1:
            mask = rates == rate
            if mask.any():
                plane[mask] = policy._past_columns(supports[mask], dtype, rate)
    return plane


class FusedTemporalHistoryCPUTests(unittest.TestCase):
    def test_cpu_is_a_clean_fallback_without_loading_cuda(self):
        policy = policy_fixture()
        supports = torch.tensor([[0, 3]])
        rates = torch.tensor([[.5, 2.]])
        with patch('dpc_nldl.fused_temporal_history.extension', side_effect=AssertionError('CUDA loaded')):
            self.assertIsNone(reconstruction_columns(policy, supports, rates, torch.float32))

    def test_literal_rate_mapping_cast_order_and_native_default(self):
        supports = torch.tensor([[0, 2, 4, 5], [6, 3, 1, 0]])
        for source_dtype in (torch.float32, torch.float64):
            for output_dtype in (None, torch.float32, torch.float64):
                policy = policy_fixture(source_dtype=source_dtype)
                rates = torch.tensor([[.5, .75, 1., 1.1], [1.5, 2., .9, float('nan')]],
                                     dtype=torch.float32)
                expected = literal_per_support(policy, supports, rates, output_dtype)
                actual = policy._reconstruction_columns(supports, None, output_dtype, rates=rates)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_literal_demo_boundary_is_checked_before_interpolation(self):
        policy = policy_fixture()
        policy.hankel_starts[3] = 65
        with self.assertRaisesRegex(ValueError, 'crosses a demonstration boundary'):
            policy._past_columns(torch.tensor([3]), rate=2.)


@cuda
class FusedTemporalHistoryCUDATests(unittest.TestCase):
    def compare(self, policy, supports, rates, dtype):
        expected = literal_reconstruction(policy, supports, rates, dtype)
        actual = reconstruction_columns(policy, supports, rates, dtype)
        self.assertIsNotNone(actual, 'CUDA extension unavailable or unexpectedly fell back')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    def test_mixed_rates_dtypes_offsets_and_action_scaling_are_bitwise_equal(self):
        supports = torch.tensor([[0, 2, 4, 5], [6, 3, 1, 0]], device='cuda')
        for source_dtype in (torch.float32, torch.float64):
            for output_dtype in (None, torch.float32, torch.float64):
                for rate_dtype in (torch.float32, torch.float64):
                    for offset in (0, 1):
                        with self.subTest(source=source_dtype, output=output_dtype,
                                          rates=rate_dtype, offset=offset):
                            policy = policy_fixture('cuda', source_dtype, offset=offset)
                            rates = torch.tensor([[.5, .75, 1., 1.1], [1.5, 2., .9, float('nan')]],
                                                 device='cuda', dtype=rate_dtype)
                            self.compare(policy, supports, rates, output_dtype)

    def test_no_scaling_arbitrary_leading_shape_and_noncontiguous_rates(self):
        policy = policy_fixture('cuda', scaled=(), history=5)
        supports = torch.tensor([[[0, 1], [3, 4]]], device='cuda')
        rates = torch.tensor([[[.5, 1.5], [.75, 2.]]], device='cuda').transpose(-1, -2)
        self.compare(policy, supports, rates, torch.float32)

    def test_rate_tensor_rounding_and_last_matching_configured_rate(self):
        # Both configured values have the same FP32 representation. The old
        # overwrite loop uses the last Python double, not the rounded tensor.
        policy = policy_fixture('cuda', rates=(1., 1.1, 1.10000001))
        supports = torch.tensor([[0, 1, 3]], device='cuda')
        rates = torch.full(supports.shape, 1.1, device='cuda', dtype=torch.float32)
        self.compare(policy, supports, rates, None)

    def test_demo_boundary_raises_original_error_and_storage_bounds_fall_back(self):
        policy = policy_fixture('cuda')
        policy.hankel_starts[3] = 65
        supports = torch.tensor([[3]], device='cuda')
        rates = torch.full((1, 1), 2., device='cuda')
        with self.assertRaisesRegex(ValueError, 'crosses a demonstration boundary'):
            reconstruction_columns(policy, supports, rates)
        # Avoid an unsafe CUDA read; the unchanged caller owns the index error.
        policy.hankel_starts[3] = 125
        self.assertIsNone(reconstruction_columns(policy, supports, torch.ones_like(rates)))
        # The batch's initial native gather fails before any temporal guard.
        policy.hankel_starts[4] = 65
        supports = torch.tensor([[3, 4]], device='cuda')
        rates = torch.tensor([[1., 2.]], device='cuda')
        self.assertIsNone(reconstruction_columns(policy, supports, rates))

    def test_intermediate_rounded_rate_boundary_is_not_hidden_by_later_overwrite(self):
        policy = policy_fixture('cuda', rates=(1., 2.00000005, 1.99999995))
        policy.hankel_starts[3] = 73
        supports = torch.tensor([[3]], device='cuda')
        rates = torch.full((1, 1), 2., device='cuda', dtype=torch.float32)
        for decode in (lambda: literal_reconstruction(policy, supports, rates),
                       lambda: reconstruction_columns(policy, supports, rates)):
            with self.assertRaisesRegex(ValueError, 'crosses a demonstration boundary'):
                decode()

    def test_unsupported_layout_and_missing_toolchain_fall_back(self):
        policy = policy_fixture('cuda')
        supports = torch.tensor([[0]], device='cuda')
        rates = torch.full((1, 1), .5, device='cuda')
        with patch('dpc_nldl.fused_temporal_history.extension', return_value=None):
            self.assertIsNone(reconstruction_columns(policy, supports, rates))
        policy.state = policy.state[:, ::2]
        self.assertIsNone(reconstruction_columns(policy, supports, rates))


if __name__ == '__main__':
    unittest.main()
