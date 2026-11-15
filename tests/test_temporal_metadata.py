"""Static temporal tensors retain literal values and invalidate on every input change."""
import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from dpc_nldl.temporal_metadata import temporal_constants, trajectory_stops
from tests.support.temporal_gather import future_fixture


def configured(policy):
    return tuple(float(rate) for rate in policy.g_lookup_rates if rate != 1)


def literal_constants(policy, source, *, actions=True):
    rate = torch.tensor(configured(policy), device=source.device, dtype=torch.float64)
    if not actions:
        empty = source.new_empty(0)
        return rate, empty, empty, torch.empty(0, device=source.device, dtype=torch.bool)
    mean = policy.action_mean.to(dtype=source.dtype).contiguous()
    scale = policy.action_scale.to(dtype=source.dtype).contiguous()
    mask = torch.zeros(source.shape[1], device=source.device, dtype=torch.bool)
    if policy.g_rate_scaled_action_dims:
        mask[list(policy.g_rate_scaled_action_dims)] = True
    return rate, mean, scale, mask


class TemporalMetadataTests(unittest.TestCase):
    def compare(self, policy, source, actions=True):
        actual = temporal_constants(policy, source, configured(policy), actions=actions)
        expected = literal_constants(policy, source, actions=actions)
        for left, right in zip(actual, expected):
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.stride(), right.stride())
            self.assertTrue(torch.equal(left.reshape(-1).view(torch.uint8),
                                        right.reshape(-1).view(torch.uint8)))
        return actual

    def test_warm_calls_allocate_no_rate_or_mask_tensors(self):
        policy = future_fixture()
        source = policy.action.float()
        first = self.compare(policy, source)
        with (patch('dpc_nldl.temporal_metadata.torch.tensor', side_effect=AssertionError('new rates')),
              patch('dpc_nldl.temporal_metadata.torch.zeros', side_effect=AssertionError('new mask'))):
            second = temporal_constants(policy, source, configured(policy))
        for old, new in zip(first, second):
            self.assertIs(old, new)

    def test_scaling_mutation_replacement_and_configuration_invalidate(self):
        policy = future_fixture()
        source = policy.action.float()
        previous = self.compare(policy, source)
        def mean_mutation(): policy.action_mean.add_(.17)
        def scale_mutation(): policy.action_scale.mul_(1.37)
        mutations = [mean_mutation, scale_mutation,
            lambda: setattr(policy, 'action_mean', policy.action_mean.clone()),
            lambda: setattr(policy, 'action_scale', policy.action_scale.clone()),
            lambda: setattr(policy, 'g_lookup_rates', (1., .25, .5, 1.5)),
            lambda: setattr(policy, 'g_rate_scaled_action_dims', [1, 3])]
        for mutation in mutations:
            mutation()
            current = self.compare(policy, source)
            self.assertIsNot(current[0], previous[0])
            previous = current
        policy.g_rate_scaled_action_dims[0] = 0
        self.assertIsNot(previous[0], self.compare(policy, source)[0])

    def test_source_identity_version_dtype_and_channel_shape_are_part_of_key(self):
        policy = future_fixture()
        source = policy.action
        first = self.compare(policy, source)
        source.add_(1)
        second = self.compare(policy, source)
        self.assertIsNot(first[0], second[0])
        third = self.compare(policy, source.clone())
        self.assertIsNot(second[0], third[0])
        single = self.compare(policy, source.float())
        self.assertEqual(single[1].dtype, torch.float32)
        policy.action_mean = torch.cat([policy.action_mean, torch.zeros(1)])
        policy.action_scale = torch.cat([policy.action_scale, torch.ones(1)])
        wide = self.compare(policy, torch.zeros(5, 5, dtype=torch.float64))
        self.assertEqual(wide[3].shape, (5,))
        self.assertIsNot(wide[3], third[3])

    def test_dimension_type_mutations_retain_literal_indexing_semantics(self):
        policy = future_fixture()
        source = policy.action.float()
        policy.g_rate_scaled_action_dims = (0, 1)
        self.compare(policy, source)
        policy.g_rate_scaled_action_dims = (False, True)
        with self.assertRaises(IndexError) as expected:
            literal_constants(policy, source)
        with self.assertRaises(IndexError) as actual:
            temporal_constants(policy, source, configured(policy))
        self.assertEqual(str(actual.exception), str(expected.exception))

        # A full-length Boolean mask is valid but selects different channels
        # from the numerically equal integer index list.
        policy.g_rate_scaled_action_dims = (0, 1, 0, 1)
        integer = self.compare(policy, source)
        policy.g_rate_scaled_action_dims = (False, True, False, True)
        boolean = self.compare(policy, source)
        self.assertIsNot(integer[3], boolean[3])
        self.assertFalse(torch.equal(integer[3], boolean[3]))

        # This PyTorch version accepts floating lists as indices. Retain those
        # literal values while still invalidating the changed configuration.
        policy.g_rate_scaled_action_dims = (0, 1)
        integer = self.compare(policy, source)
        policy.g_rate_scaled_action_dims = (0., 1.)
        floating = self.compare(policy, source)
        self.assertIsNot(integer[3], floating[3])

    def test_inference_inputs_and_outputs_are_not_cached(self):
        policy = future_fixture()
        source = policy.action.float()
        with torch.inference_mode():
            policy.action_mean = policy.action_mean.clone()
        first = self.compare(policy, source)
        second = self.compare(policy, source)
        self.assertIsNot(first[0], second[0])
        self.assertFalse(hasattr(policy, '_temporal_static_cache'))
        policy = future_fixture()
        source = policy.action.float()
        with torch.inference_mode():
            first = self.compare(policy, source)
            second = self.compare(policy, source)
        self.assertIsNot(first[0], second[0])
        self.assertFalse(hasattr(policy, '_temporal_static_cache'))
        with torch.inference_mode():
            source = policy.state.clone()
        self.compare(policy, source, actions=False)
        self.assertFalse(hasattr(policy, '_temporal_static_cache'))

    def test_mutated_cached_outputs_rebuild_and_source_banks_are_not_retained(self):
        policy = future_fixture()
        source = policy.action.float()
        previous = self.compare(policy, source)
        previous[1].add_(10)
        rebuilt = self.compare(policy, source)
        self.assertIsNot(previous[0], rebuilt[0])
        reference = weakref.ref(source)
        del source
        gc.collect()
        self.assertIsNone(reference())
        for width in range(4, 20):
            temporal_constants(policy, torch.zeros(1, width), configured(policy))
        self.assertLessEqual(len(policy._temporal_static_cache), 8)

    def test_state_constants_keep_exact_empty_shapes_and_source_dtype(self):
        policy = future_fixture()
        first = self.compare(policy, policy.state, actions=False)
        self.assertEqual([value.shape for value in first[1:]], [torch.Size([0])]*3)
        second = self.compare(policy, policy.state, actions=False)
        for old, new in zip(first, second):
            self.assertIs(old, new)

    def test_stops_track_starts_identity_version_and_sample_count(self):
        policy = future_fixture()
        first = trajectory_stops(policy)
        self.assertIs(first, trajectory_stops(policy))
        for change in (lambda: policy.trajectory_starts.add_(1),
                       lambda: setattr(policy, 'samples', policy.samples+3),
                       lambda: setattr(policy, 'trajectory_starts', policy.trajectory_starts.clone())):
            change()
            current = trajectory_stops(policy)
            self.assertIsNot(current, first)
            torch.testing.assert_close(current, policy._trajectory_stops(), rtol=0, atol=0)
            first = current
        first.add_(1)
        self.assertIsNot(first, trajectory_stops(policy))
        with torch.inference_mode():
            policy.trajectory_starts = policy.trajectory_starts.clone()
        self.assertIsNot(trajectory_stops(policy), trajectory_stops(policy))

    def test_private_stops_override_is_called_every_time(self):
        policy = future_fixture()
        with patch.object(policy, '_trajectory_stops', side_effect=[torch.tensor([9]), torch.tensor([11])]) as method:
            self.assertEqual(trajectory_stops(policy).tolist(), [9])
            self.assertEqual(trajectory_stops(policy).tolist(), [11])
        self.assertEqual(method.call_count, 2)
        self.assertFalse(hasattr(policy, '_temporal_static_cache'))


if __name__ == '__main__':
    unittest.main()
