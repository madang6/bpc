"""Exact replay, graph ownership, mutation and unchanged grouped fallbacks."""
import gc
import unittest
from tests.suites import cuda, integration
import weakref
from unittest.mock import patch

import torch

from dpc_nldl import concurrent_readout as runtime
from tests.support.readout import ragged_readout_fixture as fixture
from tests.support.readout import equal


def original(p, u, y, recovery=None):
    with patch.object(runtime, 'direct_actions', return_value=None):
        return p.direct_actions(u, y, recovery=recovery)


def snapshot(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    return tuple(snapshot(item) for item in value)


def inputs():
    p, u, y, selected = fixture('cuda')
    # Well-conditioned synthetic histories exercise the graph success path,
    # independently of the intentional singular-support test below.
    gen = torch.Generator(device='cuda').manual_seed(887)
    p.state.add_(.2*torch.randn(p.state.shape, device='cuda', dtype=p.dtype, generator=gen))
    p.action.add_(.2*torch.randn(p.action.shape, device='cuda', dtype=p.dtype, generator=gen))
    selected = [(torch.arange(2, 2+k, device='cuda'),
                 torch.arange(k, device='cuda', dtype=p.dtype)) for k in (1, 2, 3, 4, 5, 6)]
    return p, u, y, selected


class ConcurrentReadoutCPUTests(unittest.TestCase):
    @integration
    def test_cpu_has_no_graph_cache_and_default_outputs_match(self):
        p, u, y, selected = fixture()
        with patch.object(p.support_selector, 'select', return_value=selected):
            self.assertTrue(equal(p.direct_actions(u, y), original(p, u, y)))
        self.assertFalse(hasattr(p, '_concurrent_readout_cache'))

    def test_deferred_affine_guard_requires_fixed_untruncated_path(self):
        p, _, _, _ = fixture()
        p.fixed_shape_readout = False
        with self.assertRaisesRegex(ValueError, 'deferred affine guards'):
            p._solve_affine(torch.randn(1, 8, 3), torch.randn(1, 8), _deferred_errors=[])

    def test_captured_state_preserves_lookup_without_retaining_policy_metadata(self):
        p, u, y, selected = fixture()
        p.support_selector.prepare()
        # Application metadata can refer back to its policy. It is not capture state.
        p.application_owner = p
        p.support_selector.application_owner = p
        p.g_lookup_rates = list(p.g_lookup_rates)
        core = runtime._Core(p)
        query = torch.cat([u[:1].flatten(1), y[:1].flatten(1)], 1)
        columns = selected[0][0][None]
        self.assertTrue(equal(core._g_lookup_projection(query), p._g_lookup_projection(query)))
        self.assertTrue(equal(core._rate_trace(query, columns), p._rate_trace(query, columns)))
        rates = list(core.g_lookup_rates)
        p.g_lookup_rates.append(3.)
        self.assertEqual(core.g_lookup_rates, rates)
        reference = weakref.ref(p)
        del p
        gc.collect()
        self.assertIsNone(reference(), 'captured state must not retain unrelated policy metadata')
        self.assertTrue(torch.isfinite(core._g_lookup_projection(query)).all())


@cuda
class ConcurrentReadoutCUDATests(unittest.TestCase):
    def test_inherited_subclass_uses_graphs_and_custom_math_clears_them(self):
        p, u, y, selected = inputs()
        class AnnotatedPolicy(type(p)):
            label = 'application metadata'

        class CustomPolicy(AnnotatedPolicy):
            def _query_features(self, *args, **kwargs):
                return super()._query_features(*args, **kwargs) * .5

        p.__class__ = AnnotatedPolicy
        with patch.object(p.support_selector, 'select', return_value=selected):
            self.assertTrue(equal(p.direct_actions(u, y), original(p, u, y)))
            p.direct_actions(u, y)
            self.assertGreater(p._concurrent_readout_cache.last['graphs'], 0)
            p.__class__ = CustomPolicy
            self.assertTrue(equal(p.direct_actions(u, y), original(p, u, y)))
            self.assertFalse(hasattr(p, '_concurrent_readout_cache'))

    def test_exact_changed_queries_owned_outputs_and_streams(self):
        p, u, y, selected = inputs()
        retained = []
        with patch.object(p.support_selector, 'select', return_value=selected):
            for index in range(3):
                actions, states = u+index*.001, y-index*.002
                actual = p.direct_actions(actions, states)
                self.assertTrue(equal(actual, original(p, actions, states)))
                retained.append((actual, snapshot(actual)))
            cache = p._concurrent_readout_cache
            self.assertGreater(cache.last['graphs'], 0)
            self.assertFalse(cache.last['fallback'])
            self.assertEqual(len({g.stream.cuda_stream for g, _ in cache.entries.values()}), 2)
            for result, saved in retained:
                self.assertTrue(equal(result, saved))
            # A caller stream change still protects previous output copies and
            # replays each graph on its original owner stream.
            caller = torch.cuda.Stream()
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                actual = p.direct_actions(u, y)
            torch.cuda.current_stream().wait_stream(caller)
            self.assertTrue(equal(actual, original(p, u, y)))
        runtime.clear_cache(p)
        self.assertTrue(all(equal(result, saved) for result, saved in retained))

    def test_mutation_invalidation_and_unversioned_fallback(self):
        p, u, y, selected = inputs()
        p.g_rate_scaled_action_dims = list(p.g_rate_scaled_action_dims)
        with patch.object(p.support_selector, 'select', return_value=selected):
            p.direct_actions(u, y); p.direct_actions(u, y)
            for name, mutate in (
                    ('coefficients', lambda: p.W_gip.add_(.01)),
                    ('rate scaling', lambda: p.g_rate_scaled_action_dims.clear()),
                    ('selector features', lambda: p.support_selector.phase.add_(.01)),
                    ('bank replacement', lambda: p.support_selector.banks.__setitem__(
                        0, p.support_selector.banks[0].clone()))):
                with self.subTest(mutation=name):
                    old = p._concurrent_readout_cache
                    self.assertTrue(old.entries)
                    mutate()
                    self.assertTrue(equal(p.direct_actions(u, y), original(p, u, y)))
                    self.assertIsNot(old, p._concurrent_readout_cache)
                    self.assertFalse(old.entries)
                    p.direct_actions(u, y)  # Populate the replacement before its next mutation.
            with torch.inference_mode():
                p.W_gip = p.W_gip.clone()
            self.assertTrue(equal(p.direct_actions(u, y), original(p, u, y)))
            self.assertFalse(hasattr(p, '_concurrent_readout_cache'))

    def test_singular_and_poisoned_graph_results_replay_without_reselection(self):
        p, u, y, selected = inputs()
        with patch.object(p.support_selector, 'select', return_value=selected) as select:
            p.direct_actions(u, y); p.direct_actions(u, y)
            cache = p._concurrent_readout_cache
            expected = original(p, u, y)
            join = runtime._Graph.join
            def poison(graph):
                join(graph)
                graph.flags.fill_(True)
                graph.output[0].fill_(float('nan'))
            select.reset_mock()
            with patch.object(runtime._Graph, 'join', poison):
                self.assertTrue(equal(p.direct_actions(u, y), expected))
            self.assertEqual(select.call_count, 1)
            self.assertTrue(cache.last['fallback'])
            columns, distances = selected[2]
            selected[2] = (columns[:1].expand_as(columns).clone(), distances)
            expected = original(p, u, y)
            select.reset_mock()
            self.assertTrue(equal(p.direct_actions(u, y), expected))
            self.assertEqual(select.call_count, 1)
            self.assertTrue(cache.last['fallback'])

    def test_bounds_flags_and_default_override_fallback(self):
        p, u, y, selected = inputs()
        p.support_selector.prepare()
        core = runtime._Core(p)
        columns = torch.tensor([[len(p.hankel_starts)]], device='cuda')
        _, flags = core(u[:1], y[:1], columns, u.new_zeros(1, 1))
        self.assertTrue(bool(flags.any()))
        with patch.object(p.support_selector, 'select', return_value=selected):
            p.direct_actions(u, y); p.direct_actions(u, y)
            with patch.object(p, '_query_features', wraps=p._query_features):
                self.assertTrue(equal(p.direct_actions(u, y), original(p, u, y)))
                self.assertFalse(hasattr(p, '_concurrent_readout_cache'))

    def test_bounded_eviction_capture_failure_and_policy_lifetime(self):
        p, u, y, selected = inputs()
        with patch.object(runtime, 'MAX_GRAPHS', 2), patch.object(runtime, 'CACHE_BYTES', 128*1024**2):
            for item in selected[:4]:
                with patch.object(p.support_selector, 'select', return_value=[item]):
                    p.direct_actions(u[:1], y[:1])
                    self.assertTrue(equal(p.direct_actions(u[:1], y[:1]), original(p, u[:1], y[:1])))
                cache = p._concurrent_readout_cache
                self.assertLessEqual(len(cache.entries), 2)
                self.assertLessEqual(cache.bytes, runtime.CACHE_BYTES)
            self.assertGreater(cache.evictions, 0)
        runtime.clear_cache(p)
        with (patch.object(p.support_selector, 'select', return_value=selected),
              patch.object(runtime, '_Graph', side_effect=RuntimeError('capture unsupported')) as capture):
            for _ in range(3):
                self.assertTrue(equal(p.direct_actions(u, y), original(p, u, y)))
            self.assertEqual(capture.call_count, len(selected))
        runtime.clear_cache(p)
        with patch.object(p.support_selector, 'select', return_value=selected):
            p.direct_actions(u, y); p.direct_actions(u, y)
        reference = weakref.ref(p)
        del p, cache, capture  # The mocked exception retains its traceback frames.
        gc.collect()
        self.assertIsNone(reference(), 'graph cache must not keep its policy alive')


if __name__ == '__main__':
    unittest.main()
