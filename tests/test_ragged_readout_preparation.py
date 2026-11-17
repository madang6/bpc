"""Original group math, deferred scratch safety and whole-loop error order."""
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch

from bpc.bpc import Policy
from bpc.ragged_readout import prepare_readout, _eligible
from tests.support.readout import equal, ragged_readout_fixture as fixture


def literal_outer(p, u, y):
    with patch('bpc.ragged_readout.prepare_readout', return_value=None):
        return p.direct_actions(u, y)


class ReadoutPreparationCPUTest(unittest.TestCase):
    def test_explicit_prepared_selected_call_matches_every_original_trace(self):
        p, u, y, selected = fixture()
        u, y = u[:1], y[:1]
        columns, distances = selected[1]
        columns, distances = columns[None], distances[None]
        query = torch.cat([u.flatten(1), y.flatten(1)], 1)
        trace = p._rate_trace(query, columns)
        rates = trace['hankel_rates']
        prepared = (query, trace, p._reconstruction_columns(columns, query, rates=rates),
                    p._future_columns(columns, query, rates=rates),
                    p._future_columns(columns, query, rates=rates, states=True))
        expected = p.direct_actions(u, y, selected=(columns, distances))
        actual = p.direct_actions(u, y, selected=(columns, distances), _prepared=prepared)
        self.assertTrue(equal(actual, expected))

    def test_cpu_and_custom_selector_keep_existing_path_and_selection_errors(self):
        p, u, y, selected = fixture()
        self.assertIsNone(prepare_readout(p, u, y, selected))
        with patch.object(p.support_selector, 'select', side_effect=ValueError('original selection error')):
            with self.assertRaisesRegex(ValueError, 'original selection error'):
                p.direct_actions(u, y)


@cuda
class ReadoutPreparationCUDATest(unittest.TestCase):
    def test_real_preparation_all_bytes_and_changed_queries(self):
        p, u, y, selected = fixture('cuda')
        with patch.object(p.support_selector, 'select', return_value=selected):
            self.assertTrue(_eligible(p, u, y))
            prepared = prepare_readout(p, u, y, selected)
            self.assertIsNotNone(prepared, 'production staging unexpectedly fell back')
            self.assertEqual([item[3].shape for item in prepared], [torch.Size([2, k]) for k in (1, 3, 2)])
            for changed in (False, True):
                actions, states = (u+.001, y-.002) if changed else (u, y)
                self.assertTrue(equal(p.direct_actions(actions, states),
                                      literal_outer(p, actions, states)))

    def test_bad_raw_scratch_is_discarded_before_original_solves(self):
        from bpc.fused_temporal_history import reconstruction_columns
        p, u, y, selected = fixture('cuda')
        calls = []
        original_solve = p._solve_affine
        def solve(design, query):
            self.assertTrue(bool(torch.isfinite(design).all()), 'invalid scratch consumed')
            calls.append(tuple(design.shape))
            return original_solve(design, query)
        solve.__func__ = Policy._solve_affine  # Transparent profiling wrapper.
        def bad(*args, **kwargs):
            value = reconstruction_columns(*args, **kwargs)
            if kwargs.get('_errors') is not None:
                kwargs['_errors'].append(torch.tensor(True, device='cuda'))
                return torch.full_like(value, float('nan'))
            return value
        with patch.object(p.support_selector, 'select', return_value=selected):
            expected = literal_outer(p, u, y)
            with (patch.object(p, '_solve_affine', solve),
                  patch('bpc.fused_temporal_history.reconstruction_columns', side_effect=bad)):
                self.assertTrue(equal(p.direct_actions(u, y), expected))
            self.assertEqual([shape[-1] for shape in calls], [1, 3, 2])

    def test_budget_single_group_overrides_and_malformed_bounds_bypass_preparation(self):
        p, u, y, selected = fixture('cuda')
        with patch('bpc.ragged_readout.PREPARED_BYTES', 1):
            self.assertIsNone(prepare_readout(p, u, y, selected))
        self.assertIsNone(prepare_readout(p, u[:1], y[:1], selected[:1]))
        for name in ('_rate_trace', '_future_columns', '_solve_affine', '_trajectory_stops', '_interpolate'):
            with patch.object(p, name, wraps=getattr(p, name)):
                self.assertFalse(_eligible(p, u, y))
                self.assertIsNone(prepare_readout(p, u, y, selected))
        broken = list(selected)
        broken[0] = (torch.tensor([len(p.hankel_starts)], device='cuda'), selected[0][1])
        self.assertIsNone(prepare_readout(p, u, y, broken))
        p.hankel_trajectory[2] = len(p.trajectory_starts)
        self.assertIsNone(prepare_readout(p, u, y, selected))

    def test_actual_temporal_error_retains_original_error_and_selector_called_once(self):
        p, u, y, selected = fixture('cuda')
        selected[0] = (torch.tensor([0], device='cuda'), selected[0][1])
        original = p._rate_trace
        def forced_rate(query, columns, target_rows=None):
            trace = original(query, columns, target_rows)
            trace['hankel_rates'].fill_(2.)
            return trace
        forced_rate.__func__ = Policy._rate_trace
        with (patch.object(p, '_rate_trace', forced_rate),
              patch.object(p.support_selector, 'select', return_value=selected) as select):
            self.assertIsNone(prepare_readout(p, u, y, selected))
            for call in (lambda: literal_outer(p, u, y), lambda: p.direct_actions(u, y)):
                select.reset_mock()
                with self.assertRaisesRegex(ValueError, 'interpolated history crosses'):
                    call()
                self.assertEqual(select.call_count, 1)


if __name__ == '__main__':
    unittest.main()
