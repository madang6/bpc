"""One integer certificate preserves the original ragged solves and errors."""
import unittest
from unittest.mock import patch

import torch

from dpc_nldl.ragged_supports import RaggedColumns, offline_supports_valid
from tests.support.readout import ragged_policy as policy
from tests.support.readout import literal_outer_offline


class RaggedValidationTest(unittest.TestCase):
    def test_one_status_read_replaces_each_groups_checks(self):
        model = policy()
        model.g_weights = 'uniform'  # Isolate the exclusion checks from solver decisions.
        rows = torch.tensor([3, 5, 11, 13, 19, 21])
        original = torch.Tensor.__bool__
        for leave_demo_out in (False, True):
            counts = []
            results = []
            for operation in (lambda: literal_outer_offline(model, rows, leave_demo_out),
                              lambda: model._offline_g(rows, leave_demo_out=leave_demo_out)):
                calls = []
                def checked(value):
                    calls.append(value.numel())
                    return original(value)
                with patch.object(torch.Tensor, '__bool__', checked):
                    results.append(operation())
                counts.append(len(calls))
            self.assertEqual(counts, [6 if leave_demo_out else 3, 1])
            for expected, actual in zip(results[0][:3], results[1][:3]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for field in ('values', 'offsets'):
                torch.testing.assert_close(getattr(results[1][3], field),
                                           getattr(results[0][3], field), rtol=0, atol=0)

    def test_invalid_certificate_retains_original_group_error_order(self):
        model = policy()
        rows = torch.tensor([3, 11])
        starts = model.hankel_starts
        first = torch.nonzero(starts == 0).flatten()  # Same demo, but no action leakage.
        second = torch.cat([torch.nonzero(starts == 10).flatten(),
                            torch.nonzero(starts == 16).flatten()])
        selected = [(first, torch.zeros(len(first))), (second, torch.zeros(len(second)))]
        with patch.object(model.support_selector, 'select', return_value=selected):
            for operation in (lambda: literal_outer_offline(model, rows, True),
                              lambda: model._offline_g(rows, leave_demo_out=True)):
                with self.assertRaisesRegex(RuntimeError, 'target demonstration entered'):
                    operation()

    def test_invalid_indices_fall_back_without_reading_outside_storage(self):
        model = policy()
        rows = torch.tensor([3])
        for column in (len(model.hankel_starts), -len(model.hankel_starts)-1):
            values = torch.tensor([column])
            packed = RaggedColumns.from_rows([values])
            self.assertFalse(offline_supports_valid(model, rows, packed))
            with patch.object(model.support_selector, 'select', return_value=[(values, torch.zeros(1))]):
                with self.assertRaises(IndexError):
                    model._offline_g(rows)
        # Native PyTorch negative indexing remains valid when it satisfies exclusions.
        packed = RaggedColumns.from_rows([torch.tensor([-1])])
        self.assertTrue(offline_supports_valid(model, rows, packed, leave_demo_out=True))

    def test_empty_support_retains_original_error(self):
        model = policy()
        with patch.object(model.support_selector, 'select', return_value=[
                (torch.empty(0, dtype=torch.long), torch.empty(0))]):
            with self.assertRaisesRegex(RuntimeError, 'sparsemax selected an empty support'):
                model._offline_g(torch.tensor([3]))


if __name__ == '__main__':
    unittest.main()
