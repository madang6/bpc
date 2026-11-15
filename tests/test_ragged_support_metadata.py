"""Integer packing savings retain row/support order and original solve shapes."""
import unittest
from unittest.mock import patch

import torch

from dpc_nldl.ragged_supports import RaggedColumns, support_groups
from tests.support.readout import literal_outer_offline, ragged_policy as policy


class RaggedSupportMetadataTest(unittest.TestCase):
    def test_group_order_and_optional_distance_work(self):
        selected = [(torch.tensor([8, 2]), torch.tensor([.2, .8])),
                    (torch.tensor([6]), torch.tensor([.1])),
                    (torch.tensor([9, 1]), torch.tensor([.3, .7]))]
        before = list(support_groups(selected, 'cpu'))
        after = list(support_groups([(c, object()) for c, _ in selected], 'cpu', include_distances=False))
        self.assertEqual([x[0].tolist() for x in after], [[0, 2], [1]])
        for original, actual in zip(before, after):
            torch.testing.assert_close(original[0], actual[0], rtol=0, atol=0)
            torch.testing.assert_close(original[1], actual[1], rtol=0, atol=0)
            self.assertIsNone(actual[2])
        with self.assertRaisesRegex(RuntimeError, 'empty support'):
            list(support_groups([(torch.empty(0, dtype=torch.long), None)], 'cpu', include_distances=False))

    def test_known_lengths_avoid_tensor_scalar_conversion_and_preserve_empty_rows(self):
        rows = [torch.empty(0, 2), torch.tensor([[1., 2.], [3., 4.]]), torch.empty(0, 2)]
        with patch.object(torch.Tensor, '__int__', side_effect=AssertionError('unexpected scalar synchronization')):
            packed = RaggedColumns.from_rows(rows)
        restored = RaggedColumns(**packed.payload())
        self.assertEqual(packed.shape, (3, 2, 2))
        self.assertEqual(packed.shape, restored.shape)
        torch.testing.assert_close(packed.offsets, torch.tensor([0, 0, 2, 2]), rtol=0, atol=0)
        torch.testing.assert_close(packed.values, rows[1], rtol=0, atol=0)
        all_empty = RaggedColumns.from_rows([torch.empty(0, dtype=torch.long)]*2)
        self.assertEqual(all_empty.shape, (2, 0))
        self.assertEqual(all_empty.offsets.tolist(), [0, 0, 0])

    def test_offline_values_payload_and_solver_shapes_match_literal_dispatch(self):
        model = policy()
        rows = torch.tensor([3, 5, 11, 13, 19, 21])
        for leave_demo_out in (False, True):
            with self.subTest(leave_demo_out=leave_demo_out):
                shapes = []
                solve = model._solve_affine
                def traced_solve(design, query):
                    shapes.append((tuple(design.shape), tuple(query.shape)))
                    return solve(design, query)
                with patch.object(model, '_solve_affine', side_effect=traced_solve):
                    expected = literal_outer_offline(model, rows, leave_demo_out)
                    reference_shapes = shapes.copy()
                    shapes.clear()
                    with patch.object(model, '_support_endpoints', wraps=model._support_endpoints) as endpoints:
                        actual = model._offline_g(rows, leave_demo_out=leave_demo_out)
                    self.assertEqual(endpoints.call_count, 1)
                    self.assertEqual(shapes, reference_shapes)
                for left, right in zip(expected[:3], actual[:3]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertEqual(actual[3].shape, expected[3].shape)
                for key in ('values', 'offsets'):
                    torch.testing.assert_close(actual[3].payload()[key], expected[3].payload()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
