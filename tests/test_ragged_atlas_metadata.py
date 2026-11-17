"""Cache integer padding metadata without changing any atlas floating operation."""
import math
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from bpc import fitting
import bpc.bpc as bpc
from bpc.ragged_supports import RaggedColumns, pooled_mean
from tests.support.policies import (
    data_config as _data_config, method_config as _method_config, trajectories as
    _trajectories,
)


def literal_atlas(model, rows, neighbors, neighbor_actions=None):
    """Original resident atlas arithmetic and original slice-by-slice padding."""
    tile = min(fitting.FIT_ROWS, model.rff_dim)
    row_batch = 2*fitting.FIT_ROWS
    itemsize = torch.empty((), dtype=model.rff_dtype).element_size()
    capacity = max(1, (128*1024**2)//(itemsize*neighbors.shape[1]*tile))
    row_batch = min(row_batch, 2**int(math.log2(capacity)))
    if model.rff_dtype == torch.float32:
        action, state = model._float_sources()
    else:
        state, action = model.state.to(model.rff_dtype), model.action.to(model.rff_dtype)
    output = torch.empty(len(rows), model.rff_dim, device=model.device, dtype=model.rff_dtype)
    state_weight = model.rff_bank_weight[:model.state_dim]
    action_weight = model.rff_bank_weight[model.state_dim:]
    for start in range(0, model.rff_dim, tile):
        stop = min(start+tile, model.rff_dim)
        bank_phase = state @ state_weight[:, start:stop]
        bank_phase.addmm_(action, action_weight[:, start:stop])
        for first in range(0, len(rows), row_batch):
            last = min(first+row_batch, len(rows))
            query_phase = state[rows[first:last]] @ model.rff_query_weight[:, start:stop]
            selected = neighbors[first:last]
            mask = selected >= 0
            selected = selected.clamp_min(0)
            phase = bank_phase[selected]
            if neighbor_actions is not None:
                delta = neighbor_actions[first:last].to(model.device, model.rff_dtype)-action[selected]
                phase.add_(delta @ action_weight[:, start:stop])
            phase.add_(model.rff_bias[start:stop]).sub_(query_phase[:, None]).cos_()
            output[first:last, start:stop] = pooled_mean(phase, mask).mul_(model.rff_scale)
    return output


class FloatingInputs(TorchDispatchMode):
    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func) in ('aten.mm.default', 'aten.bmm.default', 'aten.addmm_.default',
                         'aten.mean.dim', 'aten.sum.dim_IntList'):
            floating = [value for value in args if torch.is_tensor(value) and value.is_floating_point()]
            if floating:
                self.calls.append((str(func), [(value.clone(), value.stride()) for value in floating]))
        return func(*args, **(kwargs or {}))


def fixture(device='cpu', dtype=torch.float32):
    method = _method_config()
    method['device'] = device
    method['rff'].update(dtype=str(dtype).removeprefix('torch.'), features=7, storage='resident')
    method['residual']['lookup'] = 'g_supports'
    model = bpc.Policy(_trajectories(), _data_config(), method)
    rows = torch.tensor([2, 3, 5, 10, 11, 13, 18, 19, 21], device=device)
    neighbors = RaggedColumns.from_rows([torch.arange(5, 5+count, device=device)
                                        for count in (1, 2, 1, 2, 4, 3, 1, 2, 3)])
    fitting.make_basis(model, rows, neighbors.align(rows, model.samples), 0, .5)
    return model, rows, neighbors


class RaggedAtlasMetadataTests(unittest.TestCase):
    device = 'cpu'

    def compare(self, model, rows, neighbors, actions=None):
        records, outputs, synchronizations = [], [], []
        convert = torch.Tensor.__int__
        for operation in (lambda: literal_atlas(model, rows, neighbors, actions),
                          lambda: fitting.feature_atlas(model, rows, neighbors, neighbor_actions=actions)):
            calls = []
            def counted(value):
                calls.append(value.numel())
                return convert(value)
            with patch.object(fitting, 'FIT_ROWS', 2), patch.object(torch.Tensor, '__int__', counted), FloatingInputs() as record:
                outputs.append(operation())
            records.append(record.calls)
            synchronizations.append(len(calls))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        self.assertEqual(synchronizations, [12 if actions is None else 24, 0])
        self.assertEqual(len(records[0]), len(records[1]))
        for (old_name, old), (new_name, new) in zip(*records):
            self.assertEqual(old_name, new_name)
            for (old_value, old_stride), (new_value, new_stride) in zip(old, new):
                self.assertEqual(old_stride, new_stride)
                torch.testing.assert_close(old_value, new_value, rtol=0, atol=0)

    def test_native_and_retimed_action_features_keep_every_gemm_slot(self):
        for dtype in (torch.float32, torch.float64):
            model, rows, neighbors = fixture(self.device, dtype)
            self.compare(model, rows, neighbors)
            actions = RaggedColumns(neighbors.values[:, None].to(model.dtype)/7,
                                    neighbors.offsets, _max_count=neighbors.shape[1])
            # Retimed action metadata is kept separate from support metadata.
            self.compare(model, rows, neighbors, actions)

    def test_padding_is_rebuilt_after_offsets_change(self):
        model, rows, neighbors = fixture(self.device)
        self.compare(model, rows, neighbors)
        lengths = neighbors.counts.flip(0)
        neighbors.offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        self.compare(model, rows, neighbors)

    def test_padding_plan_preserves_empty_rows_and_arbitrary_values(self):
        rows = [torch.empty(0, dtype=torch.long, device=self.device),
                torch.tensor([9, 4], device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.tensor([3], device=self.device)]
        packed = RaggedColumns.from_rows(rows)
        for first, metadata in zip(range(0, len(packed), 2), packed._padding_plan(2)):
            torch.testing.assert_close(packed._padded(*metadata), packed[first:first+2], rtol=0, atol=0)
        other = packed.take(torch.tensor([3, 0, 1], device=self.device))
        for first, metadata in zip(range(0, len(other), 2), other._padding_plan(2)):
            torch.testing.assert_close(other._padded(*metadata), other[first:first+2], rtol=0, atol=0)

    def test_custom_ragged_indexing_keeps_original_dispatch(self):
        class ShiftedColumns(RaggedColumns):
            def __getitem__(self, rows):
                result = super().__getitem__(rows)
                return torch.where(result >= 0, result+1, result)
        model, rows, neighbors = fixture(self.device)
        neighbors = ShiftedColumns(neighbors.values, neighbors.offsets,
                                   _max_count=neighbors.shape[1])
        with patch.object(fitting, 'FIT_ROWS', 2), patch.object(
                neighbors, '_padding_plan', side_effect=AssertionError('custom indexing bypassed')):
            expected = literal_atlas(model, rows, neighbors)
            actual = fitting.feature_atlas(model, rows, neighbors)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@cuda
class RaggedAtlasMetadataCUDATests(RaggedAtlasMetadataTests):
    device = 'cuda'


if __name__ == '__main__':
    unittest.main()
