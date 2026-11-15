"""Byte-exact pointwise RFF fusion with unchanged GEMMs and pooling."""
import math
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch

from dpc_nldl.fused_rff_phase import phase_cosine_
from dpc_nldl.ragged_supports import pooled_mean


def logical_bytes(value):
    return value.detach().reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)


def query_features(model, queries, neighbors, neighbor_actions=None, mask=None, *, fused=False):
    """Literal 5753133 feature path; only its pointwise epilogue is optional."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    output = []
    try:
        neighbors = neighbors if mask is None else neighbors.clamp_min(0)
        bank = torch.cat([model.state[neighbors], model.action[neighbors]
            if neighbor_actions is None else neighbor_actions], 2).to(model.rff_dtype)
        queries = queries.to(model.rff_dtype)
        projected = torch.cat([row[None, None] @ model.rff_query_weight for row in queries])
        tile = model.rff_dim if len(queries) == 1 else min(512, model.rff_dim)
        for start in range(0, model.rff_dim, tile):
            stop = min(start+tile, model.rff_dim)
            phase = bank @ model.rff_bank_weight[:, start:stop]
            bias, query = model.rff_bias[start:stop], projected[:, :, start:stop]
            result = phase_cosine_(phase, bias, query) if fused else None
            if result is None:
                phase.add_(bias).sub_(query).cos_()
            output.append(pooled_mean(phase, mask).mul_(model.rff_scale))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return torch.cat(output, dim=1)


def assert_negative_view_fallback(test, device):
    values = [torch.randn(*shape, device=device) for shape in ((2, 3, 9), (9,), (2, 1, 9))]
    before = [logical_bytes(value) for value in values]
    with patch('dpc_nldl.fused_rff_phase.extension', side_effect=AssertionError('CUDA loaded')):
        for index in range(3):
            with test.subTest(argument=index):
                arguments = values.copy()
                arguments[index] = torch._neg_view(arguments[index])
                test.assertTrue(arguments[index].is_neg())
                test.assertIsNone(phase_cosine_(*arguments))
    for original, current in zip(before, values):
        test.assertTrue(torch.equal(original, logical_bytes(current)))


class FusedRFFPhaseCPUTests(unittest.TestCase):
    def test_lazy_negative_views_fall_back_without_mutation(self):
        assert_negative_view_fallback(self, 'cpu')

    def test_cpu_fallback_is_nonmutating_and_does_not_load_cuda(self):
        phase = torch.randn(3, 7, 19)
        before = phase.clone()
        with patch('dpc_nldl.fused_rff_phase.extension', side_effect=AssertionError('CUDA loaded')):
            self.assertIsNone(phase_cosine_(phase, torch.ones(19), torch.zeros(3, 1, 19)))
        self.assertTrue(torch.equal(logical_bytes(before), logical_bytes(phase)))


@cuda
class FusedRFFPhaseCUDATests(unittest.TestCase):
    def test_lazy_negative_views_fall_back_without_mutation(self):
        assert_negative_view_fallback(self, 'cuda')

    def compare(self, phase, bias, projected):
        expected = phase.clone().add_(bias).sub_(projected).cos_()
        actual = phase.clone()
        address = actual.data_ptr()
        returned = phase_cosine_(actual, bias, projected)
        self.assertIs(returned, actual, 'CUDA helper unavailable or unexpected fallback')
        self.assertEqual(actual.data_ptr(), address)
        self.assertTrue(torch.equal(logical_bytes(expected), logical_bytes(actual)))

    def test_random_phases_bias_and_strided_projection_are_byte_equal(self):
        gen = torch.Generator(device='cuda').manual_seed(3189)
        for dtype in (torch.float32, torch.float64):
            for batch, count, width in ((1, 45, 16384), (2, 64, 512), (4, 100, 512), (3, 7, 19)):
                phase = torch.randn(batch, count, width, dtype=dtype, device='cuda', generator=gen)*83
                bias = torch.randn(2*width, dtype=dtype, device='cuda', generator=gen)[::2]
                projected = torch.randn(batch, 1, width+7, dtype=dtype, device='cuda', generator=gen)[..., :width]
                with self.subTest(dtype=dtype, batch=batch, count=count, width=width):
                    self.compare(phase, bias, projected)

    def test_cancellation_subnormals_nonfinite_and_signed_zero(self):
        for dtype in (torch.float32, torch.float64):
            info = torch.finfo(dtype)
            value = torch.tensor([0., -0., info.tiny, -info.tiny, info.tiny/2,
                info.max, -info.max, math.inf, -math.inf, math.nan, 1e10, -1e10],
                dtype=dtype, device='cuda')
            phase = value[None, None].expand(2, 3, -1).clone()
            bias = -value.clone()
            self.compare(phase, bias, value[None, None].expand(2, 1, -1))
            self.compare(phase, torch.zeros_like(value), torch.zeros(2, 1, len(value), device='cuda', dtype=dtype))

    def test_empty_and_unsupported_paths_do_not_mutate_inputs(self):
        phase = torch.randn(2, 3, 9, device='cuda')
        bias, query = torch.randn(9, device='cuda'), torch.randn(2, 1, 9, device='cuda')
        expected = phase.clone()
        for bad in (phase.transpose(1, 2), phase.double(), phase.requires_grad_()):
            self.assertIsNone(phase_cosine_(bad, bias, query))
        phase.requires_grad_(False)
        self.assertIsNone(phase_cosine_(phase, bias.cpu(), query))
        self.assertIsNone(phase_cosine_(phase, phase[0, 0], query))
        self.assertIsNone(phase_cosine_(phase, bias, phase[:, :1]))
        with patch('dpc_nldl.fused_rff_phase.extension', return_value=None):
            self.assertIsNone(phase_cosine_(phase, bias, query))
        self.assertTrue(torch.equal(logical_bytes(expected), logical_bytes(phase)))
        empty = phase[:0]
        self.assertIs(phase_cosine_(empty, bias, query[:0]), empty)

    def test_full_features_retimed_actions_masks_and_readout_match(self):
        from types import SimpleNamespace
        gen = torch.Generator(device='cuda').manual_seed(2881)
        def random(*shape, dtype=torch.float64):
            return torch.randn(*shape, device='cuda', dtype=dtype, generator=gen)
        for dtype in (torch.float32, torch.float64):
            model = SimpleNamespace(state=random(127, 33), action=random(127, 7),
                rff_dtype=dtype, rff_dim=1027, rff_scale=math.sqrt(2/1027),
                rff_bank_weight=random(40, 1027, dtype=dtype),
                rff_query_weight=random(33, 1027, dtype=dtype),
                rff_bias=random(1027, dtype=dtype))
            readout = random(1027, 7, dtype=dtype)
            for batch in (1, 2, 4):
                neighbors = torch.randint(127, (batch, 67), device='cuda', generator=gen)
                queries = random(batch, 33)
                for retimed in (False, True):
                    actions = random(batch, 67, 7) if retimed else None
                    for padded in (False, True):
                        mask = torch.arange(67, device='cuda')[None] < (63-torch.arange(batch, device='cuda'))[:, None] if padded else None
                        expected = query_features(model, queries, neighbors, actions, mask)
                        actual = query_features(model, queries, neighbors, actions, mask, fused=True)
                        self.assertTrue(torch.equal(logical_bytes(expected), logical_bytes(actual)))
                        self.assertTrue(torch.equal(logical_bytes(expected@readout), logical_bytes(actual@readout)))
                        self.assertTrue(torch.equal(logical_bytes(expected.norm(dim=1)), logical_bytes(actual.norm(dim=1))))


if __name__ == '__main__':
    unittest.main()
