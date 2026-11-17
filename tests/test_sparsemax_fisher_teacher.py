import unittest
from tests.suites import cuda, integration
from unittest.mock import patch
import torch

from bpc.lda import fit_discriminant
from bpc.bpc import Policy
from bpc.sparse_qp import sparsemax
from tests.support.policies import data_config, fisher_recipe, trajectories


class SparseTeacherTests(unittest.TestCase):
    def test_host_validation_is_bounded_and_checks_final_block(self):
        generator=torch.Generator().manual_seed(43)
        x=torch.randn(9,12,dtype=torch.float64,generator=generator)
        labels=torch.rand(9,8195,dtype=torch.float64,generator=generator)
        original=torch.isfinite; sizes=[]
        def finite(value):
            sizes.append(value.numel())
            return original(value)
        with patch('torch.isfinite',side_effect=finite):
            fit_discriminant(x,labels,fisher_rank=3)
        self.assertLessEqual(max(sizes),9*4096)
        for invalid in (float('nan'),float('inf'),-1.):
            labels[-1,-1]=invalid
            with self.assertRaisesRegex(ValueError,'finite features'):
                fit_discriminant(x,labels,fisher_rank=3)

    @cuda
    def test_host_teacher_preserves_gpu_fisher_metric(self):
        generator = torch.Generator().manual_seed(59)
        x = torch.randn(23, 40, dtype=torch.float64, generator=generator).cuda()
        # Shared classes retain within-class variance; unique one-hot labels
        # would make Fisher's within-class covariance singular by construction.
        logits = torch.full((23, 8201), -torch.inf, dtype=torch.float64, device='cuda')
        columns = [0, 1, 2, 3, 4, 4096, 4097, 4098, 4099, 8197, 8198, 8199, 8200]
        logits[:, columns] = .1 * torch.randn(23, len(columns), dtype=torch.float64, generator=generator).cuda()
        labels = sparsemax(logits)
        device = fit_discriminant(x, labels, fisher_rank=7)
        host = fit_discriminant(x, labels.cpu(), fisher_rank=7)
        torch.testing.assert_close(device['transform'] @ device['transform'].T,
                                   host['transform'] @ host['transform'].T, atol=1e-10, rtol=1e-10)
        self.assertEqual(device['represented_teacher_classes'], host['represented_teacher_classes'])

    def test_zero_mass_fisher_classes_have_zero_contribution(self):
        generator = torch.Generator().manual_seed(43)
        x = torch.randn(9, 12, dtype=torch.float64, generator=generator)
        positive = torch.rand(9, 4, dtype=torch.float64, generator=generator)
        labels = torch.zeros(9, 8195, dtype=x.dtype)
        labels[:, 4096:4100] = positive
        full = fit_discriminant(x, labels, fisher_rank=3)
        reduced = fit_discriminant(x, positive, fisher_rank=3)
        torch.testing.assert_close(full['transform'] @ full['transform'].T,
                                   reduced['transform'] @ reduced['transform'].T, atol=1e-10, rtol=1e-10)
        self.assertEqual(full['represented_teacher_classes'], 4)
        self.assertEqual(full['total_teacher_classes'], 8195)

    @integration
    def test_sparsemax_fit_and_coverage_use_sparse_teacher_and_full_bank(self):
        method = fisher_recipe()
        method['g']['lda']['teacher_storage'] = 'host'
        rows = trajectories(demos=4, length=32)
        policy = Policy(rows, data_config(),
                        method, fit_residuals=False)
        count = len(policy.hankel_starts)
        with patch('bpc.lda_selector.sparsemax', wraps=sparsemax) as calls:
            policy.fit_g_lookup()
            self.assertGreater(calls.call_count, 0)
        self.assertEqual(policy.g_lookup_stats['teacher_kind'], 'sparsemax')
        self.assertLess(policy.g_lookup_stats['teacher_positive_support']['mean'], 256)
        self.assertEqual(len(policy.hankel_starts), count)
        policy._project_hankel()
        selector = policy.support_selector
        selector.config['adaptive_support'] = 'coverage'
        anchors = torch.arange(20)
        rows = policy.hankel_starts[anchors] + policy.history
        with patch('bpc.lda_selector.sparsemax', wraps=sparsemax) as calls:
            result = selector._calibrate_coverage(anchors, rows)
            self.assertGreater(calls.call_count, 0)
        self.assertGreater(result['support_scale'], 0)

    def test_sparsemax_teacher_keeps_exclusions_and_zero_mass(self):
        logits = torch.tensor([[0., -.1, -2., -torch.inf],
                               [0., 0., 0., -torch.inf]], dtype=torch.float64)
        actual = sparsemax(logits)
        expected = torch.tensor([[.55, .45, 0., 0.], [1/3, 1/3, 1/3, 0.]], dtype=logits.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
