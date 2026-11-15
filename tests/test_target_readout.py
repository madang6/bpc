"""True-support affine mathematics for the batched offline target path."""
import unittest
from tests.suites import cuda
from unittest.mock import patch

import torch

from dpc_nldl.target_readout import offline_prior, solve_padded_affine
from tests.support.readout import affine_problems as fixture
from tests.support.readout import offline_fixture as temporal_fixture


class TargetReadoutTest(unittest.TestCase):
    def solve_case(self, device):
        policy, _ = fixture(device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(512)
        lengths = [1, 2, 3, 7, 12, 16, 22]
        design = torch.randn(len(lengths), 40, 32, device=device,
                             generator=generator, dtype=torch.float32)
        query = torch.randn(len(lengths), 40, device=device,
                            generator=generator, dtype=torch.float32)
        # Three different rank-deficient cases must use the original solver.
        design[3, :, 6] = design[3, :, 0]
        design[4, :, :12] = design[4, :, :1].expand(-1, 12)
        design[5, :, :16] = 0
        expected = [policy._solve_affine(design[i:i+1, :, :count].cpu(),
                    query[i:i+1].cpu())[0].to(device) for i, count in enumerate(lengths)]
        for row, count in enumerate(lengths):
            design[row, :, count:] = float('nan')
        stats = {}
        actual = solve_padded_affine(policy, design, query,
            torch.tensor(lengths, device=device), diagnostics=stats)
        self.assertGreaterEqual(stats['fallback_rows'], 3)
        for row, count in enumerate(lengths):
            torch.testing.assert_close(actual[row, :count], expected[row], atol=2e-6, rtol=2e-5)
            matrix = design[row, :, :count].double().cpu()
            centered = matrix - matrix.mean(1, keepdim=True)
            if row in (4, 5):
                centered.zero_()  # Exactly identical columns; ignore mean-rounding noise.
            minimum_norm = 1/count + torch.linalg.pinv(centered, rtol=1e-7) @ (
                query[row].double().cpu() - matrix.mean(1))
            torch.testing.assert_close(actual[row, :count].double().cpu(), minimum_norm,
                                       atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(actual[row, count:], torch.zeros_like(actual[row, count:]),
                                       rtol=0, atol=0)
        torch.testing.assert_close(actual.sum(1), query.new_ones(len(lengths)), atol=2e-6, rtol=0)

    def test_variable_counts_match_unpadded_solve_with_singular_fallback(self):
        self.solve_case('cpu')

    @cuda
    def test_cuda_variable_counts_match_unpadded_solve_with_singular_fallback(self):
        self.solve_case('cuda')

    def test_affine_optimum_is_independent_of_padding_width(self):
        policy, _ = fixture(dtype=torch.float32)
        torch.manual_seed(27)
        lengths = [2, 5, 9]
        original = torch.randn(3, 24, 9)
        query = torch.randn(3, 24)
        outputs = []
        for width in (9, 16, 64):
            design = torch.nn.functional.pad(original, (0, width-9), value=10000)
            actual = solve_padded_affine(policy, design, query, torch.tensor(lengths))
            outputs.append(actual[:, :9])
            for row, count in enumerate(lengths):
                matrix = original[row, :, :count].double()
                centered = matrix - matrix.mean(1, keepdim=True)
                reference = 1/count + torch.linalg.pinv(centered, rtol=1e-10) @ (
                    query[row].double() - matrix.mean(1))
                torch.testing.assert_close(actual[row, :count].double(), reference,
                                           atol=3e-6, rtol=3e-5)
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], atol=2e-6, rtol=2e-5)

    def test_temporal_prior_uses_true_counts_and_one_shared_projection(self):
        # Exercise scheduling using the real temporal policy/gathers and a
        # selector exposing the same distance/rate interface as Fisher LDA.
        policy, rows, queries, supports = temporal_fixture()
        policy.g_lookup = 'action_ridge'
        selected = [(supports[i, :i+1], torch.zeros(i+1)) for i in range(3)]
        expected = torch.cat([policy._g_prior(None, columns[None], queries[i:i+1].float(),
            rows=rows[i:i+1], query_history=queries[i:i+1])[0]
            for i, (columns, _) in enumerate(selected)])
        policy.support_selector = type('DistanceSelector', (), {})()
        def distance(keys, target_rows, columns):
            planes = [policy.pf, *[plane for _, plane, *_ in policy._interpolated_keys()]]
            dist = [(keys[:, None].double() - plane[columns].double()).square().sum(-1)
                    for plane in planes]
            for index, (rate, _, _, first, valid) in enumerate(policy._interpolated_keys(), 1):
                excluded = ~valid[columns] | policy._rate_target_overlap(
                    first[columns], policy.hankel_starts[columns], rate, target_rows)
                dist[index].masked_fill_(excluded, torch.inf)
            stacked = torch.stack(dist)
            return stacked.amin(0), stacked.argmin(0)
        policy.support_selector._distance = distance
        with patch('dpc_nldl.target_readout._eligible', return_value=True), patch.object(
                policy, '_g_lookup_projection', wraps=policy._g_lookup_projection) as project:
            actual, retained = offline_prior(policy, rows, selected, queries,
                                              supports_validated=True)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        self.assertEqual(project.call_count, 1)
        self.assertEqual(retained.tolist(), [1, 2, 3])
        self.assertEqual(policy._target_readout_last['selected_columns'], 6)

    def test_unvalidated_supports_keep_original_path(self):
        policy, rows, queries, supports = temporal_fixture()
        selected = [(row, torch.zeros(len(row))) for row in supports]
        self.assertIsNone(offline_prior(policy, rows, selected, queries))

    def test_nonfinite_guard_factor_overrides_success_info(self):
        policy, _ = fixture(dtype=torch.float32)
        torch.manual_seed(51)
        design, query = torch.randn(2, 20, 8), torch.randn(2, 20)
        counts = torch.tensor([4, 7])
        original = torch.linalg.cholesky_ex
        calls = 0
        def faulty_guard(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 1:
                result.L[0, 0, 0] = float('nan')
                result.info[0] = 0
            return result
        with patch('torch.linalg.cholesky_ex', side_effect=faulty_guard), patch.object(
                policy, '_solve_g_normal', wraps=policy._solve_g_normal) as fallback:
            actual = solve_padded_affine(policy, design, query, counts)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(fallback.call_args.args[0].shape, (1, 4, 4))
        self.assertTrue(torch.isfinite(actual).all())
        for row, count in enumerate(counts.tolist()):
            expected = policy._solve_affine(design[row:row+1, :, :count], query[row:row+1])
            torch.testing.assert_close(actual[row:row+1, :count], expected, atol=2e-6, rtol=2e-5)

    def test_invalid_count_is_rejected(self):
        policy, _ = fixture(dtype=torch.float32)
        for counts in (torch.tensor([0]), torch.tensor([4])):
            with self.assertRaisesRegex(ValueError, 'count outside'):
                solve_padded_affine(policy, torch.ones(1, 8, 3), torch.ones(1, 8), counts)


if __name__ == '__main__':
    unittest.main()
