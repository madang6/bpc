"""Runtime precision switches preserve fitted artifacts and reset shared variants."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from bpc.bpc import apply_variant_runtime
from bpc.io import read_config
from bpc.pipeline import _validate_methods, fit_groups


class RetrievalPrecisionTests(unittest.TestCase):
    def test_runtime_choice_does_not_change_fit_identity(self):
        config = read_config(Path(__file__).resolve().parents[1]/
            'configs/robosuite_fisher_sparsemax_local.json')
        config['methods']['bpc_fisher']['retrieval_precision'] = 'fp64'
        original = fit_groups(config)
        config['methods']['bpc_fisher']['retrieval_precision'] = 'mixed'
        _validate_methods(config, None)
        self.assertEqual(fit_groups(config), original)
        invalid = deepcopy(config)
        invalid['methods']['bpc_fisher']['retrieval_precision'] = 'float16'
        with self.assertRaisesRegex(ValueError, 'retrieval_precision'):
            _validate_methods(invalid, None)
        invalid['methods']['bpc_fisher']['retrieval_precision'] = 'mixed'
        invalid['methods']['bpc_fisher']['g_lookup'] = 'action_ridge'
        with self.assertRaisesRegex(ValueError, 'Fisher adaptive coverage'):
            _validate_methods(invalid, None)

    def test_switching_shared_policy_resets_search_and_default_mode(self):
        p = SimpleNamespace(config={'g': {}}, device=torch.device('cpu'),
            g_lookup='lda',
            support_selector=SimpleNamespace(config={'adaptive_support': 'coverage', 'fisher_rank': 70}))
        apply_variant_runtime(p, {'retrieval_precision': 'mixed'})
        self.assertEqual(p.support_selector._density_search_mode, 'mixed')
        p.support_selector._density_search = object()
        apply_variant_runtime(p, {})
        self.assertEqual(p.support_selector._density_search_mode, 'reference')
        self.assertIsNone(p.support_selector._density_search)
        p.g_lookup = 'action_ridge'
        with self.assertRaisesRegex(ValueError, 'Fisher adaptive coverage'):
            apply_variant_runtime(p, {'retrieval_precision': 'mixed'})


if __name__ == '__main__':
    unittest.main()
