import unittest
from pathlib import Path

from dpc_nldl.io import read_config

import torch

from dpc_nldl.bpc import Policy
from dpc_nldl.artifacts import _target_key
from tests.support.policies import action_ridge_config, data_config, exact_policy, trajectories


class AffineExactTest(unittest.TestCase):
    def test_exact_is_not_the_finite_iteration_control(self):
        exact, pgd = exact_policy(), exact_policy()
        pgd.g_solver, pgd.g_iterations = "pgd", 1
        generator = torch.Generator().manual_seed(9)
        design = torch.randn(2, 12, 7, generator=generator, dtype=torch.float64)
        query = torch.randn(2, 12, generator=generator, dtype=torch.float64)
        gram = design.transpose(1, 2) @ design
        rhs = (design.transpose(1, 2) @ query[:, :, None]).squeeze(2)
        self.assertGreater(float((exact._solve_g_normal(gram, rhs)
                                  - pgd._solve_g_normal(gram, rhs)).abs().max()), 1e-3)

    def test_validation_config_and_cache_identity(self):
        config = read_config(Path(__file__).resolve().parents[1] / "configs/robosuite_fisher_sparsemax_local.json")
        method = config["method"]
        pgd = {**method, "g": {**method["g"], "solver": "pgd"}}
        self.assertNotEqual(_target_key(config, method), _target_key(config, pgd))
        for update in ({"constraint": "nonnegative"}, {"l1": .01}, {"l2": .01},
                       {"solver": "unknown"}, {"svd_rcond": -.1}, {"svd_rcond": .03},
                       {"future_consistency": 1.}):
            tiny = action_ridge_config()
            tiny["g"].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                Policy(trajectories(), data_config(), tiny, fit_residuals=False)

if __name__ == "__main__":
    unittest.main()
