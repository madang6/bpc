"""Certificate-gated in-bank recovery: the certificate, the redirect, and the default-off path."""
import unittest

import torch

from dpc_nldl import bpc
from tests.support.recovery import RecoveryPolicy, calibration_policy


def _history(command, aperture):
    policy = RecoveryPolicy()
    action = torch.zeros(1, policy.history, 7, dtype=torch.float64)
    action[:, -1, 6] = command
    state = torch.zeros(1, policy.history, 1033, dtype=torch.float64)
    state[:, -1, 7] = aperture
    return policy, action, state


class RecoveryTest(unittest.TestCase):
    def test_certificate_fires_only_on_a_settled_empty_close(self):
        policy, action, state = _history(command=1.0, aperture=-0.02)
        status = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)   # one tick short
        updated, fired = policy._recovery_update(state, action, status)
        self.assertTrue(bool(fired[0]))
        self.assertEqual(float(updated[0, 1]), policy.recovery_hold)
        self.assertEqual(float(updated[0, 2]), 1.0)

    def test_a_real_grasp_does_not_fire(self):
        policy, action, state = _history(command=1.0, aperture=-0.009)
        status = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
        _, fired = policy._recovery_update(state, action, status)
        self.assertFalse(bool(fired[0]))

    def test_an_open_gripper_resets_the_counter(self):
        policy, action, state = _history(command=-1.0, aperture=-0.02)
        status = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
        updated, fired = policy._recovery_update(state, action, status)
        self.assertEqual(float(updated[0, 0]), 0.0)
        self.assertFalse(bool(fired[0]))

    def test_the_attempt_budget_is_respected(self):
        policy, action, state = _history(command=1.0, aperture=-0.02)
        status = torch.tensor([[2.0, 0.0, float(policy.recovery_attempts)]], dtype=torch.float64)
        _, fired = policy._recovery_update(state, action, status)
        self.assertFalse(bool(fired[0]), "must stop after the attempt budget")

    def test_redirect_retrieves_only_open_gripper_untried_columns(self):
        policy = RecoveryPolicy()
        columns = 40
        policy.hankel_starts = torch.arange(columns)
        policy.hankel_trajectory = torch.arange(columns) // 10        # 4 demonstrations
        policy.action = torch.zeros(columns + policy.history + 2, 7, dtype=torch.float64)
        # only columns 0-19 have an opening next action
        policy.action[torch.arange(0, 20) + policy.history, 6] = -1.0
        policy.action[torch.arange(20, columns) + policy.history, 6] = 1.0
        policy.pf = torch.randn(columns, 5, dtype=torch.float64)
        policy.pf_norm = policy.pf.square().sum(1)
        policy.g_lookup = "past"
        policy.g_lookup_matrix = torch.randn(3, 5, dtype=torch.float64)
        policy.g_lookup_scale = 1.0
        policy._interpolated_keys = lambda: []
        query = torch.randn(1, 5, dtype=torch.float64)
        tabu = torch.zeros(1, 4, dtype=torch.bool)
        chosen, _, exhausted = policy._recovery_topk(query, policy.g_supports,
            torch.ones(1,3), tabu, torch.zeros(1,dtype=torch.bool))
        self.assertFalse(bool(exhausted[0]))
        self.assertTrue(bool((chosen[0] < 20).all()), "must pick open-gripper columns only")
        tabu[0, 0] = True                                              # forbid demonstration 0
        chosen, _, _ = policy._recovery_topk(query, policy.g_supports,
            torch.ones(1,3), tabu, torch.zeros(1,dtype=torch.bool))
        self.assertTrue(bool((chosen[0] >= 10).all()) and bool((chosen[0] < 20).all()),
                        "must skip the tried demonstration and stay open-gripper")

    def test_exhaustion_is_reported_when_nothing_is_eligible(self):
        policy = RecoveryPolicy()
        columns = 12
        policy.hankel_starts = torch.arange(columns)
        policy.hankel_trajectory = torch.zeros(columns, dtype=torch.long)
        policy.action = torch.zeros(columns + policy.history + 2, 7, dtype=torch.float64)
        policy.action[torch.arange(columns) + policy.history, 6] = 1.0   # nothing opens
        policy.pf = torch.randn(columns, 5, dtype=torch.float64)
        policy.pf_norm = policy.pf.square().sum(1)
        policy.g_lookup = "past"
        policy.g_lookup_matrix = torch.randn(3, 5, dtype=torch.float64)
        policy.g_lookup_scale = 1.0
        policy._interpolated_keys = lambda: []
        _, _, exhausted = policy._recovery_topk(
            torch.randn(1, 5, dtype=torch.float64), policy.g_supports,
            torch.ones(1,3), torch.zeros(1,1,dtype=torch.bool), torch.zeros(1,dtype=torch.bool))
        self.assertTrue(bool(exhausted[0]))

    def test_recovery_is_off_by_default_everywhere(self):
        self.assertFalse(bpc.Policy.recovery)
        policy = RecoveryPolicy()
        policy.configure_recovery(None)
        self.assertFalse(policy.recovery)
        policy.configure_recovery({"enabled": False, "hold": 999})
        self.assertFalse(policy.recovery)


class ApertureCalibrationTest(unittest.TestCase):
    """The empty-close threshold is learned from the bank, not carried as a constant."""

    def test_threshold_sits_between_the_grasp_band_and_the_stop(self):
        policy = calibration_policy(grasp=-0.009, floor=-0.024)
        threshold = policy._calibrate_aperture()
        self.assertLess(threshold, -0.009, "must sit below a real grasp")
        self.assertGreater(threshold, -0.024, "must sit above the mechanical stop")

    def test_it_rejects_a_bank_with_too_few_closures(self):
        policy = calibration_policy(grasp=-0.009, floor=-0.024, demos=1)
        policy.action[:, 6] = 0.0
        with self.assertRaises(ValueError):
            policy._calibrate_aperture()

    def test_an_explicit_aperture_overrides_calibration(self):
        policy = calibration_policy(grasp=-0.009, floor=-0.024)
        policy.recovery = False
        policy.configure_recovery({"enabled": True, "aperture": -0.05, "settle": 3})
        self.assertEqual(policy.recovery_aperture, -0.05)


if __name__ == "__main__":
    unittest.main()
