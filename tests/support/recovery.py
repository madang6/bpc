"""Small policy fixtures for recovery certificates and aperture calibration."""
import torch

from dpc_nldl import bpc


class RecoveryPolicy(bpc.Policy):
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.action_dim = 7
        self.proprio_dim = 9
        self.history = 10
        self.horizon = 10
        self.g_supports = 4
        self.state_scale = torch.ones(1033, dtype=torch.float64)
        self.state_mean = torch.zeros(1033, dtype=torch.float64)
        self.action_scale = torch.ones(7, dtype=torch.float64)
        self.action_mean = torch.zeros(7, dtype=torch.float64)
        self.recovery = True
        self.recovery_aperture = -0.0126
        self.recovery_settle = 3
        self.recovery_hold = 5
        self.recovery_tabu = 0.05
        self.recovery_attempts = 3
        self._open_future = None


def calibration_policy(grasp, floor, demos=30, length=60):
    policy = RecoveryPolicy()
    n = demos * length
    policy.action = torch.zeros(n, 7, dtype=torch.float64)
    policy.state = torch.zeros(n, 1033, dtype=torch.float64)
    policy.trajectory_of = torch.arange(n) // length
    for d in range(demos):
        base = d * length
        policy.action[base + 10:base + 30, 6] = 1.0        # one sustained closure
        policy.state[base:base + n // demos, 7] = grasp    # fingers hold the object
        policy.state[base + 45:base + 50, 7] = floor       # brief full close, after release
    return policy
