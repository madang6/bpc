"""Synthetic source banks for history and future interpolation tests."""
import torch

from bpc.bpc import Policy


def policy_fixture(device='cpu', source_dtype=torch.float64, offset=1, history=10,
                   scaled=(0, 2), rates=(.5, .75, 1., 1.1, 1.5, 2.)):
    policy = Policy.__new__(Policy)
    policy.device, policy.dtype = torch.device(device), source_dtype
    policy.history, policy.g_state_offset = history, offset
    policy.g_native, policy.g_align_history = False, True
    policy.g_lookup_rates, policy.g_rate_scaled_action_dims = rates, scaled
    generator = torch.Generator().manual_seed(931)
    policy.state = torch.randn(128, 17, generator=generator, dtype=source_dtype).to(device)
    policy.action = torch.randn(128, 4, generator=generator, dtype=source_dtype).to(device)
    policy.action_mean = torch.tensor([.11, -.31, .27, -.57], device=device, dtype=source_dtype)
    policy.action_scale = torch.tensor([.17, .73, 1.13, 2.07], device=device, dtype=source_dtype)
    policy.hankel_starts = torch.tensor([22, 28, 34, 78, 84, 90, 96], device=device)
    policy.hankel_trajectory = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device)
    policy.trajectory_starts = torch.tensor([0, 64], device=device)
    policy._fit_sources = None
    return policy


def future_fixture(device='cpu', dtype=torch.float64, native_dtype=None, offset=1):
    policy = policy_fixture(device, dtype, offset=offset)
    policy.horizon, policy.proprio_dim, policy.samples = 7, 9, len(policy.state)
    policy.action_dim = policy.action.shape[1]
    policy.g_align_future = True
    rows = policy.hankel_starts[:, None]+policy.history+torch.arange(policy.horizon, device=device)
    values = policy.action[rows].flatten(1).T.to(native_dtype or dtype)+.03125
    # Deliberately strided and different from a fresh action-source gather:
    # native outputs must preserve this cached uf exactly.
    native = torch.empty(len(values), 2*values.shape[1], device=device, dtype=values.dtype)
    native[:, ::2] = values
    policy.uf = native[:, ::2]
    return policy
