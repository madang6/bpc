"""Small deterministic policy fixtures shared across independent test modules.

Factories return fresh configuration and policy state. ``small_ridge_fit`` is
only for low-rank artifact-plumbing fixtures; manuscript integration tests fit
unmocked stochastic data.
"""
import math
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import torch

from bpc.bpc import Policy
from bpc.retrieval import make_retrieval


def trajectories(demos=4, length=8):
    output = []
    for demo in range(demos):
        step = np.arange(length, dtype=np.float64)
        # Nonlinear, differently offset blocks avoid accidental distance ties.
        output.append({
            "x": (11 * demo + step + 0.07 * step**2)[None],
            "agent": (3 * demo + 0.13 * step**2)[None],
            "wrist": (-5 * demo + 0.19 * step**2 + 0.03 * step**3)[None],
            "u": (7 * demo - 0.4 * step + 0.11 * step**2)[None],
        })
    return output


def data_config():
    return {
        "state_key": "x",
        "action_key": "u",
        "feature_keys": ["agent", "wrist"],
        "r3m": {"feature_dim": 1},
    }


def method_config(lookback=2):
    return {
        "device": "cpu",
        "dtype": "float64",
        "state_dim": 3,
        "action_dim": 1,
        "history": 2,
        "scaling": {
            "enabled": False,
            "standard_deviation_correction": {"state": 0, "action": 1},
            "minimum_standard_deviation": 1e-6,
            "block_normalize": False,
        },
        "temporal_neighbors": {
            "lookback": lookback,
            "candidates": 4,
            "keep_ratio": 0.5,
            "decay": -0.25,
            "fit_batch_size": 32,
        },
        "rff": {
            "features": 4,
            "sigma_scale": 0.3,
            "ridge": 0.1,
            "seed": 0,
            "feature_pool_chunk_elements": 1000,
        },
        "g": {
            "supports": 2,
            "l1": 0.,
            "constraint": "affine",
            "solver": "exact",
            "iterations": 3,
            "future_consistency": 0.,
        },
        "residual": {"lookup": "g_supports", "target_batch_size": 2, "variance_floor": 1e-8},
        "continuation_horizon": 2,
    }


def action_ridge_config(select=True):
    method = method_config()
    if select:
        method["g"]["lookup"] = "action_ridge"
    return method


@contextmanager
def small_ridge_fit():
    """Allow low-dimensional fixtures used to test artifact plumbing, not GCV.

    The full-rank, unmocked fit is covered in test_manuscript_methods.
    """
    with (patch("torch.argmin", return_value=torch.tensor(20)),
          patch("torch.linalg.matrix_rank", return_value=torch.tensor(8))):
        yield


def selector_policy(lookback=2, fit_residuals=False):
    return Policy(
        trajectories(), data_config(), method_config(lookback),
        fit_residuals=fit_residuals,
    )


def toy_policy(history):
    """An unfitted, deterministic readout fixture: no fitting/cache dependency."""
    p = Policy.__new__(Policy)
    p.device, p.dtype = torch.device("cpu"), torch.float64
    p.history, p.lookback, p.state_dim, p.action_dim = history, history, 4, 2
    p.proprio_dim, p.horizon = 2, 3
    generator = torch.Generator().manual_seed(13)
    p.state = torch.randn(90, 4, generator=generator, dtype=p.dtype)
    p.action = torch.randn(90, 2, generator=generator, dtype=p.dtype)
    p.state_mean, p.state_scale = torch.arange(4).double(), torch.arange(1, 5).double()
    p.action_mean, p.action_scale = torch.tensor([.1, -.3]), torch.tensor([.2, 1.3])
    p.hankel_starts = torch.arange(90 - history - p.horizon + 1)
    rows = p.hankel_starts[:, None] + torch.arange(history)
    past = torch.cat([p.action[rows].flatten(1), p.state[rows].flatten(1)], 1)
    p.g_lookup, p.g_lookup_scale = "action_ridge", .7
    p.retrieval = make_retrieval(p.g_lookup)
    p.g_lookup_rates, p._rate_keys = (1.0,), None
    p.g_align_history = False
    p.g_align_future = False
    p.g_rate_scaled_action_dims = ()
    p.g_lookup_matrix = torch.randn(past.shape[1], 3, generator=generator, dtype=p.dtype)
    p.pf = p._g_lookup_projection(past)
    p.pf_norm = torch.einsum("ij,ij->i", p.pf, p.pf)
    future = p.hankel_starts[:, None] + history + torch.arange(p.horizon)
    p.uf = p.action[future].flatten(1).T
    p.g_native, p.g_tolerance = False, None
    p.g_state_offset = 0
    p.g_constraint = "affine"
    p.g_solver = "pgd"
    p.g_supports, p.g_l1, p.g_iterations, p.g_weights = 7, .03, 200, "optimized"
    p.method, p.residual_lookup, p.residual_model = "bpc", "g_supports", "rff"
    p.rff_dtype, p.rff_dim = torch.float32, 1027
    p.rff_scale = math.sqrt(2 / p.rff_dim)
    p.rff_bank_weight = torch.randn(6, p.rff_dim, generator=generator)
    p.rff_query_weight = torch.randn(4, p.rff_dim, generator=generator)
    p.rff_bias = torch.randn(p.rff_dim, generator=generator)
    p.residual_weights = .05 * torch.randn(p.rff_dim, 2, generator=generator)
    p.reset()
    return p


def exact_policy(history=5):
    policy = toy_policy(history)
    policy.g_constraint, policy.g_l1 = "affine", 0.
    policy.g_solver = "exact"
    return policy


def fisher_recipe():
    method = method_config()
    method['horizon'] = 2
    method['g'].update(lookup='lda', lookup_rates=[.5, 1., 2.],
        align_history=True, align_future=True, state_offset=1,
        future_consistency=0, l1=0, solver='exact', constraint='affine',
        lda=dict(features=24, anchors=32, shrinkage=1., sigma_scale=.5,
                 seed=9, fisher_rank=6, teacher='sparsemax', adaptive_support='coverage'))
    method['residual'].update(lookup='g_supports', action_source='retimed_first')
    return method


def manuscript_fixture():
    """Small stochastic bank with enough future coordinates for ridge's rank guard."""
    bank = trajectories(demos=4, length=32)
    generator = np.random.default_rng(241)
    for trajectory in bank:
        for name, values in trajectory.items():
            trajectory[name] = values + generator.normal(0, .2, values.shape)
        action = generator.normal(0, .3, (4, 32))
        for step in range(1, action.shape[1]):
            action[:, step] += .85 * action[:, step-1]
        trajectory['u'] = action
    method = fisher_recipe()
    method['action_dim'] = 4
    return method, bank


def temporal_policy(rates=(.5, 1., 2.)):
    p = toy_policy(5)
    p.fit_residuals = True
    p.g_constraint, p.g_solver, p.g_l1 = 'affine', 'exact', 0
    p.g_state_offset = 1
    p.g_align_history = p.g_align_future = True
    p.g_lookup_rates = rates
    p.g_rate_scaled_action_dims = (0,)
    p.samples = len(p.action)
    p.trajectory_starts = torch.tensor([0])
    p.hankel_starts = p.hankel_starts[:-1]
    p.hankel_trajectory = torch.zeros_like(p.hankel_starts)
    p.uf = p.uf[:, :-1]
    p.pf = p._g_lookup_projection(p._past_columns(torch.arange(len(p.hankel_starts))))
    p.pf_norm = p.pf.square().sum(1)
    p._fit_sources = None
    p.rff_storage = 'resident'
    return p


def interpolation_policy(rates=None, offset=0):
    method = method_config()
    method["history"] = 3
    method["g"]["state_offset"] = offset
    method["g"]["lookup"] = "action_ridge"
    bank = [
        {key: (100 * demo + scale * np.arange(14, dtype=np.float64))[None]
         for key, scale in (("x", 1), ("agent", 2), ("wrist", 3), ("u", 4))}
        for demo in range(3)]
    policy = Policy(bank, data_config(), method,
                    fit_residuals=False)
    if rates is not None:
        policy.g_lookup_rates = tuple(rates)  # isolate time-map arithmetic from recipe validation
    width = policy.history * (policy.action_dim + policy.state_dim)
    policy.g_lookup_matrix = torch.eye(width, dtype=policy.dtype)
    policy._project_hankel()
    return policy
