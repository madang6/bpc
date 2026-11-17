"""Seeded execution fixtures and an independent grouped-dispatch reference."""
import torch

from bpc.bpc import Policy
from bpc.ragged_supports import RaggedColumns, support_groups
from tests.support.policies import (
    data_config, fisher_recipe, method_config, temporal_policy, trajectories,
)


def equal(a, b):
    if torch.is_tensor(a):
        return (a.shape == b.shape and a.dtype == b.dtype and torch.equal(
            a.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8),
            b.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))


def ragged_readout_fixture(device='cpu', policy_type=Policy):
    method = fisher_recipe()
    method['device'] = device
    method['g'].update(lookup='lda', fit_dtype='float32', constraint='affine',
                       solver='exact', l1=0, lda=dict(features=8, fisher_rank=6, teacher='sparsemax', adaptive_support='coverage'))
    p = policy_type(trajectories(demos=4, length=24), data_config(), method,
               fit_residuals=True)
    generator = torch.Generator().manual_seed(127)
    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64).to(device)
    width = p.history*(p.action_dim+p.state_dim)
    p.support_selector.omega = random(width, 8)/5
    p.support_selector.phase = random(8)
    p.g_lookup_matrix = random(8, 6)
    p._project_hankel()
    p.omega, p.phase = random(p.input_dim, p.rff_dim), random(p.rff_dim)
    p._factor_rff()
    p.residual_weights = random(p.rff_dim, p.action_dim).to(p.rff_dtype)
    p.fixed_shape_readout = True
    query = p._queries(torch.as_tensor((7, 8, 9, 10, 11, 12), device=p.device))
    split = p.history * p.action_dim
    actions = query[:, :split].reshape(-1, p.history, p.action_dim)
    states = query[:, split:].reshape(-1, p.history, p.state_dim)
    selected = [(torch.arange(2, 2+count, device=device),
                 torch.arange(count, device=device, dtype=p.dtype))
                for count in (1, 3, 2, 3, 1, 2)]
    return p, actions, states, selected


class VariableSelector:
    def __init__(self, policy):
        self.policy = policy

    def select(self, queries, target_rows=None, target_demos=None, allowed=None):
        policy = self.policy
        excluded = (policy._hankel_exclusion(policy.hankel_starts, target_rows, target_demos)
                    if target_rows is not None or target_demos is not None else
                    torch.zeros(len(queries), len(policy.hankel_starts), dtype=torch.bool))
        if allowed is not None:
            excluded |= ~allowed
        output = []
        for row in range(len(queries)):
            columns = torch.nonzero(~excluded[row]).flatten()[:1 + row % 3]
            output.append((columns, queries.new_zeros(len(columns))))
        return output

    def rate_trace(self, queries, supports, target_rows=None):
        return {}


def ragged_policy():
    method = method_config()
    method["g"].update(future_consistency=0, solver="exact", constraint="affine", l1=0)
    method["residual"]["lookup"] = "g_supports"
    method["residual"]["target_batch_size"] = 5
    method["rff"].update(dtype="float32", features=16, refine_steps=4, refine_tolerance=1e-5)
    result = Policy(trajectories(), data_config(), method)
    result.support_selector = VariableSelector(result)
    result.method = "bpc"
    return result


def literal_outer_offline(model, rows, leave_demo_out):
    """Original ragged dispatch; each fixed-cardinality solve is unchanged."""
    query = model._queries(rows)
    selected = model.support_selector.select(query, target_rows=rows,
        target_demos=model.trajectory_of[rows] if leave_demo_out else None)
    prior = query.new_empty(len(rows), model.horizon*model.action_dim)
    retained = rows.new_empty(len(rows))
    excluded = (torch.bincount(model.hankel_trajectory, minlength=model.demo_count)[model.trajectory_of[rows]]
        if leave_demo_out else torch.searchsorted(model.hankel_starts, rows, right=True)
        -torch.searchsorted(model.hankel_starts, rows-model.history))
    for indices, columns, _ in support_groups(selected, model.device):
        result = model._offline_g(rows[indices], leave_demo_out=leave_demo_out,
                                  atlas=(columns, excluded[indices]))
        prior[indices], retained[indices] = result[0], result[2]
    neighbors = RaggedColumns.from_rows([model._support_endpoints(c) for c, _ in selected])
    return prior, excluded, retained, neighbors


def offline_fixture(device='cpu', rates=(.5, 1., 2.)):
    policy = temporal_policy(rates)
    policy.g_fit_dtype = torch.float32
    policy.trajectory_of = torch.zeros(policy.samples, dtype=torch.long)
    policy.device = torch.device(device)
    for key, value in vars(policy).copy().items():
        if torch.is_tensor(value):
            setattr(policy, key, value.to(device))
    rows = torch.tensor([17, 20, 23], device=device)
    query = policy._queries(rows)
    supports, _, _ = policy._hankel_topk(query, policy.g_supports, target_rows=rows)
    return policy, rows, query, supports


def affine_problems(device='cpu', dtype=torch.float64, fixed=False):
    policy = Policy.__new__(Policy)
    policy.device = torch.device(device)
    policy.history, policy.action_dim, policy.state_dim, policy.horizon = 3, 2, 9, 4
    policy.g_solver = 'exact'
    policy.fixed_shape_readout = fixed
    generator = torch.Generator(device=device).manual_seed(60821)
    problems = []
    for batch, count in ((2, 1), (4, 7), (2, 13), (3, 35)):
        design = torch.randn(batch, 33, count, device=device, dtype=dtype, generator=generator)
        if count > 1:
            design[0, :, -1] = design[0, :, 0]  # Mixed good/fallback within groups.
        query = torch.randn(batch, 33, device=device, dtype=dtype, generator=generator)
        problems.append((design, query))
    return policy, problems
