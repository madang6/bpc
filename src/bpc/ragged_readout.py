"""Bounded temporal preparation at the original ragged readout group shapes."""
import torch

from .readout_contracts import supports_readout


PREPARED_BYTES = 256 * 1024**2


def _eligible(policy, actions, states):
    if getattr(policy, 'method', None) == 'bpc_nnc':
        return False
    return (actions.is_cuda and states.device == actions.device
            and policy.g_native and policy.g_align_history and policy.g_align_future
            and len(policy.g_lookup_rates) > 1 and policy.g_lookup == 'lda'
            and policy.g_solver == 'exact' and policy.g_constraint == 'affine'
            and supports_readout(policy, online=True))


@torch.no_grad()
def prepare_readout(policy, actions, states, selected, *, _groups=None):
    """Return prepared original groups, or None for the unchanged lazy path.

    Only routing and gather scheduling change. The ordinary group order,
    projection shapes and row slots are retained. A bounds certificate precedes
    optional raw gathers; one combined status check then precedes *all* solves.
    Invalid scratch is discarded, and None replays the original groups in their
    original error order using the already-selected supports.
    """
    if not _eligible(policy, actions, states):
        return None
    from .ragged_supports import support_groups
    from .fused_temporal_history import reconstruction_columns
    from .fused_temporal_future import future_columns
    try:
        groups = list(support_groups(selected, actions.device)) if _groups is None else _groups
        if len(groups) <= 1:
            return None
        p = policy
        count = len(p.hankel_starts)
        demos = len(p.trajectory_starts)
        if (not count or not demos or p.hankel_starts.dtype != torch.long
                or p.hankel_trajectory.dtype != torch.long
                or p.hankel_starts.ndim != 1
                or p.state.ndim != 2 or p.action.ndim != 2
                or p.state.shape[1] != p.state_dim or p.action.shape[1] != p.action_dim
                or p.hankel_trajectory.shape != p.hankel_starts.shape
                or p.hankel_starts.device != actions.device
                or p.hankel_trajectory.device != actions.device
                or p.trajectory_starts.device != actions.device
                or any(columns.dtype != torch.long or columns.device != actions.device
                       or columns.ndim != 2 for _, columns, _ in groups)):
            return None
        elements = sum(columns.numel() for _, columns, _ in groups)
        width = p.history*(p.state_dim+p.action_dim) + p.horizon*(p.action_dim+p.proprio_dim)
        if elements*width*p.state.element_size() > PREPARED_BYTES:
            return None
        columns = torch.cat([columns.reshape(-1) for _, columns, _ in groups])
        invalid = (columns < -count) | (columns >= count)
        trajectories = p.hankel_trajectory[columns.clamp(-count, count-1)]
        invalid |= (trajectories < -demos) | (trajectories >= demos)
        if bool(invalid.any()):
            return None
        errors, prepared = [], []
        for rows, columns, distances in groups:
            u, y = actions[rows], states[rows]
            query = torch.cat([u.flatten(1), y.flatten(1)], 1)
            trace = p._rate_trace(query, columns)
            rates = trace.get('hankel_rates')
            plane = reconstruction_columns(p, columns, rates, _errors=errors)
            future = future_columns(p, columns, rates, _errors=errors)
            state_future = future_columns(p, columns, rates, states=True, _errors=errors)
            if plane is None or future is None or state_future is None:
                return None
            prepared.append((rows, u, y, columns, distances,
                             (query, trace, plane, future, state_future)))
        if errors and bool(torch.stack(errors).any()):
            return None
        return prepared
    except (RuntimeError, ValueError, IndexError, TypeError):
        # Preparation may reach a later group's error before an earlier solve.
        # The ordinary lazy loop is responsible for which error is observed.
        return None
