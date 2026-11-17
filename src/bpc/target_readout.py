"""Bounded offline affine readout with each query's true adaptive support.

Padding is storage only. Each row uses its own k-column Householder basis;
extra reduced coordinates are disconnected unit equations with zero RHS.
Singular rows use the existing, unpadded minimum-norm solver.
"""
import math

import torch

from .affine import normal_equations
from .ragged_supports import RaggedColumns
from .readout_contracts import supports_readout


# Includes the history plane, masked copy, and centered copy. Further small
# Gram/future scratch is charged by _tile_rows. This bounds temporary working
# storage, not the support size: a larger support is still solved in full.
WORKSPACE_BYTES = 768 * 1024**2


def _capacity(count):
    if count == 1:
        return 1
    quantum = 16 if count <= 128 else 32 if count <= 256 else 64
    return math.ceil(count / quantum) * quantum


def _eligible(policy, queries):
    if getattr(policy, 'method', None) == 'bpc_nnc':
        return False
    return (supports_readout(policy)
            and policy.g_lookup == 'lda' and policy.support_selector.config.get('fisher_rank')
            and policy.g_native and policy.g_fit_dtype == torch.float32
            and policy.g_solver == 'exact' and policy.g_constraint == 'affine'
            and policy.g_weights != 'uniform' and policy.g_tolerance is None
            and queries.device == policy.hankel_starts.device and len(queries) > 0)


def _tile_rows(policy, capacity):
    history_width = policy.history * (policy.action_dim + policy.state_dim)
    future_width = policy.horizon * policy.action_dim
    per_row = capacity * (history_width * 12 + future_width * 16)
    per_row += capacity * capacity * 40
    return max(1, WORKSPACE_BYTES // max(1, per_row))


@torch.no_grad()
def solve_padded_affine(policy, design, query, counts, *, _lengths=None, diagnostics=None):
    """Solve FP32 designs with true affine constraints, preserving singular fallback.

    ``design`` is [batch, history features, storage capacity]. Columns at or
    beyond ``counts[row]`` are ignored, including their numerical values. Only
    full-rank rows use the padded Cholesky solve. Every failed guard is passed to
    the original FP64 minimum-norm fallback at its actual cardinality, retaining its eigensolve
    and rank cutoff. This deliberately allows different GEMM rounding.
    """
    if design.ndim != 3 or query.shape != design.shape[:2]:
        raise ValueError('affine design/query shape mismatch')
    if design.dtype != torch.float32 or query.dtype != torch.float32:
        raise ValueError('padded target solve requires FP32 design and query')
    if counts.shape != (len(query),) or counts.device != query.device:
        raise ValueError('affine counts must identify every query on its device')
    lengths = counts.cpu().tolist() if _lengths is None else _lengths
    batch, _, width = design.shape
    if any(k < 1 or k > width for k in lengths):
        raise ValueError('affine support count outside padded storage')
    if width == 1:
        return query.new_ones((batch, 1))
    position = torch.arange(width, device=query.device)[None]
    valid = position < counts[:, None]
    source = design.masked_fill(~valid[:, None], 0)
    number64 = counts.double()
    root64 = number64.sqrt()
    number, root = number64.float(), root64.float()
    total = source.sum(-1)
    pivot = source.gather(2, (counts - 1)[:, None, None].expand(-1, source.shape[1], 1))
    shift = (total + root[:, None] * pivot.squeeze(2)) / (number + root)[:, None]
    active = position[:, :-1] < (counts - 1)[:, None]
    reduced = (source[:, :, :-1] - shift[:, :, None]).masked_fill(~active[:, None], 0)
    target = query - total / number[:, None]
    del source, total, pivot, shift
    gram, rhs = normal_equations(reduced, target)
    del reduced, target
    margin = 8 * torch.finfo(torch.float32).eps * gram.abs().sum(-1).amax(-1)
    # Inactive coordinates are not part of the constrained problem. Their
    # independent x=0 equations cannot change active eigenvalues or the guard.
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    diagonal.add_(~active)
    guard = gram.clone()
    guard.diagonal(dim1=-2, dim2=-1).sub_(margin[:, None] * active)
    guard_factor, guard_info = torch.linalg.cholesky_ex(guard)
    # Some CUDA potrf versions return info=0 with NaNs for an indefinite
    # matrix. A finite factor is therefore part of the guard certificate.
    good = (guard_info == 0) & torch.isfinite(guard_factor).all((1, 2))
    del guard, guard_factor
    factor, info = torch.linalg.cholesky_ex(gram)
    good &= info == 0
    coordinates = torch.cholesky_solve(rhs, factor).squeeze(2)
    # Invalid Cholesky rows are discarded below before their output escapes.
    coordinates.masked_fill_(~active, 0)
    coefficient_shift = coordinates.sum(1) / (number64 + root64)
    output = torch.nn.functional.pad(coordinates, (0, 1)) - coefficient_shift[:, None]
    output.scatter_add_(1, (counts - 1)[:, None],
                        (-root64 * coefficient_shift)[:, None])
    output.add_(number64.reciprocal()[:, None]).masked_fill_(~valid, 0)
    output = output.to(query.dtype)
    good &= torch.isfinite(output).all(1)
    bad_rows = torch.nonzero(~good).flatten().cpu().tolist()
    groups = {}
    for row in bad_rows:
        groups.setdefault(lengths[row], []).append(row)
    for count, members in groups.items():
        rows = torch.tensor(members, device=query.device)
        fallback = design[rows, :, :count].double()
        transpose = fallback.transpose(1, 2)
        output[rows, :count] = policy._solve_g_normal(
            transpose @ fallback,
            (transpose @ query[rows, :, None].double()).squeeze(2)).to(query.dtype)
    if diagnostics is not None:
        diagnostics['solve_batches'] = diagnostics.get('solve_batches', 0) + 1
        diagnostics['fallback_rows'] = diagnostics.get('fallback_rows', 0) + len(bad_rows)
        diagnostics['fallback_groups'] = diagnostics.get('fallback_groups', 0) + len(groups)
    return output


@torch.no_grad()
def offline_prior(policy, rows, selected, queries, *, packed=None,
                  supports_validated=False):
    """Return (prior, same-demo counts), or None for the existing grouped path.

    The caller performs the usual offline exclusion validation first and owns
    the ragged neighbor metadata. No support is discarded, introduced, sorted,
    or capped here. The original selector rate metric is used on the full-
    precision histories, once per query rather than once per true-k group.
    """
    if not supports_validated or not _eligible(policy, queries):
        return None
    lengths = [len(columns) for columns, _ in selected]
    if (len(lengths) != len(rows) or any(not count for count in lengths)
            or any(columns.ndim != 1 or columns.dtype != torch.long
                   or columns.device != rows.device for columns, _ in selected)):
        return None
    packed = RaggedColumns.from_rows([columns for columns, _ in selected]) if packed is None else packed
    groups = {}
    for index, count in enumerate(lengths):
        groups.setdefault(_capacity(count), []).append(index)
    p = policy
    temporal = len(p.g_lookup_rates) > 1 and (p.g_align_history or p.g_align_future)
    if temporal:
        keys = p._g_lookup_projection(queries.to(p.g_lookup_matrix.dtype)).float()
        choices = queries.new_tensor([1., *[rate for rate, *_ in p._interpolated_keys()]])
    prior = queries.new_empty(len(rows), p.horizon * p.action_dim)
    retained = rows.new_empty(len(rows))
    counts = rows.new_tensor(lengths)
    diagnostics = dict(rows=len(rows), buckets=len(groups), projection_calls=int(temporal),
                       solve_batches=0, fallback_rows=0, fallback_groups=0,
                       selected_columns=sum(lengths), padded_columns=0)
    for capacity, members in groups.items():
        tile = _tile_rows(p, capacity)
        for start in range(0, len(members), tile):
            subset = members[start:start + tile]
            indices = rows.new_tensor(subset)
            count = counts[indices]
            position = torch.arange(capacity, device=rows.device)[None]
            valid = position < count[:, None]
            # Repeat the first genuine support in scratch slots so temporal
            # decoders see valid storage/boundaries. The solve masks them out.
            address = packed.offsets[indices, None] + torch.where(valid, position, 0)
            supports = packed.values[address]
            query, target_rows = queries[indices], rows[indices]
            rates = None
            if temporal:
                _, rate_indices = p.support_selector._distance(keys[indices], target_rows, supports)
                rates = choices[rate_indices.long()]
            history = p._reconstruction_columns(supports, query, torch.float32,
                                                 target_rows, rates=rates)
            future = p._future_columns(supports, query, target_rows, rates=rates)
            coefficient = solve_padded_affine(p, history.transpose(1, 2), query.float(),
                count, _lengths=[lengths[row] for row in subset], diagnostics=diagnostics)
            prior[indices] = torch.einsum('bkh,bk->bh', future.float(), coefficient).to(p.dtype)
            retained[indices] = ((p.hankel_trajectory[supports]
                == p.trajectory_of[target_rows, None]) & valid).sum(1)
            diagnostics['padded_columns'] += len(indices) * capacity
    p._target_readout_last = diagnostics
    return prior, retained
