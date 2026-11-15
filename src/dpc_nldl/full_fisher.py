"""Full-history Fisher accumulation ported from the evaluated all-window fitter.

Disk-backed features and sparse memberships avoid a dense histories-squared
teacher. Temporary arrays are private to each fit and removed on completion.
"""
import json
import tempfile
from pathlib import Path
import numpy as np
import scipy.linalg
import scipy.sparse
from threadpoolctl import threadpool_limits
import torch


def write_json(path, value):
    path.write_text(json.dumps(value))

def save_array(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as output:
        np.save(output, value)
    temporary.replace(path)


@torch.no_grad()
def cache_features(policy, cache, progress):
    count = len(policy.hankel_starts)
    width = len(policy.support_selector.phase)
    path = cache/'features.npy'
    checkpoint = cache/'features_progress.json'
    completed = json.loads(checkpoint.read_text())['completed'] if checkpoint.exists() else 0
    if completed:
        features = np.lib.format.open_memmap(path, mode='r+')
        assert features.shape == (count, width) and features.dtype == np.float64
    else:
        features = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(count, width))
    for start in range(completed, count, 1024):
        stop = min(start+1024, count)
        columns = torch.arange(start, stop, device=policy.device)
        rows = policy.support_selector.features(policy._past_columns(columns))
        assert rows.dtype == torch.float64 and torch.isfinite(rows).all()
        features[start:stop] = rows.cpu().numpy()
        if stop % 8192 == 0 or stop == count:
            features.flush()
            write_json(checkpoint, dict(completed=stop, total=count))
        progress('history_features', stop, count)
    del features
    return np.load(path, mmap_mode='r')


@torch.no_grad()
def cache_teacher(policy, cache, progress):
    from .sparse_qp import sparsemax

    count = len(policy.hankel_starts)
    path = cache/'teacher.npz'
    if path.exists():
        teacher = scipy.sparse.load_npz(path).tocsr()
        assert teacher.shape == (count, count)
        return teacher
    future_float = policy._action_neighborhood_features()
    future = future_float.double()
    columns = torch.arange(count, device=policy.device)
    norms = future_float.square().sum(1)
    radius_count = min(4*policy.g_supports, count-1)
    pieces = []
    block_folder = cache/'teacher_blocks'
    block_folder.mkdir(exist_ok=True)
    for start in range(0, count, 2048):
        stop = min(start+2048, count)
        block_path = block_folder/f'{start:07d}.npz'
        if block_path.exists():
            block = scipy.sparse.load_npz(block_path)
            assert block.shape == (stop-start, count)
            pieces.append(block)
            continue
        row_parts, column_parts, value_parts = [], [], []
        for first in range(start, stop, 32):
            last = min(first+32, stop)
            batch = columns[first:last]
            _, nearest = policy._metric_topk(batch, columns, future_float, radius_count, 32, norms)
            radius = nearest[:, -1].double().clamp_min(1e-12)
            distance = torch.cdist(future[batch], future).square()
            overlap = ((policy.hankel_trajectory[batch, None] == policy.hankel_trajectory[None])
                       & ((policy.hankel_starts[batch, None]-policy.hankel_starts[None]).abs()
                          < policy.history+policy.horizon))
            logits = -distance/radius[:, None]
            probability = sparsemax(logits.masked_fill(overlap, -torch.inf))
            total = probability.sum(1, keepdim=True)
            assert torch.isfinite(probability).all() and (total > 0).all()
            assert (probability.masked_select(overlap) == 0).all()
            probability /= total
            rows, selected = torch.nonzero(probability > 0, as_tuple=True)
            row_parts.append((rows+first-start).cpu().numpy())
            column_parts.append(selected.cpu().numpy())
            value_parts.append(probability[rows, selected].cpu().numpy())
        block = scipy.sparse.csr_matrix((np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts))), shape=(stop-start, count))
        assert np.allclose(np.asarray(block.sum(1)).ravel(), 1, rtol=0, atol=1e-12)
        temporary = block_path.with_name(block_path.stem+'.tmp.npz')
        scipy.sparse.save_npz(temporary, block)
        temporary.replace(block_path)
        pieces.append(block)
        progress('teacher_memberships', stop, count)
    teacher = scipy.sparse.vstack(pieces, format='csr')
    temporary = path.with_name('teacher.tmp.npz')
    scipy.sparse.save_npz(temporary, teacher)
    temporary.replace(path)
    return teacher


@torch.no_grad()
def scatters(features, teacher, device, progress, cache=None, row_batch=2048, class_batch=256):
    count, width = features.shape
    assert teacher.shape[0] == count
    destination = cache/'scatter.pt' if cache is not None else None
    if destination is not None and destination.exists():
        return torch.load(destination, map_location='cpu', weights_only=False)
    mean = np.zeros(width, dtype=np.float64)
    for start in range(0, count, row_batch):
        mean += np.asarray(features[start:start+row_batch]).sum(0)/count
    mean_gpu = torch.from_numpy(mean).to(device)
    rng = np.random.default_rng(72914)
    probe = rng.standard_normal((width, min(16, width)))/np.sqrt(width)
    probe_gpu = torch.from_numpy(probe).to(device)
    projected = np.empty((count, probe.shape[1]), dtype=np.float64)
    total = torch.zeros((width, width), dtype=torch.float64, device=device)
    for start in range(0, count, row_batch):
        stop = min(start+row_batch, count)
        rows = torch.from_numpy(np.array(features[start:stop])).to(device)
        rows -= mean_gpu
        total.addmm_(rows.T, rows, alpha=1/count)
        projected[start:stop] = (rows@probe_gpu).cpu().numpy()
        progress('total_covariance', stop, count)
    del rows
    membership = teacher.T.tocsr()
    mass = np.asarray(membership.sum(1)).ravel()
    between = torch.zeros_like(total)
    for start in range(0, membership.shape[0], class_batch):
        stop = min(start+class_batch, membership.shape[0])
        block = membership[start:stop]
        positive = mass[start:stop] > 0
        if not positive.any():
            continue
        block = block[positive]
        indices, inverse = np.unique(block.indices, return_inverse=True)
        weights = torch.sparse_csr_tensor(torch.from_numpy(block.indptr.astype(np.int64)),
            torch.from_numpy(inverse.astype(np.int64)), torch.from_numpy(block.data),
            size=(block.shape[0], len(indices)), dtype=torch.float64, device=device)
        rows = torch.from_numpy(np.array(features[indices])).to(device)
        weighted_sum = torch.sparse.mm(weights, rows)
        class_mass = torch.from_numpy(mass[start:stop][positive]).to(device)
        directions = (weighted_sum-class_mass[:, None]*mean_gpu)/(class_mass[:, None]*count).sqrt()
        between.addmm_(directions.T, directions)
        progress('between_covariance', stop, membership.shape[0])
    del rows, weights, weighted_sum, directions
    total_probe = projected.T@projected/count
    class_projection = membership@projected
    positive = mass > 0
    class_projection = class_projection[positive]/np.sqrt(mass[positive, None]*count)
    between_probe = class_projection.T@class_projection
    actual_total = (probe_gpu.T@total@probe_gpu).cpu().numpy()
    actual_between = (probe_gpu.T@between@probe_gpu).cpu().numpy()
    checks = dict(total_projected_relative_error=float(np.linalg.norm(actual_total-total_probe)/np.linalg.norm(total_probe)),
        between_projected_relative_error=float(np.linalg.norm(actual_between-between_probe)/np.linalg.norm(between_probe)),
        teacher_row_mass_max_error=float(np.max(np.abs(np.asarray(teacher.sum(1)).ravel()-1))))
    assert checks['total_projected_relative_error'] < 1e-9, checks
    assert checks['between_projected_relative_error'] < 1e-9, checks
    assert checks['teacher_row_mass_max_error'] < 1e-12, checks
    total -= between
    total = (total+total.T)*.5
    between = (between+between.T)*.5
    result = dict(within=total.cpu(), between=between.cpu(), feature_count=width,
        fitting_histories=count, represented_teacher_classes=int(positive.sum()),
        total_teacher_classes=len(mass), numerical_checks=checks)
    if destination is not None:
        temporary = destination.with_suffix('.tmp')
        torch.save(result, temporary)
        temporary.replace(destination)
    return result


def solve_full(stats, rank, threads=8):
    width = stats['feature_count']
    ridge = float(stats['within'].diagonal().sum())/width
    assert np.isfinite(ridge) and ridge > 0
    within = np.array(stats['within'].numpy(), order='F', copy=True)
    within.flat[::width+1] += ridge
    between = np.array(stats['between'].numpy(), order='F', copy=True)
    with threadpool_limits(limits=threads):
        eigenvalues, transform = scipy.linalg.eigh(between, within, subset_by_index=(width-rank, width-1),
            driver='gvx', overwrite_a=True, overwrite_b=True, check_finite=True)
        left = stats['between'].numpy()@transform
        regularized = stats['within'].numpy()@transform+ridge*transform
        residual = np.linalg.norm(left-regularized*eigenvalues)/max(np.linalg.norm(left), 1e-30)
        orthogonality = np.linalg.norm(transform.T@regularized-np.eye(rank))/np.sqrt(rank)
    assert residual < 1e-7 and orthogonality < 1e-7, (residual, orthogonality)
    diagnostics = dict(shrinkage=1., ridge=ridge, rank=rank, eigenvalues=eigenvalues.tolist(),
        relative_generalized_residual=float(residual), regularized_orthogonality_error=float(orthogonality),
        features=width, fitting_histories=stats['fitting_histories'],
        represented_teacher_classes=stats['represented_teacher_classes'],
        total_teacher_classes=stats['total_teacher_classes'], numerical_checks=stats['numerical_checks'])
    return torch.from_numpy(np.ascontiguousarray(transform)), diagnostics



@torch.no_grad()
def fit_all_histories(policy):
    if float(policy.support_selector.config['shrinkage']) != 1.:
        raise ValueError('The evaluated full-history recipe requires shrinkage=1')
    def progress(stage, completed=None, total=None):
        if completed == total:
            print(f'[fit] full Fisher {stage}: {completed}/{total}', flush=True)
    with tempfile.TemporaryDirectory(prefix='bpc-full-fisher-') as folder:
        cache = Path(folder)
        features = cache_features(policy, cache, progress)
        teacher = cache_teacher(policy, cache, progress)
        supports = np.diff(teacher.indptr)
        stats = scatters(features, teacher, policy.device, progress)
        transform, diagnostics = solve_full(stats, int(policy.support_selector.config['fisher_rank']))
        diagnostics['teacher_positive_support'] = dict(minimum=int(supports.min()), mean=float(supports.mean()), maximum=int(supports.max()))
        del features, teacher
    return dict(transform=transform, **diagnostics)
