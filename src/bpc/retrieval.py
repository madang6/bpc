"""Retrieval strategies and the shared future-action teacher metric.

A strategy implements ``fit(policy)`` and ``project(policy, queries)``. Passing
policy explicitly keeps strategies stateless, including when accelerated
readouts use an explicit state snapshot. Learned tensors stay in the existing
artifact fields; Fisher alone also owns an adaptive support selector.
"""
import torch


class RawLookup:
    def fit(self, policy):
        return {"name": "past"}

    def project(self, policy, queries):
        return queries


class ActionRidgeLookup:
    def fit(self, policy):
        if policy.g_lookup_matrix is None:
            policy.g_lookup_stats = policy._fit_action_ridge_lookup()
        return policy.g_lookup_stats

    def project(self, policy, queries):
        shape = queries.shape[:-1]
        rows = queries.reshape(-1, queries.shape[-1])
        # Keep online one-query arithmetic for bank keys and batched queries.
        return (policy.g_lookup_scale
                * torch.stack([row @ policy.g_lookup_matrix for row in rows])).reshape(
            *shape, policy.g_lookup_matrix.shape[1])


class FisherLookup:
    def fit(self, policy):
        if policy.support_selector.payload is None:
            policy.g_lookup_stats = policy.support_selector.fit()
        return policy.g_lookup_stats

    def project(self, policy, queries):
        return policy.support_selector.features(queries) @ policy.g_lookup_matrix


_LOOKUPS = {"past": RawLookup, "action_ridge": ActionRidgeLookup, "lda": FisherLookup}


def make_retrieval(name):
    """Construct one of the manuscript retrieval choices."""
    try:
        return _LOOKUPS[name]()
    except KeyError:
        raise ValueError(f"unknown G lookup {name}") from None


def is_builtin_retrieval(policy):
    """Whether optimized paths may assume the configured retrieval arithmetic."""
    expected = _LOOKUPS.get(policy.g_lookup)
    actual = getattr(policy, "retrieval", None)
    return expected is not None and all(
        getattr(getattr(actual, name, None), "__func__", None) is getattr(expected, name)
        for name in ("fit", "project"))


def configure_search_precision(policy, precision, *, for_targets=False):
    """Select search arithmetic while preserving metric and calibration artifacts.

    Target fitting uses reference arithmetic on CPU and keeps a reusable search.
    Runtime variants reset their search even when choosing the same precision.
    """
    if precision not in {"fp64", "mixed"}:
        raise ValueError("retrieval_precision must be fp64 or mixed")
    selector = getattr(policy, "support_selector", None)
    if precision == "mixed" and (policy.g_lookup != "lda" or selector is None
            or selector.config.get("adaptive_support") != "coverage"
            or not selector.config.get("fisher_rank")):
        raise ValueError("mixed retrieval requires Fisher adaptive coverage")
    if selector is not None:
        cuda = policy.device.type == "cuda"
        mode = ("mixed" if precision == "mixed" and (cuda or not for_targets) else
                "fused" if cuda else "reference")
        if not for_targets or getattr(selector, "_density_search_mode", None) != mode:
            selector._density_search_mode = mode
            selector._density_search = None


def squared_distance(x, y, y2):
    distance = x @ y.T
    return (distance.mul_(-2)
            .add_(torch.einsum("ij,ij->i", x, x)[:, None])
            .add_(y2).clamp_min_(0))


@torch.no_grad()
def fit_action_ridge(policy, batch_size, progress):
    """Fit the action-predictive Mahalanobis key used only for G support lookup."""
    columns = len(policy.hankel_starts)
    width = policy.history * (policy.action_dim + policy.state_dim)
    future_width = policy.horizon * policy.action_dim
    past_sum = torch.zeros(width, device=policy.device, dtype=torch.float64)
    future_sum = torch.zeros(future_width, device=policy.device, dtype=torch.float64)
    gram = torch.zeros(width, width, device=policy.device, dtype=torch.float64)
    cross = torch.zeros(width, future_width, device=policy.device, dtype=torch.float64)
    energy = torch.zeros((), device=policy.device, dtype=torch.float64)
    for start in range(0, columns, batch_size):
        stop = min(start + batch_size, columns)
        rows = torch.arange(start, stop, device=policy.device)
        past = policy._past_columns(rows).double()
        future = policy.uf[:, start:stop].T.double()
        past_sum += past.sum(0)
        future_sum += future.sum(0)
        gram.addmm_(past.T, past)
        cross.addmm_(past.T, future)
        energy += future.square().sum()
        progress("action-ridge moments", stop, columns)
    past_mean = past_sum / columns
    future_mean = future_sum / columns
    gram.addmm_(past_mean[:, None], past_mean[None], beta=1, alpha=-columns)
    cross.addmm_(past_mean[:, None], future_mean[None], beta=1, alpha=-columns)
    energy -= columns * future_mean.square().sum()
    if energy <= 0:
        raise ValueError("action_ridge future actions have zero variance")
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    eigenvectors = eigenvectors.flip(1)
    rotated = eigenvectors.T @ cross
    explained = rotated.square().sum(1)
    scale = max(float(eigenvalues.mean()), 1e-30)
    grid = torch.logspace(
        -6, 3, 46, device=policy.device, dtype=torch.float64) * scale
    denominator = eigenvalues[:, None] + grid
    rss = (energy - 2 * (explained[:, None] / denominator).sum(0)
           + (eigenvalues[:, None] * explained[:, None]
              / denominator.square()).sum(0))
    trace = (eigenvalues[:, None] / denominator).sum(0)
    gcv = (rss / columns) / (1 - trace / columns).square().clamp_min(1e-30)
    index = int(torch.argmin(gcv))
    if index in (0, len(grid) - 1):
        raise ValueError("action_ridge GCV optimum is at the search boundary")
    ridge, rss = float(grid[index]), float(rss[index])
    r2 = 1 - rss / energy
    if r2 <= .02:
        raise ValueError(f"action_ridge fitting R^2 is too low: {r2:.6g}")
    matrix = eigenvectors @ (rotated / (eigenvalues + ridge)[:, None])
    prediction_covariance = matrix.T @ gram @ matrix / max(columns - 1, 1)
    rank = int(torch.linalg.matrix_rank(prediction_covariance))
    if rank < 8:
        raise ValueError(f"action_ridge prediction covariance rank is {rank}")
    sample = torch.linspace(0, columns - 1, min(columns, 2048),
                            device=policy.device).long()
    sample_past = policy._past_columns(sample).double() - past_mean
    prediction = sample_past @ matrix
    projected_norm = torch.linalg.vector_norm(prediction, dim=1).median()
    if projected_norm == 0:
        raise ValueError("action_ridge projection has zero matched scale")
    matched_scale = float(
        torch.linalg.vector_norm(sample_past, dim=1).median() / projected_norm)
    policy.g_lookup_matrix = matrix.to(policy.dtype)
    policy.g_lookup_scale = matched_scale
    return {
        "ridge": ridge,
        "gcv_index": index,
        "fitting_r2": r2,
        "prediction_rank": rank,
        "matched_scale": matched_scale,
    }


@torch.no_grad()
def metric_topk(policy, anchors, candidates, features, count, batch_size, norms=None):
    """Exact nonoverlapping neighbors in a supplied column feature space."""
    output, distances = [], []
    norms = (features[candidates].square().sum(1) if norms is None else norms)
    length = policy.history + policy.horizon
    for start in range(0, len(anchors), batch_size):
        batch = anchors[start:start + batch_size]
        distance = squared_distance(features[batch], features[candidates], norms)
        starts = policy.hankel_starts[candidates]
        query_starts = policy.hankel_starts[batch]
        left = torch.searchsorted(starts, query_starts - length + 1)
        right = torch.searchsorted(starts, query_starts + length - 1, right=True)
        positions = left[:, None] + torch.arange(
            2 * length - 1, device=policy.device)
        safe = positions.clamp_max(len(candidates) - 1)
        valid = ((positions < right[:, None])
                 & (policy.hankel_trajectory[candidates[safe]]
                    == policy.hankel_trajectory[batch, None]))
        rows = torch.arange(len(batch), device=policy.device)[:, None]
        distance[rows.expand_as(positions)[valid], positions[valid]] = torch.inf
        values, indices = torch.topk(distance, count, largest=False)
        if not torch.isfinite(values).all():
            raise ValueError("too few nonoverlapping action-neighborhood candidates")
        output.append(candidates[indices])
        distances.append(values)
    return torch.cat(output), torch.cat(distances)


@torch.no_grad()
def future_action_features(policy, future=None):
    action = policy.action.float()
    median = action.median(0).values
    robust = ((torch.quantile(action, .75, dim=0)
               - torch.quantile(action, .25, dim=0)) / 1.349)
    scale = torch.where(robust > 1e-6, robust, action.std(0)).clamp_min(1e-6)
    future = policy.uf.T if future is None else future
    return ((future.float().reshape(*future.shape[:-1], policy.horizon, policy.action_dim)
             - median) / scale).flatten(-2)
