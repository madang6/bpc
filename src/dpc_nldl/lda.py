"""Soft-label Fisher LDA in the exact training span; no class-squared matrix."""
import torch


@torch.no_grad()
def fit_discriminant(features, targets, shrinkage=1., *, fisher_rank):
    """Return the rank-constrained Fisher metric in the complete training span.

    QR retains the complete row span, including dependent directions: there is
    no singular-value cutoff or rank approximation. Class means have no component
    outside that span, so the omitted ridge-only subspace contributes zero.
    Fisher directions maximize between/regularized-within variance; selecting
    their rank defines a different retrieval model, not a score approximation.
    """
    if not isinstance(fisher_rank, int) or fisher_rank < 1:
        raise ValueError("a positive Fisher rank is required")
    x, target = features.double(), targets.double()
    if (x.ndim != 2 or target.ndim != 2 or len(x) != len(target) or not len(x)
            or not torch.isfinite(x).all() or not shrinkage > 0):
        raise ValueError("finite features, nonnegative labels and positive shrinkage required")
    # A full-bank host teacher can exceed 20 GiB. isfinite's temporary
    # allocations must not scale to another complete teacher matrix. Boolean
    # validation is blockwise; the floating-point reductions below are unchanged.
    for block in target.split(4096, dim=1):
        if not torch.isfinite(block).all() or (block < 0).any():
            raise ValueError("finite features, nonnegative labels and positive shrinkage required")
    total = target.sum(1, keepdim=True)
    if (total <= 0).any():
        raise ValueError("every history requires teacher mass")
    total = total.to(x.device)
    center = x.mean(0)
    x = x - center
    if x.shape[1] > len(x):
        basis, triangular = torch.linalg.qr(x.T, mode="reduced")
        reduced = triangular.T
    else:
        basis, reduced = None, x
    between_covariance = None
    represented_classes = 0
    # Fisher needs only the sum of class contributions, not all class means.
    for block in target.split(4096, dim=1):
        # A host-resident teacher avoids a full-bank FP64 allocation on the
        # GPU. Covariance accumulation keeps the same classes, weights and
        # block order; only one class block is transferred at a time.
        normalized = block.to(x.device) / total
        mass = normalized.sum(0)
        if (mass <= 0).any():
            # Zero-mass classes contribute exactly zero to Fisher covariance.
            # No smoothing, fictitious class means or retrieval-bank pruning.
            positive = mass > 0
            normalized, mass = normalized[:, positive], mass[positive]
            if not len(mass):
                continue
        represented_classes += len(mass)
        means = (reduced.T @ normalized) / mass
        between = means * (mass / len(x)).sqrt()
        if between_covariance is None:
            between_covariance = between @ between.T
        else:
            # Accumulate directly rather than allocating and then adding a
            # second span-squared covariance for every class block.
            between_covariance.addmm_(between, between.T)
    del normalized, between, means
    covariance = (reduced.T @ reduced) / len(x) - between_covariance
    covariance = (covariance + covariance.T) * .5
    ridge = shrinkage * covariance.trace() / x.shape[1]
    if not torch.isfinite(ridge) or ridge <= 0:
        raise ValueError("positive within-class variance required")
    covariance.diagonal().add_(ridge)
    factor = torch.linalg.cholesky(covariance)
    if not 0 < fisher_rank <= len(factor):
        raise ValueError('Fisher rank must fit the training span')
    whitened = torch.linalg.solve_triangular(factor, between_covariance, upper=False)
    whitened = torch.linalg.solve_triangular(factor, whitened.T, upper=False).T
    whitened = (whitened + whitened.T) * .5
    values, vectors = torch.linalg.eigh(whitened)
    values, vectors = values[-fisher_rank:], vectors[:, -fisher_rank:]
    residual = (whitened @ vectors - vectors * values).norm() / whitened.norm().clamp_min(1e-30)
    transform = torch.linalg.solve_triangular(factor.T, vectors, upper=True)
    if basis is not None:
        transform = basis @ transform
    return dict(transform=transform, ridge=float(ridge),
                relative_residual=float(residual), rank=fisher_rank,
                features=x.shape[1], eigenvalues=values.cpu().tolist(),
                represented_teacher_classes=represented_classes,
                total_teacher_classes=target.shape[1])
