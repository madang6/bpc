"""Independent numerical oracles; never call the solver under test."""
import torch


def reference_prox(proposal, threshold):
    """Independent monotone-root reference, evaluated in double precision."""
    proposal, threshold = proposal.double(), threshold.double()[:, None]
    low = proposal.amin(1, keepdim=True) - threshold - 1
    high = proposal.amax(1, keepdim=True) + threshold + 1
    for _ in range(64):
        offset = (low + high) / 2
        centered = proposal - offset
        mass = (centered.sign() * (centered.abs() - threshold).relu()).sum(1, keepdim=True)
        low, high = torch.where(mass > 1, offset, low), torch.where(mass > 1, high, offset)
    centered = proposal - (low + high) / 2
    return centered.sign() * (centered.abs() - threshold).relu()


def literal_readout(p, action, state):
    """Scalar retrieval, independent affine-L1 reference, and full-width RFF."""
    query = torch.cat([action.flatten(), state.flatten()])
    key = p.g_lookup_scale * (query @ p.g_lookup_matrix)
    distance = (-2 * (key @ p.pf.T) + key.square().sum() + p.pf_norm).clamp_min(0)
    supports = distance.topk(p.g_supports, largest=False).indices
    rows = p.hankel_starts[supports, None] + torch.arange(p.history)
    plane = torch.cat([p.action[rows].flatten(1), p.state[rows].flatten(1)], 1)
    gram, rhs = plane @ plane.T, plane @ query
    lipschitz = torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-8)
    coefficients = torch.full_like(rhs, 1 / len(rhs))
    for _ in range(p.g_iterations):
        gradient = gram @ coefficients - rhs
        coefficients = reference_prox(
            (coefficients - gradient / lipschitz)[None],
            (p.g_l1 / lipschitz)[None])[0]
    prior = p.uf[:, supports] @ coefficients
    neighbors = p.hankel_starts[supports] + p.history - 1
    bank = torch.cat([p.state[neighbors], p.action[neighbors]], 1).to(p.rff_dtype)
    phase = bank @ p.rff_bank_weight + p.rff_bias
    phase -= state[-1].to(p.rff_dtype) @ p.rff_query_weight
    feature = phase.cos().mean(0) * p.rff_scale
    residual = feature @ p.W_gip
    return supports, prior, coefficients, residual
