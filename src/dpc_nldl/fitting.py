"""Offline target construction and shared-support RFF fitting.

Fitting operates on a constructed policy and leaves its deployment parameters in
place. Run directories, caching and artifact compatibility belong to artifacts.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple
import math
import time

import torch

from .data import select_shard
from .host_paged import HostPagedAtlas
from .io import atomic_path
from .pipeline import target_execution_options
from .ragged_supports import RaggedColumns, pooled_mean
from .retrieval import configure_search_precision


FIT_ROWS = 512
SQRT2 = math.sqrt(2.0)


def progress(label, complete, total):
    if complete == total or complete % 512 == 0:
        print(f"[fit] {label} {complete}/{total}", flush=True)


def configure_target_retrieval(policy):
    """Set arithmetic only after fitting the unchanged metric/calibration."""
    options = getattr(policy, "target_execution_options", target_execution_options(policy.config))
    configure_search_precision(policy, options["target_retrieval_precision"], for_targets=True)


class TargetBatch(NamedTuple):
    """One target batch, shared by local fitting and the unchanged shard format."""
    rows: torch.Tensor
    targets: torch.Tensor
    exclusions: torch.Tensor
    same_demo: torch.Tensor
    g_neighbors: torch.Tensor | RaggedColumns

    def payload(self):
        return {name: value.payload() if isinstance(value, RaggedColumns) else value.cpu()
                for name, value in self._asdict().items()}

    @classmethod
    def from_payload(cls, payload, count):
        if "g_neighbors" not in payload:
            raise RuntimeError("fit target parts lack reusable G supports")
        values = (payload[name] for name in cls._fields)
        return cls(*(RaggedColumns(**value).take(slice(count)) if isinstance(value, dict)
                     else value[:count] for value in values))


def target_batch(policy, rows, **offline_options):
    base, excluded, same_demo, neighbors = policy._offline_g(rows, **offline_options)
    return TargetBatch(rows, policy.action[rows] - base[:, :policy.action_dim],
                       excluded, same_demo, neighbors)


@torch.no_grad()
def fit_targets(policy, folder, shard):
    """Cache leakage-safe G targets for a distributed fit."""
    configure_target_retrieval(policy)
    eligible = torch.nonzero(policy.row_in_trajectory >= policy.history).ravel()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    starts = select_shard(range(0, len(eligible), policy.target_batch), shard)
    for start in starts:
        path = folder / f"{start:09d}.pt"
        if path.is_file():
            continue
        rows = eligible[start:start + policy.target_batch]
        payload = target_batch(policy, rows).payload()
        with atomic_path(path) as temporary:
            torch.save(payload, temporary)
        progress("leakage-safe G targets", min(start + policy.target_batch, len(eligible)),
                  len(eligible))


def neighbor_elements(policy, rows, neighbors):
    neighbor_state = policy.state[neighbors]
    neighbor_action = policy.action[neighbors]
    delta = (neighbor_state - policy.state[rows, None]) / SQRT2
    return torch.cat([neighbor_state, neighbor_action, delta], 2)


def make_basis(policy, rows, neighbors, seed, sigma_scale):
    generator = torch.Generator(device=policy.device).manual_seed(seed)
    sample_rows = rows[torch.randperm(
        len(rows), generator=generator, device=policy.device)[:min(len(rows), 512)]]
    if isinstance(neighbors, RaggedColumns):
        selected = neighbors.take(sample_rows)
        keep = torch.randperm(len(selected.values), generator=generator,
                              device=policy.device)[:2000]
        query_rows = sample_rows[torch.searchsorted(selected.offsets[1:], keep, right=True)]
        states = policy.state[selected.values[keep]]
        elements = torch.cat([states, policy.action[selected.values[keep]],
                              (states - policy.state[query_rows]) / SQRT2], 1)
    else:
        elements = neighbor_elements(policy, sample_rows, neighbors[sample_rows]).reshape(
            -1, policy.input_dim)
    if len(elements) > 2000:
        keep = torch.randperm(
            len(elements), generator=generator, device=policy.device)[:2000]
        elements = elements[keep]
    distances = torch.cdist(elements, elements)
    median = float(distances[distances > 0].median().clamp_min(1e-6))
    policy.sigma = median * sigma_scale

    generator = torch.Generator(device=policy.device).manual_seed(seed)
    policy.omega = torch.randn(
        policy.input_dim, policy.rff_dim, generator=generator,
        device=policy.device, dtype=policy.dtype) / policy.sigma
    policy.phase = torch.rand(
        policy.rff_dim, generator=generator,
        device=policy.device, dtype=policy.dtype) * (2 * math.pi)
    policy._factor_rff()
    return median


def feature_atlas(policy, rows, neighbors, host_atlas=None, neighbor_actions=None):
    """Materialize the pooled map after projecting every bank row once."""
    tile = min(FIT_ROWS, policy.rff_dim)
    row_batch = 2 * FIT_ROWS
    itemsize = torch.empty((), dtype=policy.rff_dtype).element_size()
    if isinstance(neighbors, RaggedColumns):
        capacity = max(1, (128 * 1024**2) // (itemsize * neighbors.shape[1] * tile))
        row_batch = min(row_batch, 2 ** int(math.log2(capacity)))
    paged = policy.rff_storage == "host_paged"
    if policy.device.type == "cuda":
        device = (policy.device.index if policy.device.index is not None else
                  torch.cuda.current_device())
        free, _ = torch.cuda.mem_get_info(device)
        atlas_peak = (policy.state.numel() + policy.action.numel()
                      + (policy.samples + row_batch) * tile
                      + row_batch * neighbors.shape[1] * tile)
        working = itemsize * max(atlas_peak, 2 * policy.rff_dim**2) + 1_000_000_000
        atlas = itemsize * len(rows) * policy.rff_dim
        paged |= (policy.rff_storage == "auto" and policy.rff_dtype == torch.float32
                  and atlas + working > 0.95 * free)
        required = working + (3 * 8192 * math.ceil(policy.rff_dim / tile) * tile
                              * itemsize if paged else atlas)
        if required > 0.95 * free:
            raise RuntimeError(
                f"RFF atlas working set needs about {required / 1e9:.2f} GB, "
                f"but only {free / 1e9:.2f} GB is free")
    if policy.rff_dtype == torch.float32:
        action, state = policy._float_sources()
    else:
        state, action = policy.state.to(policy.rff_dtype), policy.action.to(policy.rff_dtype)
    output = ((host_atlas if host_atlas is not None else HostPagedAtlas(
        len(rows), policy.rff_dim, device=policy.device,
        page_rows=8192, feature_tile=tile)) if paged else
              torch.empty(len(rows), policy.rff_dim, device=policy.device,
                          dtype=policy.rff_dtype))
    if paged:
        output.begin_production()
        print(f"[fit] RFF atlas host-paged ({atlas / 1e9:.2f} GB)", flush=True)
    state_weight = policy.rff_bank_weight[:policy.state_dim]
    action_weight = policy.rff_bank_weight[policy.state_dim:]
    try:
        # These integer row slices and their exact original padded widths
        # repeat for every feature tile. Read counts once, preserving all
        # floating-point GEMM dimensions, strides and row placements.
        padding = (neighbors._padding_plan(row_batch)
                   if type(neighbors) is RaggedColumns else None)
        action_padding = (neighbor_actions._padding_plan(row_batch)
                          if type(neighbor_actions) is RaggedColumns else None)
        for start in range(0, policy.rff_dim, tile):
            stop = min(start + tile, policy.rff_dim)
            bank_phase = state @ state_weight[:, start:stop]
            bank_phase.addmm_(action, action_weight[:, start:stop])
            for first in range(0, len(rows), row_batch):
                last = min(first + row_batch, len(rows))
                query_phase = (state[rows[first:last]]
                               @ policy.rff_query_weight[:, start:stop])
                selected = (neighbors[first:last] if padding is None else
                            neighbors._padded(*padding[first // row_batch]))
                mask = selected >= 0 if isinstance(neighbors, RaggedColumns) else None
                selected = selected.clamp_min(0)
                phase = bank_phase[selected]
                if neighbor_actions is not None:
                    actions = (neighbor_actions[first:last] if action_padding is None else
                               neighbor_actions._padded(*action_padding[first // row_batch]))
                    action_delta = (actions.to(
                        policy.device, policy.rff_dtype) - action[selected])
                    del actions
                    phase.add_(action_delta @ action_weight[:, start:stop])
                    del action_delta
                phase.add_(policy.rff_bias[start:stop]).sub_(
                    query_phase[:, None]).cos_()
                pooled = pooled_mean(phase, mask).mul_(policy.rff_scale)
                if paged:
                    output.write_tile(page_index=first // output.page_rows,
                                      tile_index=start // tile,
                                      local_start=first % output.page_rows,
                                      values=pooled)
                else:
                    output[first:last, start:stop] = pooled
                del phase, query_phase
            del bank_phase
            print(f"[fit] RFF atlas {stop}/{policy.rff_dim}", flush=True)
        if paged:
            output.finish_production()
            output.allocate_replay_buffers()
    except BaseException:
        if paged:
            output.close()
        raise
    return output


def replay_rff(policy, features, consumer):
    if isinstance(features, HostPagedAtlas):
        features.replay(consumer)
    else:
        for start in range(0, len(features), FIT_ROWS):
            stop = min(start + FIT_ROWS, len(features))
            consumer(features[start:stop].to(policy.device), start, stop)


def refine_rff(policy, features, rhs, initial, factor):
    """Refine the deployed FP32 ridge heads using FP64 residuals."""
    steps = int(policy.config["rff"].get("refine_steps", 4))
    tolerance = float(policy.config["rff"].get("refine_tolerance", 5e-6))
    norms = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-30)

    def residual(solution):
        applied = policy.rff_ridge * solution
        def consume(page, start, stop):
            for offset in range(0, len(page), FIT_ROWS):
                batch = page[offset:offset + FIT_ROWS].double()
                applied.addmm_(batch.T, batch @ solution)
        replay_rff(policy, features, consume)
        return rhs - applied

    solution = initial
    error = residual(solution.double())
    relative = torch.linalg.vector_norm(error, dim=0) / norms
    worst = float(relative.max())
    if not math.isfinite(worst):
        raise RuntimeError("initial RFF ridge residual is non-finite")
    history = [worst]
    for _ in range(steps):
        if worst <= tolerance:
            break
        candidate = solution + torch.cholesky_solve(
            error.to(policy.rff_dtype), factor)
        candidate_error = residual(candidate.double())
        candidate_relative = torch.linalg.vector_norm(
            candidate_error, dim=0) / norms
        improved = torch.isfinite(candidate_relative) & (candidate_relative < relative)
        if not bool(improved.any()):
            break
        solution[:, improved] = candidate[:, improved]
        error[:, improved] = candidate_error[:, improved]
        relative[improved] = candidate_relative[improved]
        worst = float(relative.max())
        history.append(worst)
    if worst > tolerance:
        raise RuntimeError(
            f"deployed RFF ridge residual {worst:.3e} exceeds "
            f"{tolerance:.1e}")
    return solution, history, worst


def residual_training_neighbors(policy, g_neighbors=None):
    if g_neighbors is None:
        raise RuntimeError("shared-support RFF fitting requires retained target supports")
    return torch.nonzero(policy.row_in_trajectory >= policy.history).ravel(), g_neighbors


def fit_rff(policy, eligible, gip_targets, g_neighbors=None, host_atlas=None):
    def stamp():
        if policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)
        return time.perf_counter()

    started = stamp()
    rows, neighbors = residual_training_neighbors(policy, g_neighbors)
    neighbors_done = stamp()
    neighbor_actions = None
    actions_done = neighbors_done
    if getattr(policy, "residual_action_source", "native") == "retimed_first":
        # Reuse exactly the G-support endpoints and leakage exclusions used
        # for the residual targets. Keep the cache on CPU (FP32 for the
        # headline RFF), before releasing the compact temporal lookup keys.
        ragged = isinstance(neighbors, RaggedColumns)
        aligned = neighbors.take(rows) if ragged else None
        neighbor_actions = torch.empty(
            *((len(aligned.values),) if ragged else (len(rows), neighbors.shape[1])),
            policy.action_dim, device="cpu", dtype=policy.rff_dtype)
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            selected = neighbors[batch]
            if ragged:
                valid = selected >= 0
                selected = torch.where(valid, selected, selected[:, :1])
            values = policy.retimed_neighbor_actions(
                policy._queries(batch), selected, target_rows=batch)
            if ragged:
                first, last = aligned.offsets[[start, start + len(batch)]].tolist()
                neighbor_actions[first:last].copy_(values[valid].to("cpu", policy.rff_dtype))
            else:
                neighbor_actions[start:start + len(batch)].copy_(values.to("cpu", policy.rff_dtype))
            progress("retimed residual actions", start + len(batch), len(rows))
        if ragged:
            neighbor_actions = RaggedColumns(neighbor_actions, aligned.offsets.cpu())
        actions_done = stamp()
    policy.pf = policy.pf_norm = policy._support_screen = policy._rate_keys = None
    if policy.g_lookup == "lda":
        policy.support_selector.release()
    torch.cuda.empty_cache()
    rff = policy.config["rff"]
    median = make_basis(
        policy, rows, neighbors, int(rff["seed"]), float(rff["sigma_scale"]))
    basis_done = stamp()
    # Preserve the recorded recipe's joint solve and refinement arithmetic.
    # The first head is retained for artifact compatibility, not policy dispatch.
    targets = torch.zeros(
        len(rows), 2 * policy.action_dim, device=policy.device, dtype=policy.dtype)
    for start in range(0, len(rows), FIT_ROWS):
        stop = min(start + FIT_ROWS, len(rows))
        selected = neighbors[rows[start:stop]]
        targets[start:stop, :policy.action_dim] = (
            policy.action[rows[start:stop]]
            - pooled_mean(policy.action[selected.clamp_min(0)],
                          selected >= 0 if isinstance(neighbors, RaggedColumns) else None))
    positions = torch.searchsorted(rows, eligible)
    if (bool((positions >= len(rows)).any())
            or not torch.equal(rows[positions], eligible)):
        raise RuntimeError("RFF rows do not contain every residual target")
    targets[positions, policy.action_dim:] = gip_targets
    target_energy = gip_targets.double().pow(2).sum()
    previous_tf32 = (torch.backends.cuda.matmul.allow_tf32
                     if policy.device.type == "cuda" else None)
    if previous_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = False
    atlas = None
    try:
        atlas = (feature_atlas(policy, rows, neighbors.take(rows) if isinstance(
            neighbors, RaggedColumns) else neighbors[rows], host_atlas, neighbor_actions)
                 if policy.rff_atlas else None)
        atlas_done = stamp()
        policy._fit_sources = None
        torch.cuda.empty_cache()

        def replay(consumer):
            if atlas is not None:
                replay_rff(policy, atlas, consumer)
            else:
                for start in range(0, len(rows), FIT_ROWS):
                    stop = min(start + FIT_ROWS, len(rows))
                    features = []
                    for index, batch in enumerate(rows[start:stop].split(policy.feature_batch)):
                        selected = neighbors[batch]
                        arguments = ({"mask": selected >= 0}
                                     if isinstance(neighbors, RaggedColumns) else {})
                        if neighbor_actions is not None:
                            first = start + index * policy.feature_batch
                            arguments["neighbor_actions"] = neighbor_actions[
                                first:first + len(batch)].to(policy.device)
                        features.append(policy._query_features(
                            policy.state[batch], selected, **arguments))
                    consumer(torch.cat(features), start, stop)

        gram = torch.zeros(
            policy.rff_dim, policy.rff_dim,
            device=policy.device, dtype=policy.rff_dtype)
        rhs = torch.zeros(
            policy.rff_dim, 2 * policy.action_dim,
            device=policy.device, dtype=policy.rff_dtype)
        reference_rhs = torch.zeros_like(rhs, dtype=torch.float64) if atlas is not None else None

        def moments(features, start, stop):
            gram.addmm_(features.T, features)
            rhs.addmm_(features.T, targets[start:stop].to(policy.rff_dtype))
            if reference_rhs is not None:
                for offset in range(0, len(features), FIT_ROWS):
                    batch = features[offset:offset + FIT_ROWS].double()
                    reference_rhs.addmm_(
                        batch.T, targets[start + offset:start + offset + len(batch)].double())

        replay(moments)
        gram.diagonal().add_(policy.rff_ridge)
        moments_done = stamp()
        factor = torch.linalg.cholesky(gram)
        del gram
        joined = torch.cholesky_solve(rhs, factor)
        solve_done = stamp()
        if atlas is not None:
            joined, residuals, deployed_residual = refine_rff(
                policy, atlas, reference_rhs, joined, factor)
            policy.rff_refinement = {
                "iterations": len(residuals) - 1,
                "relative_residuals": residuals,
                "deployed_relative_residual": deployed_residual,
            }
        else:
            policy.rff_refinement = None
        policy.W_knn, policy.W_gip = joined.split(policy.action_dim, 1)
        refine_done = stamp()

        knn_error = torch.zeros((), device=policy.device, dtype=torch.float64)
        prediction_error = torch.zeros(
            (), device=policy.device, dtype=torch.float64)

        def diagnostics(features, start, stop):
            error = ((features @ joined).double() - targets[start:stop]).square()
            knn_error.add_(error[:, :policy.action_dim].sum())
            complete = policy.row_in_trajectory[rows[start:stop]] >= policy.history
            prediction_error.add_(error[complete, policy.action_dim:].sum())

        replay(diagnostics)
        diagnostics_done = stamp()
    finally:
        if isinstance(atlas, HostPagedAtlas):
            atlas.close()
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    policy.s2_knn = max(
        float(knn_error) / (len(rows) * policy.action_dim),
        policy.residual_variance_floor)
    policy.s2_gip = max(
        float(prediction_error) / gip_targets.numel(), policy.residual_variance_floor)
    policy.rff_timings = {
        "neighbors": neighbors_done - started,
        "retimed_actions": actions_done - neighbors_done,
        "basis": basis_done - actions_done,
        "atlas": atlas_done - basis_done,
        "moments": moments_done - atlas_done,
        "factor_and_solve": solve_done - moments_done,
        "refinement": refine_done - solve_done,
        "diagnostics": diagnostics_done - refine_done,
    }
    policy.rff_storage_used = ("host_paged" if isinstance(atlas, HostPagedAtlas) else
                             "resident" if atlas is not None else "none")
    del neighbors, neighbor_actions, joined, rhs, reference_rhs, targets, atlas
    del factor
    torch.cuda.empty_cache()
    return median, float(torch.sqrt(
        prediction_error / target_energy.clamp_min(1e-18)))


@torch.no_grad()
def fit(policy, target_folder=None):
    build_lookup = policy.g_lookup != "past" and policy.g_lookup_matrix is None
    eligible = torch.nonzero(policy.row_in_trajectory >= policy.history).ravel()
    # V79 overlaps pinned-host allocation with lookup fitting.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = (pool.submit(
            HostPagedAtlas, len(eligible), policy.rff_dim, device=policy.device,
            feature_tile=min(FIT_ROWS, policy.rff_dim))
            if policy.fit_residuals and policy.rff_storage == "host_paged" else None)
        lookup_stats = policy.fit_g_lookup()
        host_atlas = pending.result() if pending is not None else None
    if build_lookup and policy.support_selector is None:
        policy._project_hankel()
    if not policy.fit_residuals:
        policy.fit_stats = {
            "bank_rows": policy.samples,
            "hankel_columns": len(policy.hankel_starts),
            "g_lookup": lookup_stats,
            "g_weights": policy.g_weights,
        }
        return policy
    configure_target_retrieval(policy)
    policy._target_readout_totals = {}
    ragged = getattr(policy, "support_selector", None) is not None
    g_neighbors = ([] if ragged else torch.empty(
        policy.samples, policy.g_supports, dtype=torch.long, device=policy.device))
    cached = None
    paths = sorted(Path(target_folder).glob("*.pt")) if target_folder else []
    if paths:
        cached = [TargetBatch.from_payload(torch.load(path, map_location="cpu"),
                                           policy.target_batch) for path in paths]
        rows = torch.cat([part.rows for part in cached])
        if not torch.equal(rows, eligible.cpu()):
            raise RuntimeError("fit target parts are incomplete")
        targets = torch.cat([part.targets for part in cached]).to(policy.device)
        exclusions = [part.exclusions.to(policy.device) for part in cached]
        same_demo = [part.same_demo.to(policy.device) for part in cached]
        if ragged:
            g_neighbors.extend(part.g_neighbors.to(policy.device) for part in cached)
        else:
            for part in cached:
                g_neighbors[part.rows.to(policy.device)] = part.g_neighbors.to(policy.device)
    else:
        targets = torch.empty(
            len(eligible), policy.action_dim, device=policy.device, dtype=policy.dtype)
        exclusions, same_demo = [], []
        if policy.pf is not None:
            policy.pf = policy.pf.to(policy.device)
    target_started = time.perf_counter()
    atlas = (policy._g_support_atlas(eligible, min(
        policy.g_supports, len(policy.hankel_starts)))
             if cached is None and not ragged and policy.g_native and policy.device.type == "cuda"
             and policy.g_weights != "uniform" and policy.g_lookup != "past"
             and policy.pf is not None and policy.pf.dtype == torch.float64 else None)
    for start in range(0, len(eligible), policy.target_batch):
        rows = eligible[start:start + policy.target_batch]
        if cached is None:
            part = target_batch(policy, rows, atlas=None if atlas is None else
                                tuple(value[start:start + len(rows)] for value in atlas))
            targets[start:start + len(rows)] = part.targets
            exclusions.append(part.exclusions)
            same_demo.append(part.same_demo)
            if ragged:
                g_neighbors.append(part.g_neighbors)
            else:
                g_neighbors[rows] = part.g_neighbors
        progress("leakage-safe G targets", start + len(rows), len(eligible))
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    target_seconds = time.perf_counter() - target_started
    del atlas, cached
    if ragged:
        g_neighbors = RaggedColumns.concatenate(g_neighbors).align(eligible, policy.samples)

    median, relative_error = fit_rff(
        policy, eligible, targets, g_neighbors, host_atlas)
    exclusions = torch.cat(exclusions).float()
    same_demo = torch.cat(same_demo).float()
    policy.fit_stats = {
        "bank_rows": policy.samples,
        "eligible_rows": len(eligible),
        "hankel_columns": len(policy.hankel_starts),
        "g_lookup": lookup_stats,
        "g_weights": policy.g_weights,
        "temporal_neighbors": policy.neighbor_count,
        "residual_model": policy.residual_model,
        "residual_input_dim": policy.input_dim,
        "target_rows": "all_complete_histories",
        "g_target_seconds": target_seconds,
        "g_target_batch_size": policy.target_batch,
        "g_target_execution": dict(policy.target_execution_options),
        "g_target_readout": dict(policy._target_readout_totals),
        "g_normal_equations": (
            "none" if policy.g_weights == "uniform" else "adaptive_batched"
            if policy._target_readout_totals else "v79_microbatched"
            if (policy.g_native and policy.device.type == "cuda"
                and policy.g_tolerance is None) else "materialized"),
        "g_support_reused_for_rff": True,
        "offline_g_excluded_columns": {
            "minimum": int(exclusions.min()),
            "mean": float(exclusions.mean()),
            "maximum": int(exclusions.max()),
        },
        "retained_same_demo_supports": {
            "minimum": int(same_demo.min()),
            "mean": float(same_demo.mean()),
            "maximum": int(same_demo.max()),
        },
        "training_relative_error": relative_error,
        "gip_residual_variance": policy.s2_gip,
        "rff_features": policy.rff_dim, "rff_pairwise_median": median,
        "rff_sigma": policy.sigma, "ridge": policy.rff_ridge,
        "intercept": False,
        "knn_residual_variance": policy.s2_knn,
        "gram_rows": "all_complete_histories",
        "rff_compute_dtype": str(policy.rff_dtype).removeprefix("torch."),
        "feature_pass": ("single_preprojected_atlas" if policy.rff_atlas else
                         "batched_preprojected_features"),
        "rff_fit_seconds": policy.rff_timings,
        "rff_storage": policy.rff_storage_used,
        "rff_refinement": policy.rff_refinement,
    }
    return policy
