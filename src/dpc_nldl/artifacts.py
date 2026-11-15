"""Run-level fitting, immutable artifact staging and policy loading.

Policy construction is imported at the entry points so numerical fitting and
policy execution remain independent of run directories and cache identity.
"""

import hashlib
import json
from pathlib import Path

import torch

from .data import load_bank
from .io import atomic_path, stage_hardlink
from .pipeline import fit_groups, method_variants, temporal_method
from .retrieval import configure_search_precision


TARGET_VERSION = 2


def _target_key(config, method):
    method = temporal_method(method)
    target_method = {key: method[key] for key in (
        "device", "dtype", "state_dim", "action_dim", "history", "scaling", "g")}
    if "hankel_stride" in method:
        target_method["hankel_stride"] = method["hankel_stride"]
    target_method["spc"] = {"horizon": method["spc"]["horizon"]}
    target_method["residual"] = {
        "target_batch_size": method["residual"]["target_batch_size"],
        "lookup": method["residual"].get("lookup", "g_supports")}
    for name in ("target_retrieval_precision", "target_execution"):
        if name in method["residual"]:
            target_method["residual"][name] = method["residual"][name]
    value = json.dumps({"target_version": TARGET_VERSION,
                        "data": config["data"], "method": target_method},
                       sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _required_lookup(group):
    required = {"state_mean", "state_scale", "action_mean", "action_scale"}
    if group["method"]["g"].get("lookup", "past") != "past":
        required |= {"g_lookup_matrix", "g_lookup_scale"}
    if group["method"]["g"].get("lookup") == "lda":
        required.add("lda")
    return required


def _required_payload(group):
    required = _required_lookup(group) | {"fit"}
    if group["spec"]["fit_residuals"]:
        required |= {"omega", "phase", "W", "W_knn", "s2_gip", "s2_knn", "sigma"}
    if (group["spec"]["fit_residuals"]
            and group["method"]["residual"].get("action_source") == "retimed_first"):
        required.add("residual_action_source")
    return required


def _compatible_spec(spec, expected):
    """Accept the removed one-value legacy selector metadata."""
    if spec == expected:
        return True
    if (not isinstance(spec, dict)
            or spec.get("fit_version") != expected.get("fit_version")):
        return False
    method = spec.get("method", {})
    g, temporal = method.get("g", {}), method.get("temporal_neighbors", {})
    if (g.get("selector") != "legacy_full"
            or temporal.get("selector") != "legacy_temporal"):
        return False
    normalized = {**spec, "method": {
        **method,
        "g": {key: value for key, value in g.items() if key != "selector"},
        "temporal_neighbors": {
            key: value for key, value in temporal.items() if key != "selector"},
    }}
    return normalized == expected


def _load_fitted(path, group):
    if not Path(path).is_file():
        return None
    payload = torch.load(path, map_location="cpu")
    return (payload if _compatible_spec(payload.get("spec"), group["spec"])
            and _required_payload(group) <= payload.keys() else None)


def _selected_fit_groups(groups, requested):
    selected = set(requested.split(",")) if requested else None
    unknown = (selected or set()) - {group["name"] for group in groups}
    if unknown:
        raise ValueError(f"unknown configured fits: {sorted(unknown)}")
    return [group for group in groups
            if selected is None or group["name"] in selected]


def _lookup_path(run_dir, group):
    return Path(run_dir) / "fit_parts" / f"{group['key']}__lookup.pt"


def _load_lookup(run_dir, group):
    path = _lookup_path(run_dir, group)
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or not _required_lookup(group) <= payload.keys():
            raise ValueError("incomplete lookup payload")
        return payload
    except Exception as error:
        print(f"[fit] ignoring invalid staged lookup {path}: {error}", flush=True)
        return None


def prepare_lookups(config, run_dir, requested=None):
    """Fit each global lookup once before distributed residual-target shards."""
    from .bpc import Policy

    run_dir = Path(run_dir)
    (run_dir / "fit_parts").mkdir(parents=True, exist_ok=True)
    groups = [group for group in _selected_fit_groups(fit_groups(config), requested)
              if group["method"]["g"].get("lookup", "past") != "past"]
    trajectories = None
    for group in groups:
        if (_load_lookup(run_dir, group) is not None
                or _load_fitted(run_dir / "policies" / f"{group['key']}.pt", group) is not None):
            continue
        trajectories = trajectories or load_bank(run_dir / "bank.pt", config["data"])[0]
        policy = Policy(
            trajectories, config["data"], group["method"],
            fit_residuals=False)
        payload = {
            "g_lookup": policy.fit_g_lookup(),
            "g_lookup_matrix": policy.g_lookup_matrix.detach().cpu().contiguous(),
            "g_lookup_scale": policy.g_lookup_scale,
            **{name: getattr(policy, name).detach().cpu().contiguous() for name in (
                "state_mean", "state_scale", "action_mean", "action_scale")},
        }
        if policy.support_selector is not None:
            payload[policy.g_lookup] = policy.support_selector.payload
        with atomic_path(_lookup_path(run_dir, group)) as temporary:
            torch.save(payload, temporary)
        del policy
        torch.cuda.empty_cache()
    return run_dir / "fit_parts"


def fit_policy(config, run_dir, shard=None, requested=None):
    """Fit each distinct configured neighborhood once and stage immutable artifacts."""
    from .bpc import Policy

    run_dir, all_groups = Path(run_dir), fit_groups(config)
    groups = _selected_fit_groups(all_groups, requested)
    trajectories = None
    if shard:
        completed = set()
        for group in groups:
            if not group["spec"]["fit_residuals"]:
                continue
            key = _target_key(config, group["method"])
            if _load_fitted(run_dir / "policies" / f"{group['key']}.pt", group):
                continue
            if key in completed:
                continue
            trajectories = trajectories or load_bank(run_dir / "bank.pt", config["data"])[0]
            lookup = _load_lookup(run_dir, group)
            if (group["method"]["g"].get("lookup", "past") != "past"
                    and lookup is None):
                raise RuntimeError("prepare learned lookup before target shards")
            Policy(trajectories, config["data"], group["method"],
                   fitted=lookup, fit_residuals=True,
                   load_operators=False).fit_targets(
                       run_dir / "fit_parts" / key, shard)
            completed.add(key)
        return run_dir / "fit_parts"

    for group in groups:
        cache_name = config["experiment"].get(
            "fit_cache_name", config["experiment"]["name"])
        cache = Path(config["experiment"]["cache_dir"]) / f"{cache_name}__{group['key']}.pt"
        destination = run_dir / "policies" / f"{group['key']}.pt"
        payload = _load_fitted(destination, group)
        status, source = "run", destination
        if payload is None:
            payload = _load_fitted(cache, group)
            status, source = "cached", cache
        if payload is None:
            if group["method"]["g"].get("lookup", "past") != "past":
                prepare_lookups(config, run_dir, group["name"])
            trajectories = trajectories or load_bank(run_dir / "bank.pt", config["data"])[0]
            lookup = _load_lookup(run_dir, group)
            policy = Policy(
                trajectories, config["data"], group["method"],
                fitted=lookup, load_operators=False,
                fit_residuals=group["spec"]["fit_residuals"],
            ).fit(run_dir / "fit_parts" / _target_key(config, group["method"]))
            payload = policy.fitted_payload(group["spec"])
            with atomic_path(cache) as temporary:
                torch.save(payload, temporary)
            status, source = "fitted", cache
            del policy
            torch.cuda.empty_cache()
        stage_hardlink(source, destination)
        print(f"[fit] {status} {group['name']} -> {destination}", flush=True)

    variants = method_variants(config)
    reference = config["analysis"]["reference_method"]
    reference_fit = variants[reference]["fit"]
    reference_group = next(item for item in all_groups if item["name"] == reference_fit)
    reference_policy = run_dir / "policies" / f"{reference_group['key']}.pt"
    if reference_policy.is_file():
        stage_hardlink(reference_policy, run_dir / "policy.pt")
    return run_dir / "policies"


def apply_variant_runtime(policy, variant):
    """Apply the per-method options that are inference-time only.

    These must be re-applied for every method, because one fitted policy is
    shared by all variants in a fit group.
    """
    policy.configure_recovery(variant.get("recovery"))
    configure_search_precision(policy, variant.get("retrieval_precision", "fp64"))


def load_policy(config, run_dir, method=None, trajectories=None):
    from .bpc import Policy

    run_dir, variants, groups = Path(run_dir), method_variants(config), fit_groups(config)
    if method in variants:
        variant = variants[method]
    else:
        variant = next((item for item in variants.values()
                        if item["base"] == method), next(iter(variants.values())))
    group = next(item for item in groups if item["name"] == variant["fit"])
    path = run_dir / "policies" / f"{group['key']}.pt"
    if not path.is_file() and len(groups) == 1:
        path = run_dir / "policy.pt"
    payload = torch.load(path, map_location="cpu")
    legacy = (len(groups) == 1
              and payload.get("spec") == {"data": config["data"],
                                          "method": config["method"]})
    if (not _compatible_spec(payload.get("spec"), group["spec"]) and not legacy) or not (
            _required_payload(group) <= payload.keys()):
        raise ValueError(f"{path.name} was fitted for a different run config")
    if trajectories is None:
        trajectories, _ = load_bank(run_dir / "bank.pt", config["data"])
    base = variant["base"] if method is None or method in variants else method
    policy = Policy(
        trajectories, config["data"], group["method"], fitted=payload,
        fit_residuals="omega" in payload)
    policy.select_method(base)
    apply_variant_runtime(policy, variant)
    return policy


def fitted_payload(policy, spec):
    payload = {
        "spec": spec,
        **{name: getattr(policy, name) for name in (
            "state_mean", "state_scale", "action_mean", "action_scale")},
        "fit": policy.fit_stats,
    }
    if policy.fit_residuals:
        payload["s2_gip"] = policy.s2_gip
        payload.update({
            "omega": policy.omega,
            "phase": policy.phase,
            "W": policy.W_gip,
            "W_knn": policy.W_knn,
            "s2_knn": policy.s2_knn,
            "sigma": policy.sigma,
        })
    if policy.g_lookup != "past":
        payload["g_lookup_matrix"] = policy.g_lookup_matrix
        payload["g_lookup_scale"] = policy.g_lookup_scale
    if policy.support_selector is not None:
        payload[policy.g_lookup] = policy.support_selector.payload
    if getattr(policy, "residual_action_source", "native") != "native":
        payload["residual_action_source"] = policy.residual_action_source
    return {name: value.detach().cpu().contiguous() if torch.is_tensor(value) else value
            for name, value in payload.items()}


def load_fitted_payload(policy, payload):
    if policy.fit_residuals:
        policy.s2_gip = float(payload["s2_gip"])
        policy.omega = payload["omega"].to(policy.device, dtype=policy.dtype)
        policy.phase = payload["phase"].to(policy.device, dtype=policy.dtype)
        policy._factor_rff()
        policy.W_gip = payload["W"].to(policy.device, dtype=policy.rff_dtype)
        policy.W_knn = payload["W_knn"].to(policy.device, dtype=policy.rff_dtype)
        policy.s2_knn = float(payload["s2_knn"])
        policy.sigma = float(payload["sigma"])
    if (policy.fit_residuals
            and policy.config.get("residual", {}).get("action_source", "native") == "retimed_first"
            and payload.get("residual_action_source") != "retimed_first"):
        raise ValueError("fitted retimed_first policy requires its matching residual action source payload")
    policy.set_residual_action_source(payload.get(
        "residual_action_source", getattr(policy, "residual_action_source", "native")))
    policy.fit_stats = payload["fit"]
    return policy
