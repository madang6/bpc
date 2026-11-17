"""Translate saved artifact schemas at the input boundary.

Live configs and newly written checkpoints use BPC names exclusively. Historical
paths and tensors are left untouched; only their in-memory metadata is upgraded.
"""

from copy import deepcopy
import json
from pathlib import Path

from .io import read_config


_METHOD_NAMES = {
    "gip_direct": "bpc", "deepc_g_direct": "bpc_nnc",
    "gip_lda": "bpc_fisher", "gip_nhc": "bpc_nhc",
    "gip_nnc": "bpc_nnc", "gip_nlr": "bpc_nlr",
}
_RESIDUAL_FIELDS = {
    "W": "residual_weights", "W_knn": "auxiliary_residual_weights",
    "s2_gip": "residual_variance", "s2_knn": "auxiliary_residual_variance",
}
_FITTING_FIELDS = {
    "training": "fitting", "trained_steps": "fitted_steps",
    "training_purpose": "fitting_purpose", "training_admissible": "fitting_admissible",
    "training_episode_ids": "fitting_episode_ids",
    "training_relative_error": "fitting_relative_error", "training_r2": "fitting_r2",
}


def _rename_keys(value, names):
    result = {}
    for key, item in value.items():
        name = names.get(key, key)
        if name in result:
            raise ValueError(f"ambiguous artifact field: {name}")
        result[name] = item
    return result


def _fitting_metadata(value):
    value = _rename_keys(value, _FITTING_FIELDS)
    for field in ("config", "normalizers", "policy", "fit", "g_lookup"):
        if isinstance(value.get(field), dict):
            value[field] = _fitting_metadata(value[field])
    if value.get("point_scheme") == "train_limits":
        value["point_scheme"] = "fit_limits"
    return value


def read_capture_metadata(source):
    """Accept archived HDF5 attributes without rewriting the capture or its hash."""
    return _fitting_metadata(dict(source.attrs))


def _method_config(method):
    method = deepcopy(method)
    if "spc" in method:
        continuation = method.pop("spc")
        extra = set(continuation) - {
            "horizon", "ridge", "tracking_weight", "prior_weight", "dynamics_state_dims"}
        if extra:
            raise ValueError(f"unknown archived continuation settings: {sorted(extra)}")
        if "horizon" in continuation:
            if "continuation_horizon" in method:
                raise ValueError("ambiguous artifact field: continuation_horizon")
            method["continuation_horizon"] = continuation["horizon"]
    g, temporal = method.get("g", {}), method.get("temporal_neighbors", {})
    if (g.get("selector") == "legacy_full"
            and temporal.get("selector") == "legacy_temporal"):
        g.pop("selector")
        temporal.pop("selector")
    return method


def normalize_spec(spec):
    if not isinstance(spec, dict):
        return spec
    spec = deepcopy(spec)
    if isinstance(spec.get("method"), dict):
        spec["method"] = _method_config(spec["method"])
    if spec.get("fit_version") == 3 and spec.get("fit_posterior") is False:
        spec.pop("fit_posterior")
        spec["fit_version"] = 4
    return spec


def normalize_payload(payload):
    """Copy metadata without copying potentially large checkpoint tensors."""
    payload = _fitting_metadata(_rename_keys(payload, _RESIDUAL_FIELDS))
    if payload.get("format") == "our_dp3_pointnet_bc_v1":
        payload["format"] = "our_dp3_pointnet_bc_v2"
    if "spec" in payload:
        payload["spec"] = normalize_spec(payload["spec"])
    if "fit" in payload:
        payload["fit"] = _rename_keys(payload["fit"], {
            "gip_residual_variance": "residual_variance",
            "knn_residual_variance": "auxiliary_residual_variance"})
    return payload


def read_payload(path, **load_options):
    import torch
    return normalize_payload(torch.load(path, map_location="cpu", **load_options))


def compatible_spec(spec, expected):
    return normalize_spec(spec) == expected


def load_policy_payload(config, run_dir, group, required, *, single):
    run_dir = Path(run_dir)
    current = run_dir / "policies" / f"{group['key']}.pt"
    # Schema upgrades change the content hash, but not the fit group name.
    candidates = ([current] if current.is_file() else
                  sorted(current.parent.glob(f"{group['name']}__*.pt"))
                  + ([run_dir / "policy.pt"] if single else []))
    for path in candidates:
        if not path.is_file():
            continue
        payload = read_payload(path)
        spec = payload.get("spec")
        unversioned = (single and spec == {"data": config["data"], "method": config["method"]})
        if (compatible_spec(spec, group["spec"]) or unversioned) and required <= payload.keys():
            return payload
    raise ValueError(f"{run_dir}: policy is missing or was fitted for a different run config")


def load_run_config(path):
    """Read a recorded run, including benchmark-specific config overrides."""
    def upgrade(config):
        if isinstance(config, list):
            return [upgrade(item) for item in config]
        if not isinstance(config, dict):
            return config
        config = _rename_keys({key: upgrade(value) for key, value in config.items()}, _FITTING_FIELDS)
        if isinstance(config.get("method"), dict):
            config["method"] = _method_config(config["method"])
        if "methods" in config:
            methods = config["methods"]
            if isinstance(methods, list):
                config["methods"] = [_METHOD_NAMES.get(name, name) for name in methods]
            else:
                methods = _rename_keys(methods, _METHOD_NAMES)
                for variant in methods.values():
                    variant["base"] = _METHOD_NAMES.get(variant["base"], variant["base"])
                config["methods"] = methods
        for field in ("reference_method", "method_base"):
            if field in config:
                name = config[field]
                config[field] = _METHOD_NAMES.get(name, name)
        runtime = config.get("runtime", {})
        if runtime.get("image", "").startswith("dpc-nldl-"):
            runtime["image"] = "bpc-" + runtime["image"][len("dpc-nldl-"):]
        return config

    return upgrade(read_config(path))


def episode_path(path):
    """Find recorded episodes under either the current or archived method name."""
    path = Path(path)
    if not path.is_file():
        for old, new in _METHOD_NAMES.items():
            if path.parent.name == new:
                archived = path.parent.with_name(old) / path.name
                if archived.is_file():
                    return archived
    return path


def read_episode(path):
    row = json.loads(episode_path(path).read_text())
    name = row.get("method")
    if name is not None:
        row["method"] = _METHOD_NAMES.get(name, name)
    return row
