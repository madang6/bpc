"""PointNet input contracts and bank preparation for the native suites."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from . import adroit, dexart, dp3_adroit, dp3_dexart
from .dp3_pointnet import load_frozen_encoder, sha256, validate_source
from .io import atomic_path

CONSTRUCTION = "dp3_pointcloud_recorded_v1"


def input_contract(data: dict) -> dict:
    return (dp3_adroit.metadata(data) if data["suite"] == "adroit"
            else dp3_dexart.contract(data["env_task"]))


def contract_digest(data: dict) -> str:
    return hashlib.sha256(json.dumps(input_contract(data), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def configure_task(config: dict, benchmark: dict, task: dict) -> None:
    """Resolve one suite task; explicit task data paths take precedence."""
    suite, name = benchmark["kind"], task["name"]
    data, method = config["data"], config["method"]
    root = Path(benchmark["dataset_dir"]) / name
    for key, value in {"raw_hdf5": str(root / "raw.hdf5"),
                       "pointcloud_hdf5": str(root / "capture.hdf5"),
                       "bank": str(root / "bank.pt")}.items():
        data.setdefault(key, value)
    data.update(suite=suite, env_task=name, construction=CONSTRUCTION,
                **{f"{suite}_task": name})
    data["dp3_pointnet"].setdefault("checkpoint", str(root / "pointnet.pt"))
    if suite == "adroit":
        proprio, actions = 24, adroit.task_spec(name)["action_dim"]
        horizon, reset = 200, "independent_seed"
    else:
        spec = dexart.task_spec(name)
        proprio, actions = spec["observation_dim"], spec["action_dim"]
        horizon, reset = spec["horizon"], dexart.RESET_PROTOCOL
    method.update(state_dim=proprio + 64, action_dim=actions)
    method["scaling"]["proprio_dim"] = proprio
    method["spc"]["dynamics_state_dims"] = proprio
    method["g"]["rate_scaled_action_dims"] = [] if suite == "adroit" else list(range(6))
    config["evaluation"].setdefault("reset_protocol", reset)
    config["evaluation"].setdefault("controller_steps", horizon - config["evaluation"]["warmup_steps"])


def validate_config(config: dict) -> None:
    data, evaluation = config["data"], config["evaluation"]
    if (data.get("encoder") != "dp3_pointnet" or data["feature_keys"] != ["dp3_pointnet"]
            or data["cameras"] != ["point_cloud"]):
        raise ValueError("native suites require one dp3_pointnet point_cloud input")
    if int(evaluation.get("env_workers", 1)) != 1 or evaluation.get("save_videos"):
        raise ValueError("native suites require serial evaluation without RGB video capture")
    if any(v.get("recovery", False) for v in config["methods"].values()):
        raise ValueError("native suites do not use the Panda gripper recovery controller")
    suite, name = evaluation["benchmark"], evaluation["task"]
    if data.get("suite") != suite or data.get("env_task") != name:
        raise ValueError(f"native data/evaluation task mismatch: {data.get('suite')}/{data.get('env_task')}, {suite}/{name}")
    input_contract(data)


class PointEncoder:
    """Load the frozen checkpoint once; use batch-one arithmetic offline and online."""

    def __init__(self, data: dict) -> None:
        spec = data["dp3_pointnet"]
        self.model = load_frozen_encoder(spec["checkpoint"], spec["checkpoint_sha256"], spec["device"])
        source = self.model.source_identity
        expected = {"suite": data["suite"], "task": data["env_task"],
                    "source_contract_sha256": contract_digest(data)}
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError(f"encoder source {source} does not match live input contract {expected}")

    def __call__(self, clouds: np.ndarray) -> np.ndarray:
        return np.concatenate([self.model(cloud[None]).cpu().numpy() for cloud in clouds])


def prepare_bank(config: dict, run_dir: Path, shard: str | None = None) -> Path:
    """Reuse a bound bank, or encode pre-action clouds from a captured HDF5 dataset."""
    from .data import encoder_identity, load_bank, stage_bank, validate_bank_shape

    validate_config(config)
    data, method = config["data"], config["method"]
    path = Path(data["bank"])
    if shard:
        raise ValueError("native bank preparation is serial; do not pass --shard")
    expected = {"suite": data["suite"], "task": data["env_task"],
                "source_contract_sha256": contract_digest(data),
                "checkpoint_sha256": data["dp3_pointnet"]["checkpoint_sha256"]}
    encoder = PointEncoder(data)
    capture_hash = encoder.model.source_identity["sha256"]
    reused = path.is_file()
    if not reused:
        capture = Path(data["pointcloud_hdf5"])
        if sha256(capture) != capture_hash:
            raise ValueError(f"capture {capture} differs from the encoder training source")
        architecture = {"state_dim": method["scaling"]["proprio_dim"],
                        "action_dim": method["action_dim"], "num_points": encoder.model.num_points}
        with h5py.File(capture, "r") as source:
            ids = validate_source(source, expected, architecture)
            if len(ids) != data["demonstrations"]:
                raise ValueError(f"capture has {len(ids)} demonstrations, expected {data['demonstrations']}")
            trajectories = []
            for name in ids:
                episode = source[name]
                actions = np.asarray(episode["actions"])
                states = np.asarray(episode["state"][:-1])
                features = encoder(np.asarray(episode["point_cloud"][:-1]))
                trajectories.append({data["state_key"]: states.T,
                                     data["action_key"]: actions.T,
                                     "dp3_pointnet": features.T})
        metadata = {**expected, "capture_sha256": capture_hash,
                    "trajectory_construction": CONSTRUCTION, "encoder": encoder_identity(data),
                    "input_contract": input_contract(data), "episode_ids": ids}
        with atomic_path(path) as temporary:
            torch.save({"data": trajectories, "metadata": metadata}, temporary)
    trajectories, metadata = load_bank(path, data)
    for key, value in {**expected, "capture_sha256": capture_hash}.items():
        if metadata.get(key) != value:
            raise ValueError(f"bank {path}: {key}={metadata.get(key)!r}, expected {value!r}")
    shape = validate_bank_shape(trajectories, config)
    for trajectory in trajectories:
        if any(not np.isfinite(trajectory[key]).all() for key in (data["state_key"], data["action_key"], "dp3_pointnet")):
            raise ValueError(f"bank {path} contains nonfinite values")
    return stage_bank(config, run_dir, path, shape, metadata, reused, extra={
        **expected, "bank_sha256": sha256(path), "capture_sha256": capture_hash,
        "state_convention": f"dp3_{data['suite']}_robot_state",
    })
