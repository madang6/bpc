"""Generate or reuse an action-replayed visuomotor behavior-cloning bank."""

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .io import atomic_path, sha256 as _sha256, stage_symlink, write_json


OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
PI05_CHECKPOINT_FINGERPRINT = "b96af7ad0456c0800edb0a6c855f5912b511b4006e0802e7be4dbf2964a64f72"
PI05_TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
GENERATED_BANK_SUFFIXES = ("action_replay_v1", "recorded_state_rerender_v1")


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluation_seed(config, index):
    evaluation = config["evaluation"]
    return int(evaluation["episode_seed"]) + (
        0 if evaluation.get("reset_protocol") == "serial_seed_catalog" else index)


def select_shard(indices, shard):
    indices = list(indices)
    if not shard:
        return indices
    owned, workers = shard.split("/")
    ranks = {int(rank) for rank in owned.split(",")}
    if not ranks or min(ranks) < 0 or max(ranks) >= int(workers):
        raise ValueError("shard must be K[,K...]/N with 0 <= K < N")
    return [item for index, item in enumerate(indices) if index % int(workers) in ranks]


def _benchmark(config):
    kind = config["evaluation"].get("benchmark")
    if kind == "adroit":
        from . import adroit
        return adroit
    if kind == "dexart":
        from . import dexart
        return dexart
    if kind != "robosuite":
        raise ValueError(f"unsupported benchmark {kind!r}; expected robosuite")
    from . import robosuite
    return robosuite


def benchmark_task(config):
    return _benchmark(config).task(config)


def make_episode(config):
    return _benchmark(config).Episode(config)


def demonstration_initializers(config, indices):
    return _benchmark(config).demonstration_initializers(config["data"]["raw_hdf5"], indices)


def encoder_spec(data):
    name = data.get("encoder", "r3m")
    if name == "dp3_pointnet":
        return name, {"feature_dim": 64, **data[name]}
    if name != "r3m":
        raise ValueError(f"unsupported visual encoder {name!r}; expected r3m")
    return name, data["r3m"]


def encoder_identity(data):
    name, spec = encoder_spec(data)
    if name == "dp3_pointnet":
        return {"name": name, "feature_dim": 64,
                "checkpoint_sha256": spec["checkpoint_sha256"]}
    return {"name": name, **{key: spec[key] for key in
            ("model", "feature_dim", "resize")}}


def full_state(proprio, images, data, encode):
    if not data["feature_keys"]:
        return proprio.astype(np.float64)
    frames = np.stack([images[camera] for camera in data["cameras"]])
    features = encode(frames)
    return np.concatenate([proprio, *features]).astype(np.float64)


def load_bank(path, data=None):
    payload = torch.load(path, map_location="cpu")
    trajectories = payload.get("data", payload.get("trajectories"))
    metadata = payload.get("metadata", payload.get("meta", {}))
    if not isinstance(trajectories, list):
        raise ValueError(f"{path} does not contain a trajectory list")
    for configured, saved in (("render_convention", "render_convention"),
                              ("construction", "trajectory_construction")):
        if data and (required := data.get(configured)) and metadata.get(saved) != required:
            raise ValueError(
                f"bank {saved}={metadata.get(saved)!r} but config requires {required!r}; "
                f"regenerate the bank under a new cache path")
    return trajectories, metadata


def _bank_shape(trajectories, config):
    state_key, action_key = config["state_key"], config["action_key"]
    state_dim = action_dim = samples = None
    for index, trajectory in enumerate(trajectories):
        blocks = [np.asarray(trajectory[state_key])]
        blocks.extend(np.asarray(trajectory[key]) for key in config["feature_keys"])
        action = np.asarray(trajectory[action_key])
        lengths = {block.shape[1] for block in [*blocks, action]}
        if any(block.ndim != 2 for block in [*blocks, action]) or len(lengths) != 1:
            raise ValueError(f"trajectory {index} has inconsistent 2-D arrays")
        width = sum(block.shape[0] for block in blocks)
        if state_dim is None:
            state_dim, action_dim, samples = width, action.shape[0], 0
        if (width, action.shape[0]) != (state_dim, action_dim):
            raise ValueError(f"trajectory {index} changes state/action width")
        samples += action.shape[1]
    return {"demonstrations": len(trajectories), "samples": samples,
            "state_dim": state_dim, "action_dim": action_dim}


def validate_bank_shape(trajectories, config):
    """Validate the shared encoded-bank layout against a resolved task recipe."""
    data = config["data"]
    shape = _bank_shape(trajectories, data)
    expected = {"demonstrations": data["demonstrations"],
                "state_dim": config["method"]["state_dim"],
                "action_dim": config["method"]["action_dim"]}
    for key, value in expected.items():
        if shape[key] != value:
            raise ValueError(f"bank {key}={shape[key]} but config requires {value}")
    return shape


def make_r3m_encoder(model_name, device, resize, batch_size, *, evaluation=False):
    from r3m import load_r3m
    from .fast_r3m import R3MEncoder

    model = load_r3m(model_name).to(device)
    network = nn.Sequential(*model.module.convnet.children())
    return R3MEncoder(network, resize, batch_size, graphs=evaluation)


def _pi05_checkpoint_fingerprint(path):
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for item in sorted(item for item in root.rglob("*") if item.is_file()):
        size = item.stat().st_size
        digest.update(str(item.relative_to(root)).encode())
        digest.update(b"\0" + str(size).encode() + b"\0")
        if size < 1_000_000:
            digest.update(_sha256(item).encode())
    return digest.hexdigest()


def make_encoder(data, *, evaluation=False):
    if not data["feature_keys"]:
        return None
    name, spec = encoder_spec(data)
    if name == "dp3_pointnet":
        from .dexterous import PointEncoder
        return PointEncoder(data)
    batch_size = int(spec["batch_size"])
    if evaluation:
        batch_size = min(batch_size, len(data["cameras"]))
    return make_r3m_encoder(spec["model"], spec["device"], spec["resize"],
                            batch_size, evaluation=evaluation)


def _encode_demonstrations(config, run_dir, shard):
    import h5py

    data = config["data"]
    parts = Path(run_dir) / "bank_parts"
    indices = select_shard(range(int(data["demonstrations"])), shard)
    missing = [index for index in indices if not (parts / f"{index:03d}.pt").is_file()]
    if not missing:
        return
    seed(int(data["realization_seed"]))
    environment = make_episode(config)
    encode = make_encoder(data)
    with h5py.File(data["raw_hdf5"], "r") as source:
        demos = sorted(source["data"], key=lambda key: int(key.split("_")[1]))
        try:
            for demo_index in missing:
                key, demo = demos[demo_index], source["data"][demos[demo_index]]
                actions = np.asarray(demo["actions"])
                if (actions.ndim != 2 or actions.shape[1] != config["method"]["action_dim"]
                        or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
                    raise ValueError(f"{key} contains invalid unclipped actions")
                construction = data["construction"]
                if construction.endswith("recorded_state_rerender_v1"):
                    states = np.asarray(demo["states"])
                    if len(states) < len(actions):
                        raise ValueError(
                            f"{key} has {len(states)} states for {len(actions)} actions")
                    environment.replay_demo(demo_index, demo)
                    proprio = environment.recorded_proprio(demo)[:len(actions)]
                    frames = {camera: [] for camera in data["cameras"]}
                    for state in states[:len(actions)]:
                        _, images = environment.replay_state(state)
                        for camera in data["cameras"]:
                            frames[camera].append(images[camera])
                    features = [encode(np.asarray(frames[camera]))
                                for camera in data["cameras"]]
                    rows = np.concatenate([proprio, *features], axis=1)
                else:
                    proprio, images = environment.replay_demo(demo_index, demo)
                    rows = []
                    for action in actions:
                        rows.append(full_state(proprio, images, data, encode))
                        proprio, images, _, _, _ = environment.step(action)
                    rows = np.asarray(rows)
                dim = int(encoder_spec(data)[1]["feature_dim"]) if data["feature_keys"] else 0
                state_dim = rows.shape[1] - len(data["feature_keys"]) * dim
                trajectory = {
                    data["state_key"]: rows[:, :state_dim].T,
                    data["action_key"]: actions.T,
                    **{key: rows[:, state_dim + index * dim:
                                 state_dim + (index + 1) * dim]
                       .astype(np.float32).T
                       for index, key in enumerate(data["feature_keys"])},
                    "demo_id": demo_index,
                }
                with atomic_path(parts / f"{demo_index:03d}.pt") as temporary:
                    torch.save(trajectory, temporary)
                print(f"[data] {construction} demonstration {demo_index + 1}/"
                      f"{data['demonstrations']}", flush=True)
        finally:
            environment.close()


def _merge_parts(config, run_dir, output):
    data = config["data"]
    paths = [Path(run_dir) / "bank_parts" / f"{index:03d}.pt"
             for index in range(int(data["demonstrations"]))]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("incomplete bank parts: " + ", ".join(missing))
    object_dims = int(data.get("object_dims", 0))
    source = data.get("object_source", "object")
    metadata = {"state_convention": f"ee9+{source}{object_dims}" if object_dims else "ee9",
                "trajectory_construction": data["construction"],
                "encoder": encoder_identity(data) if data["feature_keys"] else None}
    metadata.update({key: data[key] for key in (
        "image_source", "image_size", "render_convention", "realization_seed")
        if key in data})
    payload = {
        "data": [torch.load(path, map_location="cpu") for path in paths],
        "metadata": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(output) as temporary:
        torch.save(payload, temporary)


def _representation_check(config, trajectories):
    data, evaluation = config["data"], config["evaluation"]
    rows = torch.from_numpy(np.concatenate([np.concatenate([
        np.asarray(trajectory[data["state_key"]]),
        *(np.asarray(trajectory[key]) for key in data["feature_keys"]),
    ], axis=0).T for trajectory in trajectories])).float()
    scale = rows.std(0, correction=0)
    floor = float(config["method"]["scaling"]["minimum_standard_deviation"])
    index = int(evaluation["init_index_start"])
    seed(evaluation_seed(config, index))
    task = benchmark_task(config)
    environment = make_episode(config)
    encode = make_encoder(data)
    try:
        proprio, images = environment.reset(task["initializers"][index])
        live = torch.from_numpy(full_state(proprio, images, data, encode)).float()
    finally:
        environment.close()
    z = (live - rows.mean(0)).abs() / scale.clamp_min(floor)
    floored = scale < floor
    return {"init_index": index, "floored_dimensions": int(floored.sum()),
            "max_floored_abs_z": float(z[floored].max()) if floored.any() else 0.0,
            "max_abs_z": float(z.max())}


def prepare(config, run_dir, shard=None):
    """Generate shard parts or expose one complete cached bank."""
    data = config["data"]
    if config["evaluation"].get("benchmark") in {"adroit", "dexart"}:
        from .dexterous import prepare_bank
        return prepare_bank(config, run_dir, shard)
    source = Path(data["bank"] if "bank" in data else data["reuse_bank"])
    if shard:
        if not source.is_file():
            _encode_demonstrations(config, run_dir, shard)
        return Path(run_dir) / "bank_parts"
    reused = source.is_file()
    if not reused:
        if not data.get("construction", "").endswith(GENERATED_BANK_SUFFIXES):
            raise FileNotFoundError(source)
        _encode_demonstrations(config, run_dir, None)
        _merge_parts(config, run_dir, source)

    if data.get("reset_catalog"):
        _benchmark(config).prepare_reset_catalog(config)

    trajectories, metadata = load_bank(source, data)
    shape = validate_bank_shape(trajectories, config)
    check = (_representation_check(config, trajectories)
             if data.get("construction", "").endswith(GENERATED_BANK_SUFFIXES) else {})
    return stage_bank(config, run_dir, source, shape, metadata, reused, check)


def stage_bank(config, run_dir, source, shape, metadata, reused, check=None, extra=None):
    """Publish one validated bank using the shared data-stage manifest contract."""
    data = config["data"]
    check = check or {}
    run_dir = Path(run_dir)
    write_json(run_dir / "data.json", {
        "bank": str(source.resolve()), "raw_hdf5": data["raw_hdf5"],
        **({"reset_catalog": data["reset_catalog"]}
           if data.get("reset_catalog") else {}),
        "reused_bank": reused, **shape,
        "state_convention": metadata.get("state_convention", "ee9"),
        "state_layout": [data["state_key"], *data["feature_keys"]],
        "action_layout": data["action_key"], "encoder": metadata.get("encoder"),
        "render_convention": metadata.get("render_convention"),
        "representation_check": check, **(extra or {}),
    })
    if check and check["max_floored_abs_z"] > 1:
        raise RuntimeError(f"native bank/live representation mismatch: {check}")
    stage_symlink(source, run_dir / "bank.pt")
    print(f"[data] {shape['demonstrations']} demos, {shape['samples']} samples -> {run_dir / 'bank.pt'}", flush=True)
    return run_dir / "bank.pt"
