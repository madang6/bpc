"""Task-specific OpenPI LoRA fine-tuning on exact RoboSuite replays."""

import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .data import (OPENPI_REVISION, PI05_CHECKPOINT_FINGERPRINT,
                   PI05_TOKENIZER_SHA256, _pi05_checkpoint_fingerprint,
                   _sha256, make_episode)
from .io import write_json
from .robosuite import pi05_state


FORMAT = "robosuite_pi05_lora_v1"
MODEL_NAME = "pi05_robosuite_lora"
ACTION_HORIZON = 10
ACTION_LAYOUT = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def pi05_frames(images):
    """Map shared upright renders to OpenPI's LIBERO camera convention."""
    return np.stack([images["agentview_rgb"],
                     images["eye_in_hand_rgb"]])[:, :, ::-1]


def _configure_image_keys(image_keys):
    """Select the configured camera subset in pinned OpenPI preprocessing."""
    from openpi.models import model

    original = getattr(model, "_bpc_preprocess_observation", model.preprocess_observation)
    model._bpc_preprocess_observation = original
    keys = tuple(model.IMAGE_KEYS if image_keys is None else image_keys)
    if not keys or not set(keys).issubset(model.IMAGE_KEYS):
        raise ValueError(f"invalid OpenPI image keys: {keys}")
    model.preprocess_observation = functools.partial(original, image_keys=keys)


def _experiment(run_dir):
    path = Path(run_dir).resolve()
    for root in (path, *path.parents):
        if (root / "config.json").is_file():
            return root
    raise ValueError(f"{path} is not inside an experiment run")


def _repo_id(config):
    return config["evaluation"]["task"]


def _dataset(config, run_dir):
    home = _experiment(run_dir) / "lerobot"
    os.environ["HF_LEROBOT_HOME"] = str(home)
    return home / _repo_id(config)


def _cache_tokenizer(config):
    source = Path(config["data"]["pi05"]["tokenizer"]).resolve()
    if _sha256(source) != PI05_TOKENIZER_SHA256:
        raise ValueError("PaliGemma tokenizer digest mismatch")
    home = os.environ.get("OPENPI_DATA_HOME") or config["runtime"].get("openpi_data_home")
    if not home:
        raise ValueError("runtime.openpi_data_home is required")
    os.environ["OPENPI_DATA_HOME"] = home
    target = Path(home) / "big_vision" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    if not target.exists():
        shutil.copyfile(source, target)
    if _sha256(target) != PI05_TOKENIZER_SHA256:
        raise ValueError("OpenPI cache contains the wrong PaliGemma tokenizer")


def _validate_sources(config):
    spec = config["data"]["pi05"]
    if (_pi05_checkpoint_fingerprint(spec["checkpoint"])
            != PI05_CHECKPOINT_FINGERPRINT):
        raise ValueError("pi0.5 base checkpoint fingerprint mismatch")
    root = Path(config["runtime"]["openpi_root"])
    revision = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if revision != OPENPI_REVISION:
        raise ValueError(f"OpenPI revision mismatch: {revision}")


def _identity(config, run_dir):
    source = Path(config["data"]["raw_hdf5"])
    return {
        "format": FORMAT, "task": _repo_id(config),
        "config_sha256": _sha256(_experiment(run_dir) / "config.json"),
        "source_bytes": source.stat().st_size,
    }


def _read_data(config, run_dir):
    path = Path(run_dir) / "data.json"
    if not path.is_file():
        raise RuntimeError("run the data stage before pi0.5 LoRA fitting")
    payload = json.loads(path.read_text())
    if any(payload.get(key) != value for key, value in _identity(config, run_dir).items()):
        raise ValueError("unexpected pi0.5 LoRA dataset manifest")
    norm = Path(payload["norm_stats"])
    if not norm.is_file() or _sha256(norm) != payload.get("norm_stats_sha256"):
        raise ValueError("pi0.5 normalization statistics do not match the dataset")
    return payload


def _script(config, name):
    path = Path(config["runtime"]["openpi_root"]) / "scripts" / name
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _features():
    image = {"dtype": "image", "shape": (224, 224, 3),
             "names": ["height", "width", "channel"]}
    return {"image": image, "wrist_image": image.copy(),
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }


def prepare(config, run_dir):
    """Replay each recorded state once into the exact OpenPI LeRobot schema."""
    import h5py

    data, policy = config["data"], config["policy"]
    if (config["evaluation"]["benchmark"] != "robosuite"
            or data["cameras"] != ["agentview_rgb", "eye_in_hand_rgb"]
            or int(data["image_size"]) != 224
            or int(policy["action_horizon"]) != ACTION_HORIZON):
        raise ValueError("pi0.5 LoRA requires RoboSuite, two canonical 224px cameras, and H=10")
    source, count = Path(data["raw_hdf5"]), int(data["demonstrations"])
    root, repo_id = _dataset(config, run_dir), _repo_id(config)
    fit_config = _fitting_config(config, run_dir)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    with h5py.File(source, "r") as file:
        metadata = json.loads(file["data"].attrs["env_args"])
        demos = sorted(file["data"], key=lambda key: int(key.split("_")[-1]))
        kwargs = metadata["env_kwargs"]
        if (kwargs.get("control_freq") != 20
                or kwargs.get("controller_configs", {}).get("type") != "OSC_POSE"
                or len(demos) < count):
            raise ValueError("pi0.5 LoRA requires OSC_POSE at 20 Hz and the configured demos")
        dataset = (LeRobotDataset(repo_id, root=root) if root.exists() else
                   LeRobotDataset.create(repo_id, fps=20, root=root,
                                         robot_type="panda", features=_features()))
        start = int(dataset.meta.total_episodes)
        if start > count:
            raise ValueError("LeRobot cache has too many demonstrations")
        if start < count:
            dataset.start_image_writer(num_threads=8)
            environment = None
            try:
                environment = make_episode(config)
                for index in range(start, count):
                    demo = file["data"][demos[index]]
                    actions, states = np.asarray(demo["actions"]), np.asarray(demo["states"])
                    recorded = pi05_state(demo["obs"])
                    if (len(states) != len(actions) or len(recorded) != len(actions)
                            or actions.shape[1:] != (7,)
                            or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
                        raise ValueError(f"invalid demonstration {demos[index]}")
                    environment.replay_demo(index, demo)
                    for step, (state, action) in enumerate(zip(states, actions)):
                        _, images = environment.replay_state(state)
                        frames = np.ascontiguousarray(pi05_frames(images))
                        state8 = environment.pi05_state.copy()
                        if (frames.shape != (2, 224, 224, 3) or frames.dtype != np.uint8
                                or state8.shape != (8,) or not np.isfinite(state8).all()
                                or not np.allclose(state8, recorded[step], atol=1e-5, rtol=0)):
                            raise ValueError(f"invalid replay convention in {demos[index]} step {step}")
                        dataset.add_frame({
                            "image": frames[0], "wrist_image": frames[1],
                            "state": state8,
                            "actions": action.astype(np.float32),
                            "task": data["prompt"],
                        })
                    dataset.save_episode()
                    print(f"[data] pi0.5 LeRobot demonstration {index + 1}/{count}", flush=True)
            finally:
                dataset.stop_image_writer()
                if environment is not None:
                    environment.close()
    if dataset.meta.total_episodes != count:
        raise RuntimeError("incomplete pi0.5 LeRobot dataset")
    norm_stats = fit_config.assets_dirs / repo_id / "norm_stats.json"
    if not norm_stats.is_file():
        normalizer = _script(config, "compute_norm_stats.py")
        normalizer._config.get_config = lambda _: fit_config
        normalizer.main(MODEL_NAME)
    manifest = {**_identity(config, run_dir), "source": str(source.resolve()),
                "dataset": str(root), "frames": int(dataset.meta.total_frames),
                "norm_stats": str(norm_stats), "norm_stats_sha256": _sha256(norm_stats)}
    write_json(Path(run_dir) / "data.json", manifest)
    return root


def _fitting_config(config, run_dir):
    _dataset(config, run_dir)
    _validate_sources(config)
    _cache_tokenizer(config)
    _configure_image_keys(config["policy"].get("image_keys"))
    from openpi.models import pi0_config
    from openpi.training import config as fitting, optimizer, weight_loaders

    spec = config["policy"]["fitting"]
    model = pi0_config.Pi0Config(
        pi05=True, action_horizon=ACTION_HORIZON, discrete_state_input=True,
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    checkpoint_dir = Path(run_dir) / "checkpoints" / MODEL_NAME / _repo_id(config)
    resume = checkpoint_dir.is_dir() and any(
        path.name.isdigit() for path in checkpoint_dir.iterdir())
    return fitting.TrainConfig(
        name=MODEL_NAME, exp_name=_repo_id(config),
        project_name=spec["wandb_project"], model=model,
        data=fitting.LeRobotLiberoDataConfig(
            repo_id=_repo_id(config),
            base_config=fitting.DataConfig(prompt_from_task=True),
            extra_delta_transform=False),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(Path(config["data"]["pi05"]["checkpoint"]) / "params")),
        freeze_filter=model.get_freeze_filter(), ema_decay=None,
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=10_000, peak_lr=5e-5,
            decay_steps=1_000_000, decay_lr=5e-5),
        batch_size=int(spec["batch_size"]), num_workers=int(spec["workers"]),
        num_train_steps=int(spec["steps"]), seed=int(spec["seed"]),
        log_interval=int(spec.get("log_interval", 100)),
        save_interval=int(spec["save_interval"]), keep_period=None,
        assets_base_dir=str(Path(run_dir) / "openpi_assets"),
        checkpoint_base_dir=str(Path(run_dir) / "checkpoints"),
        resume=resume, overwrite=checkpoint_dir.exists() and not resume,
        wandb_enabled=True)


def _checkpoint(config, fit_config, data):
    output = fit_config.checkpoint_dir / str(fit_config.num_train_steps - 1)
    stats = output / "assets" / _repo_id(config) / "norm_stats.json"
    if (not (output / "_CHECKPOINT_METADATA").is_file()
            or not (output / "params").is_dir() or not stats.is_file()
            or _sha256(stats) != data["norm_stats_sha256"]):
        raise ValueError("invalid pi0.5 LoRA checkpoint")
    return output


def fit(config, run_dir):
    """Run OpenPI's unmodified JAX runner with its canonical LoRA filter."""
    data = _read_data(config, run_dir)
    fit_config = _fitting_config(config, run_dir)
    if (fit_config.checkpoint_dir / str(fit_config.num_train_steps - 1)
            / "_CHECKPOINT_METADATA").is_file():
        return _checkpoint(config, fit_config, data)
    if not (fit_config.assets_dirs / _repo_id(config) / "norm_stats.json").is_file():
        raise RuntimeError("pi0.5 normalization statistics are missing")
    runner = _script(config, "train.py")
    wandb_image, wandb_init = runner.wandb.Image, runner.init_wandb
    wandb_log = None

    def init_wandb(*args, **kwargs):
        nonlocal wandb_log
        result = wandb_init(*args, **kwargs)
        wandb_log = runner.wandb.log
        runner.wandb.Image = lambda _: None
        runner.wandb.log = lambda values, *args, **kwargs: (
            None if "camera_views" in values else wandb_log(
                {f"fit/{key}": value for key, value in values.items()},
                *args, **kwargs))
        return result

    runner.init_wandb = init_wandb
    try:
        runner.main(fit_config)
    finally:
        runner.init_wandb, runner.wandb.Image = wandb_init, wandb_image
        if wandb_log is not None:
            runner.wandb.log = wandb_log
        runner.wandb.finish()
    return _checkpoint(config, fit_config, data)


class FineTunedPi05:
    """BPC rollout adapter around OpenPI's official fitted-policy transforms."""

    def __init__(self, config, run_dir):
        from openpi.policies import policy_config

        data = _read_data(config, run_dir)
        fit_config = _fitting_config(config, run_dir)
        self.policy = policy_config.create_trained_policy(
            fit_config, _checkpoint(config, fit_config, data),
            sample_kwargs={"num_steps": int(config["policy"]["flow_steps"])})
        self._configure(config, ACTION_HORIZON, config["policy"]["replan_steps"])

    def _configure(self, config, action_horizon, replan_steps):
        self.action_horizon, self.action_dim = int(action_horizon), len(ACTION_LAYOUT)
        self.replan_steps, self.prompt = int(replan_steps), config["data"]["prompt"]
        if not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError("policy.replan_steps must be within the action horizon")

    def _actions(self, output):
        actions = np.asarray(output)
        if actions.shape != (self.action_horizon, self.action_dim) or not np.isfinite(actions).all():
            raise RuntimeError(f"pi0.5 produced invalid actions {actions.shape}")
        return actions[:self.replan_steps]

    def reset(self, episode_seed):
        self.episode_seed = int(episode_seed)
        self.query = 0

    def _noise(self):
        generator = np.random.default_rng(
            np.random.SeedSequence([self.episode_seed, self.query]))
        noise = generator.standard_normal(
            (self.action_horizon, 32), dtype=np.float32)
        self.query += 1
        return noise

    def plan(self, state, images):
        frames = np.ascontiguousarray(pi05_frames(images))
        state = np.asarray(state, dtype=np.float32)
        if frames.shape != (2, 224, 224, 3) or state.shape != (8,):
            raise ValueError("pi0.5 expects two 224px cameras and an 8-D state")
        output = self.policy.infer({
            "observation/image": frames[0],
            "observation/wrist_image": frames[1],
            "observation/state": state,
            "prompt": self.prompt,
        }, noise=self._noise())
        return self._actions(output["actions"]), output["policy_timing"]["infer_ms"]
