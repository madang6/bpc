"""DexArt task catalog and seeded DP3 point-cloud episodes."""

from copy import deepcopy
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

UPSTREAM_REVISION = "47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97"
CAMERA = "instance_1"
IMAGE_SIZE = 84
PROPRIO_DIM = ACTION_DIM = 22
RESET_PROTOCOL = "seeded_instance_catalog"

# third_party/dexart-release/dexart/env/task_setting.py at UPSTREAM_REVISION.
TASK_SPECS = {
    "faucet": {
        "seen_ids": [148, 693, 822, 857, 991, 1011, 1053, 1288, 1343, 1370, 1466],
        "unseen_ids": [1556, 1633, 1646, 1667, 1741, 1832, 1925],
        "horizon": 250, "observation_dim": 32, "rand_pos": 0.1, "rand_degree": 90,
    },
    "bucket": {
        "seen_ids": [100431, 100435, 100438, 100439, 100441, 100444, 100446, 100448, 100454, 100461, 100462],
        "unseen_ids": [100468, 100470, 100473, 100482, 100484, 100486, 102352, 102358],
        "horizon": 100, "observation_dim": 33, "rand_pos": 0.05, "rand_degree": 0,
    },
    "laptop": {
        "seen_ids": [11395, 11405, 11406, 11477, 11581, 11586, 9996, 10090, 10098, 10101, 10125],
        "unseen_ids": [9748, 9912, 9918, 9960, 9968, 9992],
        "horizon": 250, "observation_dim": 32, "rand_pos": 0.1, "rand_degree": 60,
    },
    "toilet": {
        "seen_ids": [102677, 102687, 102689, 102692, 102697, 102699, 102701, 102703, 102707, 102708, 103234, 102663, 102666, 102667, 102669, 102670, 102675],
        "unseen_ids": [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102654, 102658],
        "horizon": 250, "observation_dim": 32, "rand_pos": 0.2, "rand_degree": 45,
    },
}


def task_spec(name: str) -> dict:
    if name not in TASK_SPECS:
        raise ValueError(f"unknown DEXArt task {name!r}; expected {tuple(TASK_SPECS)}")
    return {"name": name, "language": name, "action_dim": ACTION_DIM,
            **deepcopy(TASK_SPECS[name])}


def make_environment(data: dict, split: str = "seen", instance_id: int | None = None) -> Any:
    """Construct the upstream environment; optional simulation dependencies stay lazy."""
    spec = task_spec(data["dexart_task"])
    if split not in {"seen", "unseen"}:
        raise ValueError(f"unknown DEXArt instance split {split!r}")
    indices = spec[f"{split}_ids"]
    if instance_id is not None:
        if instance_id not in indices:
            raise ValueError(f"DEXArt {spec['name']} instance {instance_id} is not in {split}: {indices}")
        indices = [instance_id]
    if data["cameras"] != ["point_cloud"] or int(data["image_size"]) != IMAGE_SIZE:
        raise ValueError(f"DEXArt requires point_cloud input and image_size {IMAGE_SIZE}, got {data['cameras']}, {data['image_size']}")
    root = Path(data["dexart_root"]).resolve()
    if not (root / "dexart/env/create_env.py").is_file():
        raise FileNotFoundError(f"DEXArt source missing under {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from dexart.env.create_env import create_env
    if not Path(sys.modules[create_env.__module__].__file__).resolve().is_relative_to(root):
        raise ValueError(f"DEXArt imported from outside configured source {root}")
    from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG

    if (TRAIN_CONFIG[spec["name"]][split] != spec[f"{split}_ids"]
            or RANDOM_CONFIG[spec["name"]] != {key: spec[key] for key in ("rand_pos", "rand_degree")}):
        raise ValueError(f"installed DEXArt task settings differ from pinned revision {UPSTREAM_REVISION}")
    env = create_env(task_name=spec["name"], use_visual_obs=True, use_gui=False,
                     is_eval=True, pc_noise=True, pc_seg=True, index=indices,
                     img_type="robot", rand_pos=spec["rand_pos"],
                     rand_degree=spec["rand_degree"], frame_skip=10)
    if (env.robot.dof != ACTION_DIM or env.action_space.shape != (ACTION_DIM,)
            or env.horizon != spec["horizon"]):
        raise ValueError(f"DEXArt runtime dimensions/horizon differ: dof={env.robot.dof}, action={env.action_space.shape}, horizon={env.horizon}; spec={spec}")
    return env


def reset_environment(env: Any, seed: int) -> dict:
    # Upstream reset(seed=...) ignores seed. Reset uses Python random, global NumPy,
    # and env.np_random, so all three must receive the recorded seed explicitly.
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    return env.reset()


def task(config: dict) -> dict:
    evaluation = config["evaluation"]
    name = evaluation["task"]
    spec = task_spec(name)
    if evaluation["reset_protocol"] != RESET_PROTOCOL:
        raise ValueError(f"DEXArt requires {RESET_PROTOCOL}, got {evaluation['reset_protocol']}")
    split = evaluation.get("instance_split", "seen")
    if split not in {"seen", "unseen"}:
        raise ValueError(f"unknown DEXArt instance split {split!r}")
    ids = spec[f"{split}_ids"]
    indices = range(int(evaluation["init_index_start"]), int(evaluation["init_index_stop"]))
    return {"name": name, "language": spec["language"], "init_source": RESET_PROTOCOL,
            "initializers": {index: {"seed": int(evaluation["episode_seed"]) + index,
                                     "instance_id": ids[index % len(ids)]}
                             for index in indices}}


class Episode:
    """Serial native episodes with benchmark success separate from termination."""

    def __init__(self, config: dict) -> None:
        self.data = config["data"]
        self.evaluation = config["evaluation"]
        self.spec = task_spec(self.data["dexart_task"])
        if self.evaluation["task"] != self.spec["name"]:
            raise ValueError(f"DEXArt data/evaluation tasks disagree: {self.spec['name']}, {self.evaluation['task']}")
        self.split = self.evaluation.get("instance_split", "seen")
        if self.data.get("dexart_split", "seen") != self.split:
            raise ValueError(f"DEXArt data/evaluation splits disagree: {self.data.get('dexart_split', 'seen')}, {self.split}")
        self.state_dim, self.action_dim = self.spec["observation_dim"], ACTION_DIM
        self.env = None
        self.seed = None
        self.steps = 0

    def reset(self, initializer: dict) -> tuple[np.ndarray, dict]:
        self.close()
        self.seed = int(initializer["seed"])
        self.instance_id = int(initializer["instance_id"])
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.steps = 0
        self.final_success = False
        self.terminated = self.truncated = False
        from . import dp3_dexart
        self.env = dp3_dexart.make_environment(self.data, self.split, self.instance_id)
        observation = reset_environment(self.env, self.seed)
        if int(self.env.index) != self.instance_id:
            raise ValueError(f"DEXArt requested instance {self.instance_id}, reset loaded {self.env.index}")
        return self._observation(observation)

    def _observation(self, observation: dict) -> tuple[np.ndarray, dict]:
        from . import dp3_dexart
        return dp3_dexart.policy_observation(observation, self.spec['name'])

    def step(self, action: np.ndarray) -> tuple:
        if self.env is None or self.terminated or self.truncated:
            raise RuntimeError(f"cannot step DEXArt episode: seed={self.seed}, steps={self.steps}")
        applied = np.asarray(action, dtype=np.float64)
        if applied.shape != (ACTION_DIM,) or not np.isfinite(applied).all():
            raise ValueError(f"invalid DEXArt normalized action: {applied}")
        applied = np.clip(applied, -1, 1)
        observation, reward, done, info = self.env.step(applied)
        if not np.isfinite(reward):
            raise ValueError(f"nonfinite DEXArt reward {reward} at seed={self.seed}, step={self.steps + 1}")
        self.steps += 1
        self.final_success = bool(self.env.is_eval_done)
        # Upstream TimeLimit.truncated is false even at its cap because is_done()
        # includes the cap. Use the native early_done predicate to disambiguate.
        self.terminated = bool(done and self.env.early_done)
        self.truncated = bool(self.steps >= self.spec["horizon"] and not self.terminated)
        if bool(done) != (self.terminated or self.truncated):
            raise ValueError(f"unexpected DEXArt done={done}, early_done={self.env.early_done}, step={self.steps}")
        proprio, images = self._observation(observation)
        return proprio, images, float(reward), bool(done), {
            **info, "success": self.final_success, "applied_action": applied,
            "terminated": self.terminated, "truncated": self.truncated,
        }

    def close(self) -> None:
        if self.env is None:
            return
        env, self.env = self.env, None
        env.close()
