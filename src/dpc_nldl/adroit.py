"""Gymnasium Adroit with robot-only proprioception and DP3 depth clouds."""

import json
from pathlib import Path

import numpy as np

from . import dp3_adroit

TASKS = {
    "door": ("AdroitHandDoor-v1", 39, 28),
    "hammer": ("AdroitHandHammer-v1", 46, 26),
    "pen": ("AdroitHandPen-v1", 45, 24),
    "relocate": ("AdroitHandRelocate-v1", 39, 30),
}


def task_spec(name: str) -> dict:
    if name not in TASKS:
        raise ValueError(f"unknown Adroit task {name!r}; expected {tuple(TASKS)}")
    env_id, observation_dim, action_dim = TASKS[name]
    return {"name": name, "env_id": env_id, "observation_dim": observation_dim,
            "action_dim": action_dim}


def task(config: dict) -> dict:
    evaluation = config["evaluation"]
    name = evaluation["task"]
    task_spec(name)
    if evaluation["reset_protocol"] != "independent_seed":
        raise ValueError(f"Adroit requires independent_seed, got {evaluation['reset_protocol']!r}")
    return {"name": name, "language": name, "init_source": "independent_seed",
            "initializers": {i: {"seed": int(evaluation["episode_seed"]) + i}
                             for i in range(evaluation["init_index_start"],
                                            evaluation["init_index_stop"])}}


def make_environment(data: dict):
    # Optional simulator imports must not become RoboSuite dependencies.
    import gymnasium as gym
    import gymnasium_robotics
    from gymnasium.envs.registration import EnvSpec

    gym.register_envs(gymnasium_robotics)
    expected = task_spec(data["adroit_task"])
    metadata = json.loads(Path(data["raw_hdf5"]).with_name("metadata.json").read_text())
    spec = metadata["env_spec"]
    spec = json.loads(spec) if isinstance(spec, str) else spec
    if spec["id"] != expected["env_id"]:
        raise ValueError(f"dataset environment {spec['id']!r}, expected {expected['env_id']!r}")
    return gym.make(EnvSpec.from_json(json.dumps(spec)))


class Episode:
    """Return native termination separately from info['success']."""

    def __init__(self, config: dict) -> None:
        self.data = config["data"]
        self.name = self.data["adroit_task"]
        if self.name != config["evaluation"]["task"]:
            raise ValueError("Adroit data and evaluation tasks disagree")
        self.spec = task_spec(self.name)
        self.state_dim, self.action_dim = 24, self.spec["action_dim"]
        self.env = make_environment(self.data)
        if (self.env.observation_space.shape != (self.spec["observation_dim"],)
                or self.env.action_space.shape != (self.action_dim,)):
            self.env.close()
            raise ValueError(f"unexpected Adroit dimensions for {self.name}")
        self.done = True

    def reset(self, initializer: dict) -> tuple[np.ndarray, dict]:
        self.env.reset(seed=int(initializer["seed"]))
        self.done = False
        return dp3_adroit.observed(self.env, self.data)

    def step(self, action: np.ndarray) -> tuple:
        if self.done:
            raise RuntimeError("cannot step a completed or uninitialized Adroit episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"invalid Adroit action shape/values: {action}")
        applied = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        _, reward, terminated, truncated, info = self.env.step(applied)
        if not np.isfinite(reward):
            raise ValueError(f"nonfinite Adroit reward {reward}")
        self.done = bool(terminated or truncated)
        state, clouds = dp3_adroit.observed(self.env, self.data)
        return state, clouds, float(reward), self.done, {
            **info, "success": bool(info["success"]), "applied_action": applied,
            "terminated": bool(terminated), "truncated": bool(truncated)}

    def close(self) -> None:
        try:
            dp3_adroit.close(self.env)
        finally:
            self.env.close()
