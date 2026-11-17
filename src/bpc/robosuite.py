"""RoboSuite realization of the state, images, resets, and success signal."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .io import atomic_path
from .pipeline import ROBOSUITE_RENDER_CONVENTION


CAMERAS = {
    "agentview_rgb": "agentview",
    "eye_in_hand_rgb": "robot0_eye_in_hand",
}


def pi05_state(obs):
    """The exact 8-D state constructed by OpenPI's LIBERO runtime."""
    quaternion = np.asarray(obs["robot0_eef_quat"])
    scalar = np.clip(quaternion[..., 3], -1, 1)
    denominator = np.sqrt(1 - scalar ** 2)
    scale = np.divide(2 * np.arccos(scalar), denominator,
                      out=np.zeros_like(scalar), where=denominator > 1e-8)
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"])[..., :3],
        quaternion[..., :3] * scale[..., None],
        np.asarray(obs["robot0_gripper_qpos"])[..., :2],
    ], axis=-1).astype(np.float32)


def _register_tasks():
    import mimicgen  # noqa: F401
    import robosuite_task_zoo  # noqa: F401


def _compact_renderer(image_size):
    """Avoid unused framebuffer/shadow allocations before every context creation."""
    from robosuite.environments import base
    from robosuite.utils.binding_utils import MjRenderContextOffscreen

    def create_context(sim, device_id):
        sim.model.vis.global_.offwidth = image_size
        sim.model.vis.global_.offheight = image_size
        if not np.any(sim.model.light_castshadow):
            sim.model.vis.quality.shadowsize = 64
        return MjRenderContextOffscreen(sim, device_id=device_id)

    base.MjRenderContextOffscreen = create_context


class RobosuiteEpisode:
    def __init__(self, config):
        _register_tasks()
        import robosuite

        data = config["data"]
        self.cameras = data["cameras"]
        self.image_size = int(data["image_size"])
        _compact_renderer(self.image_size)
        with h5py.File(data["raw_hdf5"], "r") as source:
            metadata = json.loads(source["data"].attrs["env_args"])
        kwargs = metadata["env_kwargs"]
        kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
        )
        self.env = robosuite.make(metadata["env_name"], **kwargs)

    def _hide_collision_geometries(self):
        context = getattr(self.env.sim, "_render_context_offscreen", None)
        if context is None:
            camera = CAMERAS[self.cameras[0]]
            self.env.sim.render(
                camera_name=camera, width=self.image_size, height=self.image_size)
            context = self.env.sim._render_context_offscreen
        context.vopt.geomgroup[0] = 0

    @staticmethod
    def _proprio(obs):
        return np.concatenate([
            np.asarray(obs["robot0_eef_pos"])[..., :3],
            np.asarray(obs["robot0_eef_quat"])[..., :4],
            np.asarray(obs["robot0_gripper_qpos"])[..., :2],
        ], axis=-1).astype(np.float64)

    def _observation(self, obs=None):
        obs = obs or self.env._get_observations(force_update=True)
        state = self._proprio(obs)
        self.pi05_state = pi05_state(obs)
        self._hide_collision_geometries()
        images = {
            name: np.ascontiguousarray(self.env.sim.render(
                camera_name=CAMERAS[name], width=self.image_size,
                height=self.image_size)[::-1])
            for name in self.cameras
        }
        return state, images

    def reset(self, initializer):
        if isinstance(initializer, (int, np.integer)):
            np.random.seed(int(initializer))
            return self._observation(self.env.reset())
        self.env.reset()
        xml = self.env.edit_model_xml(initializer["model"])
        self.env.reset_from_xml_string(xml)
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(initializer["states"])
        self.env.sim.forward()
        return self._observation()

    def replay_demo(self, _, demo):
        return self.reset({
            "model": demo.attrs["model_file"],
            "states": np.asarray(demo["states"][0]),
        })

    def replay_state(self, state):
        self.env.sim.set_state_from_flattened(np.asarray(state))
        self.env.sim.forward()
        return self._observation()

    @staticmethod
    def recorded_proprio(demo):
        return RobosuiteEpisode._proprio(demo["obs"])

    def step(self, action):
        obs, reward, _, info = self.env.step(action)
        state, images = self._observation(obs)
        return state, images, float(reward), bool(self.env._check_success()), info

    def close(self):
        self.env.close()


Episode = RobosuiteEpisode
RENDER_CONVENTION = ROBOSUITE_RENDER_CONVENTION


def task(config):
    evaluation = config["evaluation"]
    indices = range(int(evaluation["init_index_start"]),
                    int(evaluation["init_index_stop"]))
    name = evaluation["task"]
    protocol = evaluation.get("reset_protocol", "independent_seed")
    if protocol == "serial_seed_catalog":
        catalog = torch.load(config["data"]["reset_catalog"], map_location="cpu")
        initializers = {index: {"model": catalog["model"],
                                "states": catalog["states"][index]}
                        for index in indices}
    elif protocol == "independent_seed":
        seed = int(evaluation["episode_seed"])
        initializers = {index: seed + index for index in indices}
    else:
        raise ValueError(f"unknown RoboSuite reset protocol {protocol}")
    return {
        "name": name,
        "language": name.replace("_", " "),
        "init_source": ("seeded_robosuite_resets" if protocol == "independent_seed"
                        else protocol),
        "initializers": {index: initializers[index] for index in indices},
    }


def prepare_reset_catalog(config):
    """Freeze one serial reset stream so reset IDs survive sharding."""
    path = config["data"].get("reset_catalog")
    if not path:
        return
    stop = int(config["evaluation"]["init_index_stop"])
    if Path(path).is_file():
        saved = torch.load(path, map_location="cpu")
        if "states" in saved and all(index in saved["states"] for index in range(stop)):
            return
    from .data import seed

    seed_value = int(config["evaluation"]["episode_seed"])
    seed(seed_value)
    environment = RobosuiteEpisode(config)
    model, states = None, {}
    try:
        for index in range(stop):
            environment.env.reset()
            model = environment.env.model.get_xml()
            states[index] = np.asarray(environment.env.sim.get_state().flatten())
    finally:
        environment.close()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        torch.save({"seed": seed_value, "model": model, "states": states}, temporary)


def demonstration_initializers(path, indices):
    with h5py.File(path, "r") as source:
        demos = sorted(source["data"], key=lambda name: int(name.split("_")[-1]))
        return {
            index: {
                "model": source["data"][demos[index]].attrs["model_file"],
                "states": np.asarray(source["data"][demos[index]]["states"][0]),
            }
            for index in indices
        }
