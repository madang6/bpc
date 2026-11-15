"""Public DP3-derived DEXArt inputs; not a claim about unreleased SDP 3D code."""

from copy import deepcopy
from typing import Any

import numpy as np

from . import dexart

CONVENTION = "dp3_dexart_xyz_robot_state_v1"
POINT_KEY = "point_cloud"
OBSERVED_POINTS = 1024
ROBOT_POINTS = 96
ROBOT_LINKS = (
    "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip",
    "link_15.0", "link_3.0", "link_7.0", "link_11.0",
    "link_14.0", "link_2.0", "link_6.0", "link_10.0",
)
ROBOT_CONFIG = {"robot": {name: 8 for name in ROBOT_LINKS}}
POINT_CONFIG = {
    "use_seg": True, "use_2frame": True, "num_points": OBSERVED_POINTS,
    "pose_perturb_level": 0.5, "process_fn_kwargs": {"noise_level": 0.5},
}


def contract(task: str) -> dict:
    """Describe consumed values and their audited upstream provenance."""
    spec = dexart.task_spec(task)
    fields = [
        {"name": "robot_qpos", "slice": [0, 22], "source": "robot.get_qpos"},
        {"name": "palm_linear_velocity", "slice": [22, 25], "source": "palm_link.get_velocity"},
        {"name": "palm_angular_velocity", "slice": [25, 28], "source": "palm_link.get_angular_velocity"},
        {"name": "palm_world_position", "slice": [28, 31], "source": "palm_link.get_pose().p"},
    ]
    if task == "bucket":
        fields.append({"name": "palm_axis_world_z", "slice": [31, 32],
                       "source": "(palm_rotation @ [1,0,0])[2]"})
    fields.append({"name": "elapsed_fraction", "slice": [spec["observation_dim"] - 1, spec["observation_dim"]],
                   "source": "current_step / horizon"})
    return {
        "convention": CONVENTION, "task": task, "upstream_revision": dexart.UPSTREAM_REVISION,
        "reference": "public DP3; exact SDP 3D implementation unavailable",
        "state_dim": spec["observation_dim"], "state_fields": fields,
        "state_dtype": "preserve native floating dtype; no normalization in this adapter",
        "point_key": POINT_KEY, "point_shape": [OBSERVED_POINTS + ROBOT_POINTS, 3],
        "point_dtype": "float32", "point_frame": "robot base",
        "point_order": ["observed camera XYZ", "imagined robot XYZ"],
        "observed_source": "instance_1 Position texture, calibrated camera-to-robot transform, fixed workspace crop, native sampling/noise",
        "robot_source": "known robot visual meshes transformed by robot link poses; twelve links, eight points each",
        "native_point_config": deepcopy(POINT_CONFIG), "native_imagination_config": deepcopy(ROBOT_CONFIG),
        "excluded": ["RGB", "segmentation labels", "oracle_state", "goal geometry", "object geometry imagination"],
        "preprocessing": "XYZ slicing and concatenation only; no extra crop, resampling, normalization or FPS",
    }


def policy_observation(native: dict, task: str) -> tuple[np.ndarray, dict]:
    """Match DP3's XYZ-only observed/imagined inputs, with full robot state."""
    spec = dexart.task_spec(task)
    state = np.asarray(native["state"])
    if state.shape != (spec["observation_dim"],) or state.dtype.kind != "f" or not np.isfinite(state).all():
        raise ValueError(f"{task}: invalid robot state shape/dtype/values {state.shape}/{state.dtype}")
    observed = np.asarray(native[f"{dexart.CAMERA}-point_cloud"])
    imagined = np.asarray(native["imagination_robot"])
    if observed.shape not in {(OBSERVED_POINTS, 3), (OBSERVED_POINTS, 6)}:
        raise ValueError(f"{task}: expected observed points (1024,3) or (1024,6), got {observed.shape}")
    if imagined.shape != (ROBOT_POINTS, 7):
        raise ValueError(f"{task}: expected native robot imagination (96,7), got {imagined.shape}")
    # RGB and imagination class labels are deliberately not inspected.
    points = np.concatenate((observed[:, :3], imagined[:, :3]), axis=0).astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError(f"{task}: nonfinite consumed XYZ coordinates")
    return state.copy(), {POINT_KEY: points}


def validate_environment(env: Any, task: str) -> None:
    """Check configured native processing and actual robot geometry sources."""
    spec = dexart.task_spec(task)
    if env.task_config_name != task or env.robot.dof != dexart.PROPRIO_DIM or env.horizon != spec["horizon"]:
        raise ValueError(f"{task}: incompatible environment identity, robot dimensions or horizon")
    cameras = env.camera_infos
    if set(cameras) != {dexart.CAMERA} or cameras[dexart.CAMERA].get("point_cloud") != POINT_CONFIG:
        raise ValueError(f"{task}: native point processing differs: {cameras}")
    if env.imagination_infos != ROBOT_CONFIG or set(env.imagination_data) != {"robot"}:
        raise ValueError(f"{task}: only the pinned robot imagination is admitted: {env.imagination_infos}")
    geometry = env.imagination_data["robot"]
    if set(geometry) != set(ROBOT_LINKS):
        raise ValueError(f"{task}: unexpected imagined links: {tuple(geometry)}")
    robot_links = {id(link) for link in env.robot.get_links()}
    for name in ROBOT_LINKS:
        actor, points, label = geometry[name]
        if (id(actor) not in robot_links or actor.get_name() != name or label != 1
                or np.asarray(points).shape != (8, 3) or not np.isfinite(points).all()):
            raise ValueError(f"{task}: imagined geometry is not the expected robot link {name}")


def make_environment(data: dict, split: str = "seen", instance_id: int | None = None) -> Any:
    """Use the pinned existing factory and validate provenance before returning it."""
    env = dexart.make_environment(data, split, instance_id)
    try:
        validate_environment(env, data["dexart_task"])
    except BaseException:
        env.close()
        raise
    return env
