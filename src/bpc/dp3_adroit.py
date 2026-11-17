"""Pinned DP3 Adroit robot24 and depth-derived XYZ observations.

The legacy cloud generator uses camera zero for extrinsics while rendering the
named camera. This reproduces that convention explicitly. MuJoCo's Renderer
already returns upright metric depth; legacy mujoco_py depth conversion is not
applied again. Call close(env) when the owning environment is closed.
"""
from copy import deepcopy
import math

import numpy as np

REVISION = '47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97'
IMAGE_SIZE = 84
POINT_COUNT = 512
SOURCE_HASHES = {
    'adroit_assets/DAPG_pen.xml': 'dba9d80f2d9a3732ced8216f2a4f196fd50987038194792abc94857a68b4ce17',
    'adroit_assets/DAPG_door.xml': '19a37b659a7e97743f9812718d4f1f4ae85f1554b1a09d073e8df24c8983ca90',
    'adroit_assets/DAPG_hammer.xml': 'd1c4f1648660113ddde07f7c0ac19fe1e7b65b0b9cf0c45c484e16cf7e3a91c4',
    'adroit_assets/DAPG_assets.xml': '74e3d795e2ddf487b237d07a109711ff884baaeeafba41bf141406e3bfc32dd2',

    'gym_util/mujoco_point_cloud.py': 'ce2ca9d855a40b5c0294ee0c69d619f2a553fcddf9e39bad33fc8d1a52ac9a40',
    'gym_util/mjpc_diffusion_wrapper.py': 'd507fe47a9b9c769fa9a514dcb07ca31bfffe9adbf9b9892b61cdc9c3833449f',
    'env/adroit/rrl_local/rrl_multicam.py': '0f825eefada9b65e85ea192e7f2265597c4167e497c0d72005d4d26744e2a01e',
}
TASKS = {
    'pen': {'qpos_dim': 30, 'qpos_start': 0, 'camera': 'vil_camera', 'minimum_z': -0.79,
            'rotate': False, 'position': [0., -.8, .8], 'quaternion': [.92388, .382683, 0., 0.]},
    'door': {'qpos_dim': 30, 'qpos_start': 4, 'camera': 'top', 'minimum_z': -0.499,
             'rotate': True, 'position': [0., 0., 1.5], 'quaternion': [math.cos(1.57), 0., 0., math.sin(1.57)]},
    'hammer': {'qpos_dim': 33, 'qpos_start': 2, 'camera': 'top', 'minimum_z': -0.099,
               'rotate': True, 'position': [0., 0., 1.1], 'quaternion': [math.cos(1.57), 0., 0., math.sin(1.57)]},
    'relocate': {'qpos_dim': 36, 'qpos_start': 6, 'camera': 'vil_camera', 'minimum_z': -10.,
                 'rotate': False, 'position': [0., -.8, .8], 'quaternion': [.92388, .382683, 0., 0.],
                 'extension': True},
}


def task_spec(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f'no pinned DP3 Adroit point-cloud configuration for {task!r}; supported: {list(TASKS)}')
    return deepcopy(TASKS[task])


def metadata(data: dict) -> dict:
    spec = task_spec(data['adroit_task'])
    if spec.get('extension') and data.get('dp3_adroit', {}).get('allow_relocate_extension') is not True:
        raise ValueError('Relocate has no upstream DP3 cloud configuration; explicit allow_relocate_extension is required')
    angle = math.radians(45)
    rotation = [[1., 0., 0.], [0., math.cos(angle), math.sin(angle)], [0., -math.sin(angle), math.cos(angle)]]
    return {'construction': 'dp3_adroit_robot24_depth_xyz_v1', 'source_revision': REVISION,
            'source_sha256': dict(SOURCE_HASHES), 'task': data['adroit_task'],
            'qpos_indices': list(range(spec['qpos_start'], spec['qpos_start'] + 24)),
            'camera': spec['camera'],
            'camera_override': {'fallback_slot': 'vil_camera', 'position': spec['position'],
                'quaternion_wxyz': spec['quaternion'], 'normalize_quaternion': True,
                'fovy_degrees': 45., 'mode': 'fixed', 'body_id': 0,
                'camera0_unchanged': True, 'physics_model_unchanged': True},
            'upstream_task_provided': not spec.get('extension', False),
            'extension_label': 'relocate_pen_view_full_fixed_crop_v1' if spec.get('extension') else None, 'image_size': IMAGE_SIZE, 'points': POINT_COUNT,
            'channels': ['x', 'y', 'z'], 'output_dtype': 'float32',
            'depth_convention': 'mujoco_renderer_upright_metric_depth',
            'intrinsics': 'named_camera_fovy; fx=fy=84/(2*tan(fovy/2)); cx=cy=42',
            'extrinsics': 'legacy_camera_enumeration_index0_cam_mat0_and_camera_body_model_pos',
            'rotation': rotation if spec['rotate'] else None,
            'scale': [1., 1., 1.], 'offset': [0., 0., 1.] if spec['rotate'] else [0., 0., 0.],
            'crop_minimum': [-10., -10., spec['minimum_z']], 'crop_maximum': [10., 10., 10.],
            'crop_bounds': 'strict', 'sampling': 'fps_xyz_float32_start0_first_argmax',
            'sampling_backend': 'numpy_reference_v1', 'padding': 'append_zero_xyz_if_count_at_most512',
            'parity_scope': 'legacy projection/crop and deterministic FPS algorithm; GPU tie-order parity requires a runtime comparison'}


def robot_qpos(qpos: np.ndarray, task: str) -> np.ndarray:
    spec = task_spec(task)
    value = np.asarray(qpos)
    if value.shape != (spec['qpos_dim'],):
        raise ValueError(f'{task} qpos shape {value.shape}; expected {(spec["qpos_dim"],)}')
    selected = np.array(value[spec['qpos_start']:spec['qpos_start'] + 24], dtype=np.float64, copy=True)
    if not np.isfinite(selected).all():
        raise ValueError(f'{task} robot qpos contains nonfinite values: {selected}')
    return selected


def sample_xyz(points: np.ndarray, count: int = POINT_COUNT) -> np.ndarray:
    original = np.asarray(points, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3 or not np.isfinite(original).all():
        raise ValueError(f'expected finite XYZ cloud, got shape {original.shape}')
    if type(count) is not int or count < 1:
        raise ValueError(f'invalid point count {count!r}')
    if len(original) <= count:
        return np.pad(original, ((0, count - len(original)), (0, 0))).astype(np.float32)
    # PyTorch3D's default FPS casts coordinates to float32 and starts at index0.
    points32 = original.astype(np.float32)
    nearest = np.full(len(original), np.inf, dtype=np.float32)
    indices = np.empty(count, dtype=np.int64)
    current = 0
    for index in range(count):
        indices[index] = current
        difference = points32 - points32[current]
        squared = difference * difference
        distance = (squared[:, 0] + squared[:, 1]) + squared[:, 2]
        np.minimum(nearest, distance, out=nearest)
        current = int(np.argmax(nearest))
    return original[indices].astype(np.float32)


def project_depth(depth: np.ndarray, model: object, camera_id: int, task: str) -> np.ndarray:
    """Return cropped, unsampled legacy XYZ; depth is already upright meters."""
    contract = metadata({'adroit_task': task, 'dp3_adroit': {'allow_relocate_extension': task == 'relocate'}})
    depth = np.asarray(depth, dtype=np.float64)
    if depth.shape != (IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError(f'expected finite nonnegative metric depth {(IMAGE_SIZE, IMAGE_SIZE)}, got {depth.shape}')
    if not 0 <= camera_id < len(model.cam_fovy):
        raise ValueError(f'camera index {camera_id} outside model')
    fovy = float(model.cam_fovy[camera_id])
    if not 0 < fovy < 180:
        raise ValueError(f'invalid named-camera vertical field of view {fovy}')
    focal = IMAGE_SIZE / (2 * math.tan(math.radians(fovy) / 2))
    y, x = np.indices(depth.shape)
    camera = np.stack(((x - IMAGE_SIZE / 2) * depth / focal,
                       (y - IMAGE_SIZE / 2) * depth / focal, depth), axis=-1)
    camera = camera[depth > 0]
    # Deliberately do not substitute camera_id or runtime cam_xpos/cam_xmat.
    # The pinned generator enumerates a one-camera list and therefore uses0.
    rotation = np.asarray(model.cam_mat0[0]).reshape(3, 3) @ np.diag([1., -1., -1.])
    position = np.asarray(model.body_pos[int(model.cam_bodyid[0])])
    points = camera @ rotation.T + position
    if contract['rotation'] is not None:
        points = points @ np.asarray(contract['rotation']).T
    points += np.asarray(contract['offset'])
    keep = np.all(points > np.asarray(contract['crop_minimum']), axis=1)
    keep &= np.all(points < np.asarray(contract['crop_maximum']), axis=1)
    return points[keep]


def prepare_camera_in_place(env: object, task: str) -> int:
    """Own an environment-local camera override; close restores the original slot."""
    spec = task_spec(task)
    native, model = env.unwrapped, env.unwrapped.model
    try:
        camera_id = int(model.camera(spec['camera']).id)
    except KeyError:
        if spec['camera'] != 'top':
            raise ValueError(f"missing pinned DP3 camera {spec['camera']!r} for {task}") from None
        try:
            camera_id = int(model.camera('vil_camera').id)
        except KeyError:
            raise ValueError(f'missing spare vil_camera slot for pinned {task} top camera') from None
    if camera_id == 0:
        raise ValueError('cannot override camera0: legacy DP3 projection depends on it')
    backup = getattr(native, '_dp3_adroit_camera_backup', None)
    if backup is not None:
        if backup['task'] != task:
            raise ValueError(f'camera override belongs to {backup["task"]}, requested {task}')
        return camera_id
    fields = ('cam_pos', 'cam_quat', 'cam_mat0', 'cam_fovy', 'cam_mode', 'cam_bodyid', 'cam_targetbodyid')
    native._dp3_adroit_camera_backup = {'task': task, 'id': camera_id,
        'values': {name: np.array(getattr(model, name)[camera_id], copy=True) for name in fields}}
    model.cam_pos[camera_id] = spec['position']
    quaternion = np.asarray(spec['quaternion'], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    model.cam_quat[camera_id] = quaternion
    w, x, y, z = quaternion
    model.cam_mat0[camera_id] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]).ravel()
    model.cam_fovy[camera_id] = 45.
    model.cam_mode[camera_id] = 0
    model.cam_bodyid[camera_id] = 0
    model.cam_targetbodyid[camera_id] = -1
    return camera_id


def render_depth(env: object, camera: int) -> np.ndarray:
    # MuJoCo is optional: importing this module and testing projection needs no GL.
    import mujoco

    native = env.unwrapped
    renderer = getattr(native, '_dp3_adroit_depth_renderer', None)
    if renderer is None:
        renderer = mujoco.Renderer(native.model, height=IMAGE_SIZE, width=IMAGE_SIZE)
        native._dp3_adroit_depth_renderer = renderer
    mujoco.mj_camlight(native.model, native.data)
    renderer.update_scene(native.data, camera=camera)
    renderer.enable_depth_rendering()
    return renderer.render().copy()


def observed(env: object, data: dict) -> tuple[np.ndarray, dict]:
    metadata(data)
    native = env.unwrapped
    camera_id = prepare_camera_in_place(env, data['adroit_task'])
    robot = robot_qpos(native.data.qpos, data['adroit_task'])
    depth = render_depth(env, camera_id)
    cloud = project_depth(depth, native.model, camera_id, data['adroit_task'])
    return robot, {'point_cloud': sample_xyz(cloud)}


def close(env: object) -> None:
    native = env.unwrapped
    renderer = getattr(native, '_dp3_adroit_depth_renderer', None)
    if renderer is not None:
        renderer.close()
        del native._dp3_adroit_depth_renderer
    backup = getattr(native, '_dp3_adroit_camera_backup', None)
    if backup is not None:
        for name, value in backup['values'].items():
            getattr(native.model, name)[backup['id']] = value
        del native._dp3_adroit_camera_backup
