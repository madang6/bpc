"""DP3-style PointNet trained on our demonstrations; not released SDP weights."""

import math
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from .io import sha256

REVISION = '47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97'
OBSERVATIONS = 2
ACTION_HORIZON = 16
FEATURES = 64
RANGE_EPS = 1e-4
FORMAT = 'our_dp3_pointnet_bc_v1'


class PointNetXYZ(nn.Module):
    """Exact configured DP3 XYZ layers, including max pooling and LayerNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, 64), nn.LayerNorm(64), nn.ReLU(),
                                 nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
                                 nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU())
        self.final_projection = nn.Sequential(nn.Linear(256, FEATURES), nn.LayerNorm(FEATURES))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] == 0:
            raise ValueError(f'expected [B,N,3] points, got {tuple(points.shape)}')
        return self.final_projection(torch.max(self.mlp(points), dim=1)[0])


class BehaviorCloner(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.pointnet = PointNetXYZ()
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, FEATURES), nn.ReLU(), nn.Linear(FEATURES, FEATURES))
        self.head = nn.Sequential(nn.Linear(OBSERVATIONS * FEATURES * 2, 256), nn.ReLU(),
                                  nn.Linear(256, ACTION_HORIZON * action_dim))
        self.action_dim = action_dim

    def forward(self, points: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch, history, count, channels = points.shape
        if history != OBSERVATIONS or channels != 3 or state.shape[:2] != (batch, history):
            raise ValueError(f'invalid BC input shapes {tuple(points.shape)}, {tuple(state.shape)}')
        visual = self.pointnet(points.reshape(batch * history, count, channels)).reshape(batch, history, FEATURES)
        fused = torch.cat((visual, self.state_mlp(state)), dim=-1)
        return self.head(fused.flatten(1)).reshape(batch, ACTION_HORIZON, self.action_dim)


def masked_action_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape or mask.shape != target.shape[:2] or mask.dtype != torch.bool or not mask.any():
        raise ValueError(f'invalid action loss shapes/mask {predicted.shape}, {target.shape}, {mask.shape}')
    # Select before subtraction so padded targets cannot influence the objective.
    return (predicted[mask] - target[mask]).square().mean()


def episode_split(ids: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if len(ids) < 2 or len(set(ids)) != len(ids) or not 0 < fraction < 1:
        raise ValueError(f'invalid episode split: count={len(ids)}, fraction={fraction}')
    permutation = np.random.default_rng(seed).permutation(len(ids))
    count = max(1, min(len(ids) - 1, math.ceil(len(ids) * fraction)))
    validation = {ids[index] for index in permutation[:count]}
    return [name for name in ids if name not in validation], [name for name in ids if name in validation]


def validate_source(source: h5py.File, data: dict, model: dict) -> list[str]:
    if source.attrs.get('training_admissible') != True:
        raise ValueError('source is not training-admissible')
    for key in ('suite', 'task', 'source_contract_sha256'):
        if source.attrs.get(key) != data[key] or not data[key]:
            raise ValueError(f'{key}: source {source.attrs.get(key)!r} differs from config {data[key]!r}')
    if re.fullmatch('[0-9a-f]{64}', data['source_contract_sha256']) is None:
        raise ValueError('source contract must be a canonical SHA256')
    if data['suite'] not in ('adroit', 'dexart'):
        raise ValueError(f'unsupported suite {data["suite"]}')
    unexpected = [name for name in source if re.fullmatch(r'(episode|attempt)_\d+', name) is None]
    if unexpected:
        raise ValueError(f'unexpected source groups {unexpected}')
    ids = sorted((name for name in source if name.startswith('episode_')),
                 key=lambda name: int(name.removeprefix('episode_')))
    for name in ids:
        group = source[name]
        length = len(group['actions'])
        expected = {'state': (length + 1, model['state_dim']),
                    'point_cloud': (length + 1, model['num_points'], 3),
                    'actions': (length, model['action_dim'])}
        if length < 1:
            raise ValueError(f'{name}: empty action trajectory')
        for key, shape in expected.items():
            if group[key].shape != shape or group[key].dtype.kind != 'f':
                raise ValueError(f'{name}/{key}: expected floating {shape}, got {group[key].shape}/{group[key].dtype}')
    return ids


def _limits(minimum, maximum):
    minimum, maximum = torch.as_tensor(minimum).clone(), torch.as_tensor(maximum).clone()
    span = maximum - minimum
    constant = span < RANGE_EPS
    span = torch.where(constant, torch.full_like(span, 2), span)
    scale = 2 / span
    offset = torch.where(constant, -minimum, -1 - minimum * scale)
    return {'min': minimum, 'max': maximum, 'scale': scale, 'offset': offset}


def fit_normalizers(source: h5py.File, training_ids: list[str], suite: str) -> dict:
    bounds = {}
    for name in training_ids:
        group = source[name]
        for field in ('state', 'actions', 'point_cloud'):
            values = np.asarray(group[field][:-1] if field != 'actions' else group[field][:], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f'{name}/{field}: nonfinite training values')
            values = values.reshape(-1, values.shape[-1])
            lower, upper = values.min(axis=0), values.max(axis=0)
            if field in bounds:
                lower = np.minimum(lower, bounds[field][0])
                upper = np.maximum(upper, bounds[field][1])
            bounds[field] = (lower, upper)
    result = {field: _limits(*pair) for field, pair in bounds.items()}
    scheme = 'train_limits' if suite == 'adroit' else 'identity'
    if scheme == 'identity':
        result['point_cloud']['scale'] = torch.ones(3)
        result['point_cloud']['offset'] = torch.zeros(3)
    result['point_scheme'] = scheme
    result['training_episode_ids'] = list(training_ids)
    return result


class EpisodeWindows:
    def __init__(self, source: h5py.File, episode_ids: list[str]) -> None:
        self.source = source
        self.indices = [(name, step) for name in episode_ids for step in range(len(source[name]['actions']))]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        name, step = self.indices[index]
        group = self.source[name]
        previous = max(0, step - 1)
        clouds = np.stack([group['point_cloud'][previous], group['point_cloud'][step]]).astype(np.float32)
        state = np.stack([group['state'][previous], group['state'][step]]).astype(np.float32)
        available = min(ACTION_HORIZON, len(group['actions']) - step)
        actions = np.zeros((ACTION_HORIZON, group['actions'].shape[1]), dtype=np.float32)
        actions[:available] = group['actions'][step:step + available]
        if not all(np.isfinite(value).all() for value in (clouds, state, actions)):
            raise ValueError(f'{name} step={step}: nonfinite sample')
        return {'point_cloud': torch.from_numpy(clouds), 'state': torch.from_numpy(state),
                'actions': torch.from_numpy(actions), 'mask': torch.arange(ACTION_HORIZON) < available}


def batch(windows: EpisodeWindows, indices: list[int], normalizers: dict, device: str) -> dict:
    samples = [windows[int(index)] for index in indices]
    result = {key: torch.stack([sample[key] for sample in samples]).to(device) for key in samples[0]}
    for field in ('point_cloud', 'state', 'actions'):
        result[field] = result[field] * normalizers[field]['scale'].to(device) + normalizers[field]['offset'].to(device)
    return result


class FrozenPointEncoder(nn.Module):
    def __init__(self, pointnet: PointNetXYZ, scale: torch.Tensor, offset: torch.Tensor, num_points: int) -> None:
        super().__init__()
        self.pointnet = pointnet
        self.num_points = num_points
        self.register_buffer('scale', scale)
        self.register_buffer('offset', offset)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(points, device=self.scale.device, dtype=torch.float32)
        if points.ndim != 3 or points.shape[1:] != (self.num_points, 3) or not torch.isfinite(points).all():
            raise ValueError(f'invalid frozen encoder input {tuple(points.shape)}')
        features = self.pointnet(points * self.scale + self.offset)
        if features.shape != (len(points), FEATURES) or not torch.isfinite(features).all():
            raise ValueError('invalid frozen point features')
        return features


def load_frozen_encoder(checkpoint: str | Path, expected_sha256: str, device: str) -> FrozenPointEncoder:
    if sha256(checkpoint) != expected_sha256:
        raise ValueError(f'checkpoint SHA mismatch: {checkpoint}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('format') != FORMAT or type(payload.get('trained_steps')) is not int or payload['trained_steps'] <= 0:
        raise ValueError('checkpoint is not a trained PointNet BC export')
    if payload.get('training_purpose') != 'deployment' or payload.get('training_admissible') is not True or payload['trained_steps'] != payload['config']['training']['steps']:
        raise ValueError('checkpoint is smoke-only, inadmissible, or incomplete')
    source = payload['source']
    if source.get('suite') not in ('adroit', 'dexart') or not source.get('task') or re.fullmatch('[0-9a-f]{64}', source.get('source_contract_sha256', '')) is None or re.fullmatch('[0-9a-f]{64}', source.get('sha256', '')) is None:
        raise ValueError(f'invalid source identity {source}')
    data = payload['config']['data']
    if any(source[key] != data[key] for key in ('suite', 'task', 'source_contract_sha256')):
        raise ValueError('checkpoint source identity differs from its recorded config')
    if data.get('sha256', source['sha256']) != source['sha256']:
        raise ValueError('checkpoint data hash differs from its recorded config')
    training_ids, validation_ids = payload['training_episode_ids'], payload['validation_episode_ids']
    if not training_ids or not validation_ids or set(training_ids) & set(validation_ids):
        raise ValueError('checkpoint training/validation split is empty or overlaps')
    if payload.get('finite_gradients_verified') is not True:
        raise ValueError('checkpoint lacks finite-gradient verification')
    normalizers = payload['normalizers']
    expected_scheme = 'train_limits' if source['suite'] == 'adroit' else 'identity'
    if normalizers.get('point_scheme') != expected_scheme or normalizers.get('training_episode_ids') != payload['training_episode_ids']:
        raise ValueError('point normalization or training split identity mismatch')
    values = normalizers['point_cloud']
    for key in ('scale', 'offset', 'min', 'max'):
        if not isinstance(values[key], torch.Tensor) or values[key].shape != (3,) or not torch.isfinite(values[key]).all():
            raise ValueError(f'invalid point normalization {key}')
    if (values['max'] < values['min']).any():
        raise ValueError('point normalization bounds reversed')
    expected = _limits(values['min'], values['max']) if expected_scheme == 'train_limits' else {'scale': torch.ones(3), 'offset': torch.zeros(3)}
    if not all(torch.equal(values[key], expected[key]) for key in ('scale', 'offset')):
        raise ValueError('point normalization coefficients differ from declared scheme')
    pointnet = PointNetXYZ()
    pointnet.load_state_dict(payload['pointnet_state_dict'], strict=True)
    if not all(torch.isfinite(value).all() for value in pointnet.state_dict().values()):
        raise ValueError('nonfinite pointnet checkpoint parameters')
    model = payload['config']['model']
    if type(model['num_points']) is not int or model['num_points'] <= 0:
        raise ValueError(f'invalid checkpoint point count {model["num_points"]!r}')
    encoder = FrozenPointEncoder(pointnet, values['scale'], values['offset'], model['num_points']).to(device).eval()
    encoder.source_identity = dict(source)
    return encoder
