"""Fixed-step auxiliary behavior cloning for our DP3-style point encoder."""

import argparse
import json
import math
import platform
import resource
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from dpc_nldl.io import atomic_path, read_config
from dpc_nldl.dp3_pointnet import (BehaviorCloner, EpisodeWindows, FORMAT, REVISION, batch,
                                  episode_split, fit_normalizers, masked_action_mse, sha256,
                                  validate_source)


def write(path: Path, value: dict) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def evaluate(model: BehaviorCloner, windows: EpisodeWindows, indices: list[int], normalizers: dict,
             batch_size: int, device: str) -> float:
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            values = batch(windows, indices[start:start + batch_size], normalizers, device)
            loss = masked_action_mse(model(values['point_cloud'], values['state']), values['actions'], values['mask'])
            size = int(values['mask'].sum()) * model.action_dim
            total += float(loss) * size
            count += size
    return total / count


def train(config: dict) -> dict:
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'config.json', config)
    started = time.monotonic()
    status = {'run/status': 'running', 'trained_steps': 0, 'method': FORMAT,
              'claim': 'Our auxiliary behavior-cloning encoder, not SDP pretrained weights.'}
    write(output / 'status.json', status)
    try:
        data, architecture, settings = config['data'], config['model'], config['training']
        for key in ('steps', 'batch_size', 'log_every', 'validation_samples'):
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError(f'{key} must be positive int, got {settings[key]!r}')
        for key in ('state_dim', 'action_dim', 'num_points'):
            if type(architecture[key]) is not int or architecture[key] < 1:
                raise ValueError(f'{key} must be positive int, got {architecture[key]!r}')
        for key in ('learning_rate', 'weight_decay'):
            if not math.isfinite(settings[key]) or settings[key] < 0:
                raise ValueError(f'{key} must be finite/nonnegative, got {settings[key]!r}')
        if settings['learning_rate'] == 0:
            raise ValueError('learning_rate must be positive')
        device, seed = settings['device'], settings['seed']
        if settings.get('purpose', 'deployment') != 'deployment':
            raise ValueError(f"unsupported training purpose {settings['purpose']!r}")
        torch.manual_seed(seed)
        if str(device).startswith('cuda'):
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        generator = np.random.default_rng(seed)
        source_hash = sha256(data['hdf5'])
        if data.get('sha256', source_hash) != source_hash:
            raise ValueError('training HDF5 hash differs from configured source')
        with h5py.File(data['hdf5'], 'r') as source:
            ids = validate_source(source, data, architecture)
            training_ids, validation_ids = episode_split(ids, data['validation_fraction'], data['split_seed'])
            normalizers = fit_normalizers(source, training_ids, data['suite'])
            training_windows, validation_windows = EpisodeWindows(source, training_ids), EpisodeWindows(source, validation_ids)
            validation_indices = np.random.default_rng(data['split_seed']).choice(
                len(validation_windows), min(settings['validation_samples'], len(validation_windows)), replace=False).tolist()
            model = BehaviorCloner(architecture['state_dim'], architecture['action_dim']).to(device)
            if str(device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
            status.update(source_sha256=source_hash, training_episode_ids=training_ids,
                          validation_episode_ids=validation_ids, training_purpose='deployment',
                          training_admissible=bool(source.attrs.get('training_admissible') == True),
                          source_contract_sha256=data['source_contract_sha256'],
                          normalization_scheme=normalizers['point_scheme'],
                          python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, device=device)
            write(output / 'status.json', status)
            with (output / 'metrics.jsonl').open('x') as metrics:
                for step in range(1, settings['steps'] + 1):
                    model.train()
                    indices = generator.integers(len(training_windows), size=settings['batch_size']).tolist()
                    values = batch(training_windows, indices, normalizers, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = masked_action_mse(model(values['point_cloud'], values['state']), values['actions'], values['mask'])
                    if not torch.isfinite(loss):
                        raise ValueError(f'nonfinite training loss at step={step}')
                    loss.backward()
                    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                        raise ValueError(f'nonfinite/missing gradients at step={step}')
                    optimizer.step()
                    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                        raise ValueError(f'nonfinite parameters at step={step}')
                    status['trained_steps'] = step
                    if step == 1 or step % settings['log_every'] == 0 or step == settings['steps']:
                        validation_loss = evaluate(model, validation_windows, validation_indices, normalizers,
                                                   settings['batch_size'], device)
                        if not math.isfinite(validation_loss):
                            raise ValueError(f'nonfinite validation loss at step={step}')
                        metric = {'progress/global_step': step, 'train/loss': float(loss.detach()),
                                  'val/loss': validation_loss, 'elapsed_seconds': time.monotonic() - started}
                        metrics.write(json.dumps(metric, allow_nan=False) + '\n')
                        metrics.flush()
                        print(json.dumps(metric, allow_nan=False), flush=True)
                        status.update(last_metrics=metric, elapsed_seconds=metric['elapsed_seconds'])
                        write(output / 'status.json', status)
            payload = {'format': FORMAT, 'upstream_revision': REVISION, 'config': config,
                       'trained_steps': status['trained_steps'], 'training_purpose': 'deployment',
                       'training_admissible': status['training_admissible'], 'finite_gradients_verified': True,
                       'source': {'sha256': source_hash, **{key: data[key] for key in ('suite', 'task', 'source_contract_sha256')}},
                       'training_episode_ids': training_ids, 'validation_episode_ids': validation_ids,
                       'normalizers': normalizers,
                       'pointnet_state_dict': {key: value.detach().cpu() for key, value in model.pointnet.state_dict().items()}}
        if sha256(data['hdf5']) != source_hash:
            raise ValueError('source HDF5 changed during training')
        with (output / 'final.pt').open('xb') as stream:
            torch.save(payload, stream)
        status.update({'run/status': 'completed', 'checkpoint_sha256': sha256(output / 'final.pt'),
                       'checkpoint': str(output / 'final.pt'), 'config_sha256': sha256(output / 'config.json'),
                       'peak_process_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                       'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
                       'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device) if str(device).startswith('cuda') else 0,
                       'elapsed_seconds': time.monotonic() - started})
        write(output / 'status.json', status)
        return status
    except BaseException as error:
        status.update({'run/status': 'crashed', 'error': repr(error), 'elapsed_seconds': time.monotonic() - started})
        write(output / 'status.json', status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    train(read_config(arguments.config))


if __name__ == '__main__':
    main()
