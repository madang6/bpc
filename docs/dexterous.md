# Adroit and DexArt

## Setup and run

Use a separate Python environment for each simulator, with CUDA-enabled PyTorch,
NumPy, h5py, and the simulator's dependencies:

| Suite | Simulator |
|---|---|
| Adroit | Gymnasium-Robotics 1.4.2 and MuJoCo 3.2.3 |
| DexArt | SAPIEN 2.2.1 and `third_party/dexart-release` from `YanjieZe/3D-Diffusion-Policy@47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`, including its assets |

Set `runtime.native_python`, `experiment.output_dir`, and `experiment.cache_dir`
in [dexterous_common.json](../configs/dexterous_common.json). Set
`benchmarks[0].dataset_dir` in the suite config. For DexArt, also set
`data.dexart_root` to your DexArt checkout.

Each task directory has this layout:

```text
<dataset_dir>/<task>/
  pointnet.pt      # frozen encoder checkpoint
  capture.hdf5     # demonstration point clouds, states, and actions
  bank.pt          # generated from capture.hdf5 if absent
  raw.hdf5         # Adroit source dataset
  metadata.json    # Adroit env_spec, beside raw.hdf5
```

Datasets, checkpoints, and simulator assets are supplied separately. Task-level
`data` settings can override these paths. A matching `bank.pt` can be reused
without reading `capture.hdf5`.

From the repository root, run the appropriate suite in its environment:

```bash
python scripts/run_experiment.py configs/adroit_pointnet.json
python scripts/run_experiment.py configs/dexart_pointnet.json
```

The configs run Door, Hammer, and Pen for Adroit, and Toilet, Faucet, and Laptop
for DexArt. Evaluation is serial (`evaluation.env_workers=1`). Each config runs
one fit on 30 resets. To evaluate four fits, repeat with `method.g.lda.seed` set
to 0, 1, 2, and 3, using a different `experiment.name` for each run.

## Preparing a PointNet checkpoint

The supplied configs identify specific checkpoint files by SHA256. To fit a
new encoder, edit [pointnet_fit.json](../configs/pointnet_fit.json) for your
task, capture path, input dimensions, input-contract hash, and output directory,
then run:

```bash
PYTHONPATH=src python scripts/fit_dp3_pointnet.py --config configs/pointnet_fit.json
```

Set the task's `data.dp3_pointnet.checkpoint` to the resulting `final.pt` and
`data.dp3_pointnet.checkpoint_sha256` to its SHA256. Use a new `data.bank` path
when changing the encoder or capture.

## Capture format

The HDF5 root requires `suite`, `task`, `source_contract_sha256`, and
`fitting_admissible=true` attributes. The input specification is defined by
`input_contract(data)` in [dexterous.py](../src/bpc/dexterous.py).

Each `episode_N` group contains floating-point arrays:

| Array | Shape |
|---|---|
| `state` | `(T+1, state_dim)` |
| `point_cloud` | `(T+1, num_points, 3)` |
| `actions` | `(T, action_dim)` |

The bank uses the first `T` observations. Adroit inputs have 512 XYZ points and
24 state values; DexArt inputs have 1,120 XYZ points and 32 state values.
Demonstration capture tools are not included in this repository.
