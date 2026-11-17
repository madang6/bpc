# Simulator setup

## RoboSuite / MimicGen

The experiment launcher uses Docker with NVIDIA GPU support. The supplied config
selects `bpc-robosuite:latest` in `runtime.image`.

To build that image, provide:

- A `conda-pack` environment archive containing `bin/conda-unpack`.
- A simulator base image tagged `bpc-simulator-base:latest`, providing timezone
  data at `/opt/conda/share/zoneinfo`. Tag your existing simulator image with this
  name before building.
- A directory containing `robosuite`, `robomimic`, `mimicgen`, and
  `robosuite-task-zoo` source checkouts and their assets.

These inputs are separate from this repository. From the repository root:

```bash
scripts/build_paper_runtime.sh /path/to/bpc_environment.tar.gz
scripts/build_robosuite_runtime.sh /path/to/simulator-sources
```

In `configs/robosuite_common.json`, set:

| Setting | Path or value |
|---|---|
| `benchmarks[0].dataset_dir` | Task HDF5 demonstrations, such as `square_d0.hdf5` |
| `benchmarks[0].bank_dir` | Writable directory for encoded demonstration banks |
| `experiment.output_dir` | Run outputs |
| `experiment.cache_dir` | Fitted-policy cache and temporary fitting storage |
| `runtime.readonly_data_root` | Dataset root mounted into Docker |
| `runtime.r3m_cache` (optional) | Frozen R3M weights; defaults to `~/.r3m`, or set `BPC_R3M_CACHE` |
| `runtime.data_workers`, `runtime.fit_workers`, `runtime.workers` | GPU assignments; `["local=0"]` selects local GPU 0 |
| `evaluation.env_workers` | Number of parallel simulator environments |

The runner prepares or reuses the encoded bank before fitting and evaluation.
Fisher fitting needs substantial host memory and temporary disk space; the
largest feature array alone is about 45 GB. Docker uses the experiment cache
for temporary storage. For native runs, set `TMPDIR` to a writable filesystem
with sufficient space.

The ablation entrypoints are `configs/robosuite_nhc_local.json`,
`configs/robosuite_nnc_local.json`, and `configs/robosuite_nlr_local.json`.
Pass one of these configs to the same launcher to run an ablation.

## Adroit and DexArt

See [native-suite setup](dexterous.md#setup-and-run) for environment versions,
dataset layout, checkpoint preparation, and input requirements.

## Paths and workers

The configs use `/data` for inputs and `/data/bpc` for outputs and caches.
Set these paths and the GPU assignments for your machine.

The π₀.₅ recipe uses an activated local OpenPI environment (`native_python:
"python"`), an OpenPI checkout at `/opt/openpi`, and checkpoints under
`/data/checkpoints/openpi`. Adjust `runtime.openpi_root` and its `native_env.PYTHONPATH`
together if your checkout is elsewhere.
