# Adroit and DexArt

These suites use the existing BPC pipeline and current sparsemax-Fisher policy.
The integration adds only native environments, the paper's PointNet input path,
and bank preparation. Both suites use the same launcher, policy fitting and
artifact cache, rollout loop, episode recording, and analyzer as RoboSuite.
Policy defaults come from `configs/bpc_common.json`; suite templates contain
environment settings and explicit policy overrides. The auxiliary PointNet BC
trainer is separate encoder preparation, not another BPC policy or evaluator.
The integration does not import the historical campaign controllers,
alternative encoders, checkpoints, datasets, or older entropy-LDA optimizer.

## Setup and run

Use separate native environments: Adroit uses Gymnasium-Robotics 1.4.2 and
MuJoCo 3.2.3; DexArt uses SAPIEN 2 and the DP3-vendored DexArt release at
`YanjieZe/3D-Diffusion-Policy@47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`
(`third_party/dexart-release`, with its assets and dependencies). SAPIEN 3 is
not a substitute. Both need NumPy, h5py, and the existing PyTorch policy runtime.
Simulator imports are optional, so RoboSuite does not acquire these dependencies.
No dependency installation or asset download happens implicitly.

Edit `configs/dexterous_common.json` for output/cache paths and
`runtime.native_python` (defaults to `python` on PATH; use an absolute environment
interpreter path for reproducibility). The suite configs are
`configs/adroit_pointnet.json` and `configs/dexart_pointnet.json`. Set their
`benchmarks[0].dataset_dir`, and for DexArt set `data.dexart_root` to the
DexArt source directory. By default each task directory contains:

```text
<dataset_dir>/<task>/
  bank.pt          # optional; generated from capture.hdf5 if absent
  pointnet.pt      # frozen auxiliary BC encoder export
  capture.hdf5     # needed to generate the bank
  raw.hdf5         # original Adroit dataset (not needed by DexArt)
  metadata.json    # Adroit dataset env_spec, beside raw.hdf5
```

Task-level `data` overrides can point to existing files without copying them.
The configured checkpoint hashes identify the archived 10/100-demo PointNet
exports. If training a new encoder, update the checkpoint path and SHA256 and
use a new bank path. A bank must match its encoder, task, input contract, and
capture hash; a matching array shape alone is insufficient.

```bash
python scripts/run_experiment.py configs/adroit_pointnet.json
python scripts/run_experiment.py configs/dexart_pointnet.json
```

Select the desired tasks by editing the suite's `benchmarks[0].tasks` list.
The same `--run` and `--stage fit|evaluate|analyze` options as RoboSuite work
for resuming. Native environments run serially (`env_workers=1`), with no
RGB video capture or Panda recovery controller. The launcher's native mode
uses the configured interpreter and `MUJOCO_GL=egl`. No Docker image is added.

## Inputs and encoder preparation

Adroit uses the audited DP3 depth projection, crop, and deterministic farthest
point sampling: 512 XYZ points plus 24 hand joint positions. Relocate is an
explicit extension of the Pen camera/crop, not an upstream DP3 task definition.
DexArt concatenates 1,024 observed XYZ points and 96 robot-mesh points; its
state has 32 components (33 for Bucket). RGB, cloud color/labels, and oracle
object state are not policy inputs. The state includes the native DexArt
elapsed-time component, as in the historical input contract.

PointNet produces 64 features. It was trained separately on each selected
task dataset using 2,000 auxiliary BC updates and then frozen; it is not a
released SDP checkpoint. The retained training command is:

```bash
PYTHONPATH=src python scripts/train_dp3_pointnet.py --config configs/pointnet_train.json
```

The example training config is for Door. Adapt its task, dimensions, capture
path, and input-contract hash for other tasks. It uses batch size 32, learning
rate 1e-4, weight decay 1e-6, seed 0, and a 10% episode-level validation split.
The auxiliary model sees two observations and predicts 16 actions; BPC itself
uses a 10-step history, a 10-step continuation, and executes one action per step.
Training writes the resolved config, metrics, status and `final.pt`; evaluation
loads only its frozen point encoder. Adroit XYZ normalization uses training-set
limits; DexArt uses identity XYZ normalization. No policy head is retained.

The bank builder accepts the historical capture schema, not arbitrary XYZ files:
root attributes include `suite`, `task`, `source_contract_sha256`, and
`training_admissible=true`; each `episode_N` contains floating arrays
`state[T+1,P]`, `point_cloud[T+1,N,3]`, and `actions[T,A]`.
The contract is returned by `dexterous.input_contract(data)`. Only pre-action
states/clouds (the first T rows) are encoded into the bank. Existing capture
tools and original environment setup records remain in
`chengine/bpc:deployment/gip_benchmarks`; they are not duplicated here.

## Evaluation and historical numbers

Both suites use 10 demonstration warmup actions. Adroit episodes have at most
200 steps; DexArt has 250, except Bucket's 100. Reset seeds start at 950000.
DexArt cycles a seeded instance catalog (seen instances by default). Adroit
actions remain continuous normalized actuators and are not scaled by temporal
rate. DexArt rate-scales only its first six Cartesian velocity components.

The native episode JSON distinguishes `success_final`, `success_any`,
`success_steps`, `terminated`, `truncated`, and `return`. Its `success` field,
consumed by the existing analyzer, is **final-step native task success**.
Time limits and generic environment termination are never counted as success.
Transient success does not stop native evaluation. Executed actions populate
the controller's history. RoboSuite retains its existing anytime-success rule.

The following counts belong to the **September 13 entropy-LDA campaign**, not
to a new evaluation of the sparsemax-Fisher implementation in this repository:

| Suite | Task | Final successes / rollouts |
|---|---|---:|
| Adroit | Door | 118 / 120 |
| Adroit | Hammer | 101 / 120 |
| Adroit | Pen | 34 / 120 |
| Adroit | Relocate | 0 / 120 |
| DexArt | Toilet | 94 / 120 |
| DexArt | Faucet | 57 / 120 |
| DexArt | Laptop | 74 / 120 |
| DexArt | Bucket | 54 / 120 |

The paper reports the six tasks excluding Relocate and Bucket. Counts combine
four retrieval-fit seeds (0–3), each using the same 30 resets, with 10 Adroit
or 100 DexArt demonstrations. The provided configs run one metric seed; change
`method.g.lda.seed` and the experiment name for each repeat. Residual RFF seed
stays 0. Source evidence is the original
`experiments/sparsemax_qp/artifacts/benchmark_transfer_20260913/{grid,resume_grid}_events.jsonl`
and `09_13_04_05_00_steady-ibis_entropy_*_dp3_pointnet_s*.json` configs in
`chengine/bpc`. The geometry and checkpoint code were ported from the frozen
`deployment/gip_benchmarks/runtime` package there. The current suite configs
retain 16,384 retrieval and residual RFFs, rank 70, and up to 8,192 anchors,
but use the clean repository's radius-normalized sparsemax teacher and adaptive
support instead of the historical entropy teacher and fixed 64 supports.
Native configs reserve at least 64 windows outside the fitting anchors for
adaptive-support calibration (`method.g.lda.calibration_holdout`). This matters
for the small Adroit banks: the fitting anchor count is capped at
`min(8192, bank_windows - 64)`. RoboSuite's existing anchor selection is unchanged.

## Validation

```bash
PYTHONPATH=src python -m unittest tests.test_dexterous tests.test_dexterous_bank -v
python scripts/run_tests.py --suite unit
python scripts/run_tests.py --suite integration
```

Tests exercise input/config contracts, terminal failure and transient success,
bank/checkpoint binding, and small CPU policy fits with dexterous action widths.
They do not establish a native rollout success rate.

On 2026-11-05, commit `dabe91e` completed native smoke tests on Dagobah's RTX
3090 using existing simulator dependencies, archived banks and frozen encoders.
Both ran the normal launcher through data, fit, evaluate and analyze, with seed
950000 and 10 warmup steps. Smoke overrides were 256 retrieval/residual RFFs,
128 fitting anchors, rank 16 and one episode; other suite settings were retained.

| Task | Demos | Controller steps | Final success | Total wall time | Sampled peak device usage |
|---|---:|---:|---|---:|---:|
| Adroit Door | 10 | 190 | Yes | 203 s | 857 MiB |
| DexArt Faucet | 100 | 240 | No | 208 s | 1,047 MiB |

Both processes exited zero; observations and proposed/executed actions were
finite, executed actions stayed in [-1, 1], and the saved summaries matched episode
outcomes. Both episodes reached their native time limits (truncated, not
terminated). Future-error trace padding beyond the horizon was excluded from
finite-value checks. Optional CUDA extensions failed to compile against the
server's compiler; the existing PyTorch fallback completed both runs. Times
include compilation attempts; device usage was sampled once per second and
includes the desktop baseline, so these are not performance measurements.

These checks validate two native execution paths with reduced fitting sizes.
They do not validate all eight tasks, the default 16,384-feature fits, compiled
CUDA kernels, or reproduction of the historical paper scores.
