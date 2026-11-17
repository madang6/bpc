# Behavior Predictive Control

[Paper](https://arxiv.org/abs/2609.30134) · [Project page](https://trainingfreebehaviorcloning.github.io) · [Stanford MSL](https://msl.stanford.edu)

This codebase accompanies the paper [**Training-free Behavior Cloning**](https://arxiv.org/abs/2609.30134),
which introduces Behavior Predictive Control (BPC). It provides fitting and
evaluation for RoboSuite/MimicGen, Adroit, and DexArt.

Run commands from the repository root. You will need an NVIDIA GPU, the simulator
environment, demonstration data, and encoder checkpoints. Update the paths in the
configs for your machine.

## RoboSuite / MimicGen

Use the prepared Docker runtime with NVIDIA GPU support. Build instructions and
runtime prerequisites are in the [simulator setup guide](docs/simulator_setup.md#robosuite--mimicgen).

Set dataset, checkpoint, output, and cache paths in
[robosuite_common.json](configs/robosuite_common.json), then run:

```bash
python3 scripts/run_experiment.py configs/robosuite_fisher_sparsemax_local.json
```

Tasks: Square, Stack, Coffee, Hammer Cleanup, Mug Cleanup, Nut Assembly,
Stack Three, and Threading.

## Adroit

Use a Python environment with CUDA-enabled PyTorch, Gymnasium-Robotics, MuJoCo,
and the project dependencies. Prepare the task data and PointNet checkpoints
using the [native-suite setup guide](docs/dexterous.md#setup-and-run).

Set your environment's Python path and output directories in
[dexterous_common.json](configs/dexterous_common.json), and dataset paths in
[adroit_pointnet.json](configs/adroit_pointnet.json). Activate that environment and run:

```bash
python scripts/run_experiment.py configs/adroit_pointnet.json
```

Tasks: Door, Hammer, and Pen.

## DexArt

Use a Python environment with CUDA-enabled PyTorch, SAPIEN 2, and the DexArt source
and assets. Prepare the task data and PointNet checkpoints using the
[native-suite setup guide](docs/dexterous.md#setup-and-run).

Set your environment's Python path and output directories in
[dexterous_common.json](configs/dexterous_common.json), and dataset and simulator
paths in [dexart_pointnet.json](configs/dexart_pointnet.json). Activate that environment and run:

```bash
python scripts/run_experiment.py configs/dexart_pointnet.json
```

Tasks: Toilet, Faucet, and Laptop.

## Results

The configuration, fitted policies, and evaluation results are saved under
`experiment.output_dir` for each run. To resume evaluation from an existing run:

```bash
python scripts/run_experiment.py --run /path/to/run --stage evaluate
```
