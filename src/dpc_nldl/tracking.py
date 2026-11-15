"""Minimal experiment publication."""

import hashlib
import os
from pathlib import Path


def publish(config, run_dir):
    """Publish the frozen config and complete results as two W&B artifacts."""
    key = os.environ.get("WANDB_API_KEY")
    key_file = Path(config.get("runtime", {}).get("wandb_api_key_file", ""))
    if not key and key_file.is_file():
        key = key_file.read_text().strip()
    if not key:
        return
    import wandb

    wandb.login(key=key)
    run_dir = Path(run_dir)
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "dpc-nldl"),
        entity=os.environ.get("WANDB_ENTITY"), name=run_dir.name, dir=str(run_dir),
        id=run_dir.name, resume="allow", job_type="evaluation",
        group=os.environ.get("WANDB_RUN_GROUP"))
    results = [run_dir / "summary.json", *run_dir.glob("tasks/*/*/results.csv")]
    if (run_dir / "results.csv").is_file():
        results.append(run_dir / "results.csv")
    digest = hashlib.sha256(run_dir.name.encode()).hexdigest()[:12]
    artifact_prefix = f"{run_dir.name[:100]}-{digest}"
    for kind, paths in (("config", [run_dir / "config.json"]),
                        ("results", results)):
        artifact = wandb.Artifact(f"{artifact_prefix}-{kind}", type=kind)
        for path in paths:
            artifact.add_file(path, name=str(path.relative_to(run_dir)))
        run.log_artifact(artifact)
    run.finish()
