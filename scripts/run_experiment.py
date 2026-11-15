#!/usr/bin/env python3
"""Run one config through its benchmark-specific CUDA image."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", help="single experiment JSON")
    parser.add_argument(
        "--stage", choices=["all", "data", "prefit", "fit", "evaluate", "analyze"],
        default="all")
    parser.add_argument("--method", help="run one method named in the config")
    parser.add_argument("--run", help="existing timestamped run for a partial stage")
    parser.add_argument("--shard", metavar="K[,K...]/N", help="run zero-based shard residues K of N")
    parser.add_argument("--workers", help="comma-separated HOST=GPU[@WEIGHT] workers; HOST may be local")
    parser.add_argument("--detach", action="store_true", help="keep a distributed coordinator running")
    parser.add_argument("--benchmark", help=argparse.SUPPRESS)
    parser.add_argument("--fit", help=argparse.SUPPRESS)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _run_config(args):
    if args.run:
        return json.loads((Path(args.run).resolve() / "config.json").read_text())
    if not args.config:
        raise SystemExit("CONFIG is required unless --run points to an existing run")
    sys.path.insert(0, str(PROJECT / "src"))
    from dpc_nldl.io import read_config
    return read_config(args.config)


def _jobs(config, run=".", benchmark=None):
    sys.path.insert(0, str(PROJECT / "src"))
    from dpc_nldl.pipeline import _validate_methods, task_configs
    _validate_methods(config, None)
    jobs = task_configs(config, run, benchmark)
    for scoped, _, _ in jobs:
        _validate_methods(scoped, None)
    return jobs


def _worker(item):
    target_gpu, separator, weight = item.rpartition("@")
    target_gpu, weight = ((target_gpu, int(weight)) if separator and weight.isdigit()
                          else (item, 1))
    target, gpu = target_gpu.rsplit("=", 1) if "=" in target_gpu else (target_gpu, "all")
    return target, gpu, weight


def _environment(runtime):
    environment = os.environ.copy()
    key_file = Path(runtime.get("wandb_api_key_file", ""))
    if key_file.is_file():
        environment["WANDB_API_KEY"] = key_file.read_text().strip()
    return environment


def _fit_order(jobs, lookups_only=False, requested=None):
    from dpc_nldl.pipeline import method_variants, task_fit

    order = []
    for scoped, _, label in jobs:
        variants = method_variants(scoped)
        selected = task_fit(label, requested)
        order.extend((f"{label}:{name}" if len(jobs) > 1 else name)
                     for name in dict.fromkeys(
            item["fit"] for item in variants.values()
            if (not lookups_only or item["g_lookup"] != "past")
            and (selected is None or item["fit"] in selected.split(","))))
    return order


def _fit_allocation(jobs, workers, requested=None):
    order = _fit_order(jobs, requested=requested)
    allocation = [[] for _ in workers]
    cycle = [index for index, worker in enumerate(workers) for _ in range(worker[2])]
    for index, name in enumerate(order):
        allocation[cycle[index % len(cycle)]].append(name)
    return allocation


def host(args, inside_command=None):
    config = _run_config(args)
    jobs = _jobs(config, args.run or ".", args.benchmark)
    runtime = jobs[0][0]["runtime"]
    environment = _environment(runtime)
    data_root = os.environ.get("DPC_DATA_ROOT")
    r3m_cache = os.environ.get("DPC_R3M_CACHE", runtime.get("r3m_cache", "/home/admin/.r3m"))
    gpus = os.environ.get("DPC_GPUS", runtime["gpus"])
    if ((os.environ.get("DPC_WORKER") or runtime.get("native_coordinator"))
            and (native := runtime.get("native_python"))):
        environment["DPC_NATIVE"] = "1"
        environment.update({key: value.format(project=PROJECT)
                            for key, value in runtime.get("native_env", {}).items()})
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
            environment["MUJOCO_EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]
        if (config.get("policy", {}).get("name") == "pi05_lora"
                and args.stage in ("fit", "analyze")
                and not environment.get("WANDB_API_KEY")):
            raise RuntimeError("a W&B API key is required for pi0.5 LoRA curves")
        print(f"[runtime] {native} on GPU(s) {os.environ.get('CUDA_VISIBLE_DEVICES')}",
              flush=True)
        subprocess.run(_inside_command(args, native), env=environment, check=True)
        return
    command = [
        "docker", "run", "--rm", "--gpus", gpus,
        "--shm-size", runtime["shm_size"],
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-e", "NUMBA_CACHE_DIR=/tmp",
        "-e", "MPLCONFIGDIR=/tmp/dpc-nldl-matplotlib",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
        "-e", f"MUJOCO_GL={runtime['mujoco_gl']}",
        "-e", f"PYOPENGL_PLATFORM={runtime['mujoco_gl']}",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", f"PYTHONPATH=/opt/openpi/src:{PROJECT / 'src'}",
        "-v", f"{PROJECT}:{PROJECT}",
        "-v", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro",
    ]
    for key in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE",
                "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_RUN_GROUP"):
        if environment.get(key):
            command += ["-e", key]
    if not os.environ.get("DPC_SKIP_DATA_MOUNT"):
        source = data_root or runtime["readonly_data_root"]
        command += ["-v", f"{source}:{runtime['readonly_data_root']}:ro"]
    writable = [Path(args.run or config["experiment"]["output_dir"]).resolve()]
    fit_cache = os.environ.get("DPC_FIT_CACHE")
    cache_dir = (config["experiment"].get("cache_dir")
                 if args.stage in ("all", "fit") else None)
    if cache_dir and not fit_cache:
        writable.append(Path(cache_dir).resolve())
    for path in writable:
        if not path.is_relative_to(PROJECT):
            path.mkdir(parents=True, exist_ok=True)
            command += ["-v", f"{path}:{path}"]
    if args.stage in ("all", "data") and not args.shard:
        for bank_dir in {Path(scoped["data"]["bank"]).parent
                         for scoped, _, _ in jobs if "bank" in scoped["data"]}:
            bank_dir.mkdir(parents=True, exist_ok=True)
            command += ["-v", f"{bank_dir}:{bank_dir}"]
    if fit_cache and cache_dir:
        Path(fit_cache).mkdir(parents=True, exist_ok=True)
        command += ["-v", f"{fit_cache}:{cache_dir}"]
    if cache_dir and any(scoped.get("method", {}).get("g", {}).get("lda", {}).get("fit_all_histories", False) for scoped, _, _ in jobs):
        command += ["-e", f"TMPDIR={cache_dir}"]
    if any(scoped.get("policy", {}).get("name") != "pi05_lora"
           and scoped["data"]["feature_keys"] for scoped, _, _ in jobs):
        command += ["-v", f"{r3m_cache}:/tmp/.r3m:ro"]
    command += [
        "-v", "/dev/null:/tmp/robosuite.log",
        "-w", str(PROJECT),
        runtime["image"],
    ]
    if inside_command:
        command += inside_command
        print(f"[runtime] {runtime['image']} on GPU(s) {gpus}", flush=True)
        subprocess.run(command, env=environment, check=True)
        return
    command += _inside_command(args, "python")
    print(f"[runtime] {runtime['image']} on GPU(s) {gpus}", flush=True)
    subprocess.run(command, env=environment, check=True)


def _inside_command(args, python):
    command = [python, str(PROJECT / "scripts" / "run_experiment.py")]
    if args.config:
        command.append(str(Path(args.config).resolve()))
    command.extend(["--stage", args.stage, "--inside"])
    if args.run:
        command.extend(["--run", str(Path(args.run).resolve())])
    if args.method:
        command.extend(["--method", args.method])
    if args.shard:
        command.extend(["--shard", args.shard])
    if args.benchmark:
        command.extend(["--benchmark", args.benchmark])
    if args.fit:
        command.extend(["--fit", args.fit])
    return command


def _command(run, stage, method=None, shard=None, benchmark=None, fit=None):
    command = [sys.executable, str(Path(__file__).resolve()),
               "--run", str(run), "--stage", stage]
    if method:
        command += ["--method", method]
    if shard:
        command += ["--shard", shard]
    if benchmark:
        command += ["--benchmark", benchmark]
    if fit:
        command += ["--fit", fit]
    return command


def _launch(worker, run, stage, remotes, method=None, shard=None,
            benchmark=None, fit=None, data_root=None):
    target, gpu, _ = worker
    visible = gpu
    gpu = f"device={gpu}" if gpu != "all" else gpu
    if target == "local":
        return subprocess.Popen(
            _command(run, stage, method, shard, benchmark, fit),
            env={**os.environ, "DPC_WORKER": "1", "DPC_GPUS": gpu,
                 "CUDA_VISIBLE_DEVICES": visible})
    root = remotes[target]
    command = [
        "env", f"DPC_GPUS={gpu}", "DPC_WORKER=1",
        f"CUDA_VISIBLE_DEVICES={visible}",
        f"DPC_R3M_CACHE={Path(root).parent / '.r3m'}",
        *([f"DPC_FIT_CACHE={Path(root).parent / 'cache'}"]
          if stage == "fit" else []),
        f"DPC_DATA_ROOT={data_root}" if data_root else "DPC_SKIP_DATA_MOUNT=1",
        "python3", f"{root}/scripts/run_experiment.py", "--run",
        f"{root}/experiments/{run.name}", "--stage", stage,
    ]
    for flag, value in (("--method", method), ("--shard", shard),
                        ("--benchmark", benchmark), ("--fit", fit)):
        if value:
            command += [flag, value]
    return subprocess.Popen(["ssh", target, shlex.join(command)])


def _wait(processes):
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"worker exit codes: {codes}")


def _collect(remotes, jobs, run, products, fits=None):
    from dpc_nldl.pipeline import task_fit

    directories = {"bank_parts", "fit_parts", "policies", "episodes", "traces"}
    for target, root in remotes.items():
        for _, task_dir, label in jobs:
            if fits is not None and not task_fit(label, ",".join(fits[target])):
                continue
            task_dir.mkdir(parents=True, exist_ok=True)
            relative = task_dir.relative_to(run)
            for product in products:
                source = f"{target}:{root}/experiments/{run.name}/{relative}/{product}"
                subprocess.run([
                    "rsync", "-a", source + ("/" if product in directories else ""),
                    str(task_dir / product if product in directories else task_dir) + "/",
                ], check=True)


def _stage_fit_parts(remotes, jobs, run, targets=None):
    for target, root in remotes.items():
        if targets is not None and target not in targets:
            continue
        for _, task_dir, _ in jobs:
            if not (task_dir / "fit_parts").is_dir():
                continue
            relative = task_dir.relative_to(run)
            subprocess.run([
                "rsync", "-a", f"{task_dir / 'fit_parts'}/",
                f"{target}:{root}/experiments/{run.name}/{relative}/fit_parts/",
            ], check=True)


def run_all(args):
    """Run every stage through the same configured worker paths."""
    sys.path.insert(0, str(PROJECT / "src"))
    from dpc_nldl.io import write_json

    config = _run_config(args)
    _jobs(config)
    if args.run:
        run = Path(args.run).resolve()
    else:
        from dpc_nldl.pipeline import _new_run
        run = _new_run(config)
    if args.detach:
        log_path = run / "coordinator.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                _command(run, "all", args.method), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"[coordinator] pid {process.pid}; log {log_path}", flush=True)
        return
    timing_path = run / "stage_timings.json"
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else []
    for stage in ("data", "fit", "evaluate"):
        started = time.perf_counter()
        record = {"stage": stage, "started_unix": time.time(), "completed": False}
        timings.append(record)
        write_json(timing_path, timings)
        try:
            subprocess.run(_command(run, stage, args.method), check=True)
            record["completed"] = True
        finally:
            record.update(seconds=time.perf_counter() - started, finished_unix=time.time())
            write_json(timing_path, timings)
    print(f"[run] {run}", flush=True)


def _stage_run(target, root, run):
    # Banks are immutable: stage once and hard-link them into each run.
    banks = f"{Path(root).parent}/banks"
    subprocess.run(["rsync", "-aL", "--include=*/", "--include=bank.pt", "--exclude=*",
                    f"{run}/", f"{target}:{banks}/"], check=True)
    subprocess.run(["rsync", "-aL", f"--link-dest={banks}", f"{run}/",
                    f"{target}:{root}/experiments/{run.name}/"], check=True)


def distribute(args):
    """Run configured data, fit, or evaluation workers, then merge."""
    if args.stage not in ("data", "fit", "evaluate"):
        raise SystemExit("--workers supports --stage data, fit, or evaluate")
    if args.stage in ("fit", "evaluate") and not args.run:
        raise SystemExit(f"distributed {args.stage} requires --run")
    if args.shard:
        raise SystemExit("--workers assigns shards; do not also pass --shard")
    config = _run_config(args)
    pi05_lora = config.get("policy", {}).get("name") == "pi05_lora"
    if args.run:
        run = Path(args.run).resolve()
    else:
        sys.path.insert(0, str(PROJECT / "src"))
        from dpc_nldl.pipeline import _new_run
        run = _new_run(config)
    jobs = _jobs(config, run, args.benchmark)
    if args.detach:
        command = _command(run, args.stage, args.method,
                           benchmark=args.benchmark, fit=args.fit) + ["--workers", args.workers]
        log_path = run / "coordinator.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(f"[coordinator] pid {process.pid}; log {log_path}", flush=True)
        return
    runtime = jobs[0][0]["runtime"]
    if args.stage == "data" and all(
            (bank := scoped["data"].get("bank") or scoped["data"].get("reuse_bank"))
            and Path(bank).is_file() for scoped, _, _ in jobs):
        args.run, args.config, args.workers = str(run), None, None
        host(args)
        print(f"[run] {run}", flush=True)
        return
    try:
        workers = [_worker(item) for item in args.workers.split(",")]
    except ValueError:
        raise SystemExit("workers must be HOST=GPU[@positive integer weight]") from None
    if any(not target or weight < 1 or (gpu != "all" and not gpu.isdigit())
           for target, gpu, weight in workers):
        raise SystemExit("workers must be HOST=GPU[@positive integer weight]")
    items = (len(jobs) if pi05_lora and args.stage in ("data", "fit") else min(
        int(scoped["data"]["demonstrations"]) if args.stage in ("data", "fit") else
        int(scoped["evaluation"]["init_index_stop"])
        - int(scoped["evaluation"]["init_index_start"])
        for scoped, _, _ in jobs))
    slots = sum(worker[2] for worker in workers)
    if slots > items:
        raise SystemExit("worker weights cannot exceed configured items")
    remotes, remote_data = {}, {}
    needs_data = args.stage in ("data", "evaluate") or pi05_lora
    for target, _, _ in workers:
        if target == "local" or target in remotes:
            continue
        home = subprocess.check_output(["ssh", target, "pwd"], text=True).strip()
        worker_root = runtime.get("worker_roots", {}).get(target,
                                                           f"{home}/.dpc-nldl")
        root = f"{worker_root.rstrip('/')}/{run.name}"
        subprocess.run(["ssh", target, "mkdir", "-p",
                        f"{root}/experiments/{run.name}",
                        f"{Path(root).parent}/cache"], check=True)
        subprocess.run(["rsync", "-a", "--exclude=.git", "--exclude=experiments",
                        "--exclude=wandb", "--exclude=__pycache__",
                        f"{PROJECT}/", f"{target}:{root}/"], check=True)
        if any(scoped.get("policy", {}).get("name") != "pi05_lora"
               and scoped["data"]["feature_keys"] for scoped, _, _ in jobs):
            cache = Path(runtime.get("r3m_cache", "~/.r3m")).expanduser()
            subprocess.run(["rsync", "-a", f"{cache}/",
                            f"{target}:{Path(root).parent}/.r3m/"], check=True)
        _stage_run(target, root, run)
        if runtime.get("native_python") or needs_data:
            remote_data[target] = str(Path(root).parent / "data")
        if needs_data:
            readonly = Path(runtime["readonly_data_root"])
            sources = set()
            for scoped, _, _ in jobs:
                data = scoped["data"]
                sources.add(data["raw_hdf5"])
                if args.stage == "evaluate":
                    sources.update(filter(None, (data.get("reset_catalog"),)))
                if pi05_lora:
                    sources.update(data["pi05"][key] for key in ("checkpoint", "tokenizer"))
            for source in sorted(map(Path, sources)):
                destination = Path(remote_data[target]) / source.relative_to(readonly)
                subprocess.run(["ssh", target, "mkdir", "-p",
                                str(destination.parent)], check=True)
                suffix = "/" if source.is_dir() else ""
                subprocess.run(["rsync", "-a", str(source) + suffix,
                                f"{target}:{destination}{suffix}"], check=True)
        remotes[target] = root

    count = len(workers)
    allocation = (_fit_allocation(jobs, workers, args.fit)
                  if args.stage == "fit" and not pi05_lora else None)
    if allocation:
        lookup_fits = set(_fit_order(jobs, lookups_only=True))
        if lookup_fits:
            preparations, preparation_remotes, preparation_fits = [], {}, {}
            for worker, names in zip(workers, allocation):
                names = [name for name in names if name in lookup_fits]
                if not names:
                    continue
                preparations.append(_launch(
                    worker, run, "prefit", remotes, benchmark=args.benchmark,
                    fit=",".join(names)))
                if worker[0] != "local":
                    preparation_remotes[worker[0]] = remotes[worker[0]]
                    preparation_fits.setdefault(worker[0], []).extend(names)
            _wait(preparations)
            _collect(preparation_remotes, jobs, run, ("fit_parts",), preparation_fits)
            _stage_fit_parts(remotes, jobs, run)

    processes = []
    slot = 0
    for worker in workers:
        target, _, weight = worker
        owned = ",".join(map(str, range(slot, slot + weight)))
        shard = None if args.stage == "fit" and count == 1 else f"{owned}/{slots}"
        slot += weight
        processes.append(_launch(
            worker, run, args.stage, remotes, args.method, shard,
            args.benchmark, args.fit, remote_data.get(target)))
    _wait(processes)
    products = (("data.json",) if pi05_lora and args.stage == "data" else
                () if pi05_lora and args.stage == "fit" else
                ("bank_parts",) if args.stage == "data" else
                ("fit_parts",) if args.stage == "fit" and count > 1 else
                ("policies",) if args.stage == "fit" else
                ("episodes", "traces"))
    _collect(remotes, jobs, run, products,
             {target: args.fit.split(",") for target in remotes}
             if args.stage == "fit" and args.fit else None)
    if pi05_lora and args.stage == "evaluate":
        _wait([_launch(workers[0], run, "analyze", remotes,
                       benchmark=args.benchmark, data_root=remote_data.get(workers[0][0]))])
    if pi05_lora and args.stage in ("data", "fit"):
        print(f"[run] {run}", flush=True)
        return
    if args.stage == "fit" and count > 1:
        active_remotes = {worker[0] for worker, names in zip(workers, allocation)
                          if names and worker[0] != "local"}
        _stage_fit_parts(remotes, jobs, run, active_remotes)
        processes = []
        for worker, names in zip(workers, allocation):
            if not names:
                continue
            processes.append(_launch(
                worker, run, "fit", remotes, benchmark=args.benchmark,
                fit=",".join(names)))
        _wait(processes)
        _collect({target: remotes[target] for target in active_remotes},
                 jobs, run, ("policies",), {
                     target: [name for worker, names in zip(workers, allocation)
                              if worker[0] == target for name in names]
                     for target in active_remotes})
    if args.stage == "fit":
        environment = {**os.environ, "DPC_WORKER": "1"}
        subprocess.run(_command(run, "fit", benchmark=args.benchmark, fit=args.fit),
                       env=environment, check=True)
        print(f"[run] {run}", flush=True)
        return
    args.run, args.config, args.workers, args.shard = str(run), None, None, None
    host(args)
    if args.stage == "evaluate":
        args.stage = "analyze"
        host(args)
    print(f"[run] {run}", flush=True)


def inside(args):
    sys.path.insert(0, str(PROJECT / "src"))
    from dpc_nldl.pipeline import run
    run(
        str(Path(args.config).resolve()) if args.config else None,
        stage=args.stage,
        run_dir=str(Path(args.run).resolve()) if args.run else None,
        method=args.method,
        shard=args.shard,
        benchmark=args.benchmark,
        fit=args.fit,
    )


if __name__ == "__main__":
    args = arguments()
    runtime = _run_config(args)["runtime"] if not args.inside else {}
    distributed_all = (args.stage == "all" and not args.shard and not args.workers
                       and not os.environ.get("DPC_WORKER")
                       and any(runtime.get(key) for key in
                               ("data_workers", "fit_worker", "fit_workers", "workers")))
    if (not args.inside and not args.shard and not args.workers
            and not os.environ.get("DPC_WORKER")
            and args.stage in ("data", "fit", "evaluate")):
        workers = (runtime.get("fit_workers", [runtime["fit_worker"]]
                               if "fit_worker" in runtime else [])
                   if args.stage == "fit" else
                   runtime.get("data_workers" if args.stage == "data" else "workers", []))
        args.workers = ",".join(workers) or None
    if args.detach and not args.workers and not distributed_all:
        raise SystemExit("--detach requires distributed data, fit, or evaluation workers")
    (run_all(args) if distributed_all else distribute(args) if args.workers
     else inside(args) if args.inside else host(args))
