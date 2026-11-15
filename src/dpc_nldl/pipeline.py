"""The experiment: data -> fit -> evaluate -> analyze."""

import hashlib
import json
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .io import merge_config, read_config, write_json


STAGES = ("data", "fit", "evaluate", "analyze")
METHODS = ("deepc_g_direct", "gip_direct")
RAW_METHODS = {"pi05_lora"}
RESIDUAL_METHODS = {"gip_direct"}
FIT_VERSION = 3
G_LOOKUPS = {"past", "action_ridge", "lda"}
G_WEIGHTS = {"optimized", "uniform"}
ROBOSUITE_RENDER_CONVENTION = "offscreen_collision_hidden_v1"
ROBOSUITE_CONSTRUCTIONS = {"robosuite_action_replay_v1",
                           "robosuite_recorded_state_rerender_v1"}


def method_variants(config):
    methods = config["methods"]
    variants = ({name: {"base": name} for name in methods}
                if isinstance(methods, list) else deepcopy(methods))
    for variant in variants.values():
        if variant.get("base") not in set(METHODS) | RAW_METHODS:
            raise ValueError(f"unknown configured method: {variant.get('base')}")
        variant.setdefault("g_lookup", "past")
        variant.setdefault("g_weights", "optimized")
        if variant["g_lookup"] not in G_LOOKUPS:
            raise ValueError(f"unknown G lookup {variant['g_lookup']}")
        if variant["g_weights"] not in G_WEIGHTS:
            raise ValueError(f"unknown G weights {variant['g_weights']}")
        variant["fit"] = ("default" if variant["g_lookup"] == "past"
                          else variant["g_lookup"])
        if variant["g_weights"] != "optimized":
            variant["fit"] += f"_{variant['g_weights']}"
    return variants


def validate_method_config(method):
    """Reject retired research choices at both configuration and Policy entry points."""
    g, residual = method.get("g", {}), method.get("residual", {})
    if g.get("lookup", "past") not in G_LOOKUPS:
        raise ValueError(f"unknown G lookup {g['lookup']}")
    if g.get("weights", "optimized") not in G_WEIGHTS:
        raise ValueError(f"unknown G weights {g['weights']}")
    for field in ("svd_rcond", "future_consistency"):
        if g.get(field, 0):
            raise ValueError(f"g.{field} is not supported")
    if g.get("constraint", "affine") != "affine":
        raise ValueError("g.constraint must be affine")
    if residual.get("model", "rff") != "rff":
        raise ValueError("residual.model must be rff")
    if residual.get("lookup", "g_supports") != "g_supports":
        raise ValueError("residual.lookup must be g_supports")
    if g.get("lookup") == "lda":
        lda = g.get("lda", {})
        if lda.get("retime", False):
            raise ValueError("LDA teacher retiming is not supported")
        if lda.get("teacher", "sparsemax") != "sparsemax":
            raise ValueError("LDA teacher must be sparsemax")
        if lda.get("teacher_distance_scale", "radius") != "radius":
            raise ValueError("LDA teacher_distance_scale must be radius")
        if not lda.get("fisher_rank") or int(lda["fisher_rank"]) <= 0:
            raise ValueError("LDA requires a positive fisher_rank")
        if lda.get("adaptive_support", "coverage") != "coverage":
            raise ValueError("LDA adaptive_support must be coverage")


def target_execution_options(method):
    """Expand the offline-target arithmetic/execution options without mutation.

    These belong to the fitted residual, unlike the inference-only method
    variant's ``retrieval_precision``. Non-default choices therefore enter both
    the full fit specification and the offline-target cache specification.
    """
    residual = method.get("residual", {})
    options = {
        "target_retrieval_precision": residual.get("target_retrieval_precision", "fp64"),
        "target_execution": residual.get("target_execution", "legacy"),
    }
    if options["target_retrieval_precision"] not in {"fp64", "mixed"}:
        raise ValueError("residual.target_retrieval_precision must be fp64 or mixed")
    if options["target_execution"] not in {"legacy", "batched"}:
        raise ValueError("residual.target_execution must be legacy or batched")
    return options


def temporal_method(method):
    """One temporal recipe; expand its archived spelling for stable fit identities."""
    method = deepcopy(method)
    validate_method_config(method)
    execution = target_execution_options(method)
    # Explicit historical defaults must still find the historical fitted model
    # and target cache. Actual execution changes retain distinct identities.
    for key, default in (("target_retrieval_precision", "fp64"),
                         ("target_execution", "legacy")):
        if execution[key] == default:
            method.get("residual", {}).pop(key, None)
    g = method.get("g", {})
    rates = tuple(map(float, g.get("lookup_rates", [1])))
    if (1 not in rates or len(set(rates)) != len(rates)
            or any(not math.isfinite(rate) or rate <= 0 for rate in rates)):
        raise ValueError("g.lookup_rates requires unique positive rates including 1")
    if len(rates) > 1:
        expected = {"state_offset": 1, "align_history": True, "align_future": True,
                    "rate_scaled_action_dims": list(range(min(6, method["action_dim"]))),
                    "temporal_support_screen": True}
        residual = method["residual"]
        for section, values in ((g, expected), (residual, {"action_source": "retimed_first"})):
            for key, value in values.items():
                if key in section and section[key] != value:
                    raise ValueError(f"{key}={section[key]!r} is superseded by the consistent temporal recipe")
                section[key] = value
        if residual.get("lookup", "g_supports") != "g_supports":
            raise ValueError("temporal retrieval requires independent G-support RFF correction")
    return method


def fit_groups(config):
    variants, groups = method_variants(config), {}
    for variant in variants.values():
        groups.setdefault(variant["fit"], []).append(variant)
    output = []
    for name, members in groups.items():
        method = temporal_method(config["method"])
        lookup = members[0]["g_lookup"]
        if lookup != "past":
            method["g"]["lookup"] = lookup
        else:
            method.get("g", {}).pop("lookup", None)
        weights = members[0]["g_weights"]
        if weights != "optimized":
            method["g"]["weights"] = weights
        else:
            method.get("g", {}).pop("weights", None)
        validate_method_config(method)
        data = deepcopy(config["data"])
        data.pop("reset_catalog", None)
        bases = {item["base"] for item in members}
        spec = {
            "fit_version": FIT_VERSION, "data": data, "method": method,
            "fit_residuals": bool(set(bases) & RESIDUAL_METHODS),
            # Preserve the identity of existing manuscript-recipe fit artifacts.
            "fit_posterior": False,
        }
        digest = hashlib.sha256(json.dumps(
            spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
        output.append({"name": name, "key": f"{name}__{digest}",
                       "method": method, "spec": spec})
    return output


def load_config(path):
    config = read_config(path)
    target = os.environ.get("DPC_DATA_ROOT") if os.environ.get("DPC_NATIVE") else None
    if target:
        source = config["runtime"]["readonly_data_root"].rstrip("/")

        def remap(value):
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, str) and (value == source or value.startswith(source + "/")):
                return target.rstrip("/") + value[len(source):]
            return value

        config = remap(config)
    return config


def _new_run(config):
    root = Path(config["experiment"]["output_dir"])
    now = datetime.now(ZoneInfo(config["runtime"]["timezone"]))
    name = now.strftime("%Y%m%d-%H%M%S") + "__" + config["experiment"]["name"]
    run_dir = root / name
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", config)
    return run_dir


def _validate_methods(config, method):
    policy = config.get("policy", {}).get("name")
    if policy is not None and policy not in RAW_METHODS:
        raise ValueError(f"unknown policy {policy}")
    if policy is None and config.get("data", {}).get("encoder", "r3m") != "r3m":
        raise ValueError("BPC visual encoder must be r3m")
    configured = method_variants(config)
    if not configured:
        raise ValueError("methods must be nonempty")
    extra = set().union(*(set(item) for item in configured.values())) - {
        "base", "fit", "g_lookup", "g_weights", "recovery",
        "retrieval_precision"}
    if extra:
        raise ValueError(f"unknown method fields: {sorted(extra)}")
    validate_method_config(config.get("method", {}))
    target_options = target_execution_options(config.get("method", {}))
    for variant in configured.values():
        if variant["base"] not in RAW_METHODS:
            resolved = deepcopy(config["method"])
            resolved.setdefault("g", {})["lookup"] = variant["g_lookup"]
            validate_method_config(resolved)
        precision = variant.get("retrieval_precision", "fp64")
        if precision not in {"fp64", "mixed"}:
            raise ValueError("retrieval_precision must be fp64 or mixed")
        if precision == "mixed" and (variant["g_lookup"] != "lda" or
                config["method"]["g"].get("lda", {}).get("adaptive_support") != "coverage" or
                not config["method"]["g"].get("lda", {}).get("fisher_rank")):
            raise ValueError("mixed retrieval requires Fisher adaptive coverage")
        if (target_options["target_retrieval_precision"] == "mixed"
                and variant["base"] in RESIDUAL_METHODS
                and (variant["g_lookup"] != "lda"
                     or config["method"]["g"].get("lda", {}).get("adaptive_support") != "coverage"
                     or not config["method"]["g"].get("lda", {}).get("fisher_rank"))):
            raise ValueError("mixed target retrieval requires Fisher adaptive coverage")
    if config["analysis"]["reference_method"] not in configured:
        raise ValueError("analysis reference_method must be configured")
    if method is not None and method not in configured:
        raise ValueError(f"method {method} is not selected by this run config")
    raw = set(configured) & RAW_METHODS
    if ((policy in RAW_METHODS and set(configured) != {policy})
            or (policy not in RAW_METHODS and raw)):
        raise ValueError("a pi0.5 policy requires its matching single method")


def task_configs(config, run_dir, selected_benchmark=None):
    """Resolve each configured benchmark task into the existing single-task pipeline."""
    if "benchmarks" not in config:
        evaluation = config["evaluation"]
        if evaluation.get("benchmark") != "robosuite":
            raise ValueError(f"unknown benchmark {evaluation.get('benchmark')}")
        return [(config, Path(run_dir), f"robosuite/{evaluation['task']}")]
    if len({item["kind"] for item in config["benchmarks"]}) != len(config["benchmarks"]):
        raise ValueError("configure each benchmark kind once")
    resolved = []
    for benchmark in config["benchmarks"]:
        kind = benchmark["kind"]
        if kind != "robosuite":
            raise ValueError(f"unknown benchmark {kind}")
        if selected_benchmark and kind != selected_benchmark:
            continue
        for item in benchmark["tasks"]:
            task = {"name": item} if isinstance(item, str) else item
            name = task["name"]
            scoped = deepcopy(config)
            scoped.pop("benchmarks")
            suffix = f"__{kind}-{name}"
            scoped["experiment"]["name"] += suffix
            if "fit_cache_name" in scoped["experiment"]:
                scoped["experiment"]["fit_cache_name"] += suffix
            for section in ("runtime", "evaluation", "data"):
                scoped[section] = merge_config(scoped[section], benchmark.get(section, {}))
            scoped["methods"] = deepcopy(task.get("methods", scoped["methods"]))
            if "method_base" in task:
                if isinstance(scoped["methods"], list):
                    raise ValueError("method_base requires named method variants")
                for variant in scoped["methods"].values():
                    variant["base"] = task["method_base"]
            scoped["analysis"]["reference_method"] = task.get(
                "reference_method", scoped["analysis"]["reference_method"])
            for section in ("method", "evaluation", "data"):
                if section in scoped:
                    scoped[section] = merge_config(scoped[section], task.get(section, {}))
            scoped["evaluation"].update(benchmark=kind, task=name)
            construction = benchmark.get("construction", "robosuite_action_replay_v1")
            if construction not in ROBOSUITE_CONSTRUCTIONS:
                raise ValueError(f"unknown {kind} construction {construction}")
            scoped["data"].update(
                raw_hdf5=str(Path(benchmark["dataset_dir"]) /
                             task.get("dataset", f"{name}.hdf5")),
                construction=construction,
            )
            if "bank_dir" in benchmark:
                scoped["data"]["bank"] = str(
                    Path(benchmark["bank_dir"]) /
                    task.get("bank", f"{name}.pt"))
            scoped["data"]["render_convention"] = ROBOSUITE_RENDER_CONVENTION
            if scoped["evaluation"].get("reset_protocol") == "serial_seed_catalog":
                bank = Path(scoped["data"]["bank"])
                seed = int(scoped["evaluation"]["episode_seed"])
                scoped["data"]["reset_catalog"] = str(
                    bank.with_suffix(f".resets_seed{seed}.pt"))
            resolved.append((scoped, Path(run_dir) / "tasks" / kind / name,
                             f"{kind}/{name}"))
    if not resolved:
        raise ValueError("benchmarks must select at least one task")
    return resolved


def aggregate(config, run_dir):
    jobs = task_configs(config, run_dir)
    summaries = {
        label: json.loads((task_dir / "summary.json").read_text())
        for _, task_dir, label in jobs if (task_dir / "summary.json").is_file()
    }
    write_json(Path(run_dir) / "summary.json", {
        "experiment": config["experiment"]["name"],
        "tasks": summaries,
    })


def task_fit(label, requested):
    """Resolve hidden task-qualified fit selections, retaining unqualified names."""
    if requested is None:
        return None
    return ",".join(name for item in requested.split(",")
                    for task, separator, name in [item.rpartition(":")]
                    if not separator or task == label)


def run(config_path=None, stage="all", run_dir=None, method=None, shard=None,
        benchmark=None, fit=None):
    if shard and stage not in ("data", "fit", "evaluate"):
        raise ValueError("--shard is only valid with --stage data, fit, or evaluate")
    if run_dir is None:
        if config_path is None:
            raise ValueError("a config is required to create a run")
        if stage not in ("all", "data"):
            raise ValueError("partial stages after data require --run")
        config = load_config(config_path)
        _validate_methods(config, method)
        run_dir = _new_run(config)
    else:
        run_dir = Path(run_dir).resolve()
        config = load_config(run_dir / "config.json")
        _validate_methods(config, method)
    jobs = task_configs(config, run_dir, benchmark)
    if fit:
        unknown = {item.rpartition(":")[0] for item in fit.split(",") if ":" in item}
        unknown -= {label for _, _, label in jobs}
        if unknown:
            raise ValueError(f"unknown fit tasks: {sorted(unknown)}")
    if (config.get("policy", {}).get("name") == "pi05_lora"
            and shard and stage in ("data", "fit")):
        from .data import select_shard
        jobs, shard = select_shard(jobs, shard), None
    for scoped, _, _ in jobs:
        _validate_methods(scoped, method)
    if shard:
        for _, task_dir, _ in jobs:
            products = (("bank_parts",) if stage == "data" else
                        (("fit_parts",) if stage == "fit" else ("episodes", "traces")))
            for product in products:
                (task_dir / product).mkdir(parents=True, exist_ok=True)
    selected = STAGES if stage == "all" else (stage,)
    print(f"[run] {run_dir}", flush=True)
    for current in selected:
        for scoped, task_dir, label in jobs:
            scoped_fit = task_fit(label, fit)
            if current in ("prefit", "fit") and scoped_fit == "":
                continue
            print(f"[{current}] {label}", flush=True)
            pi05 = scoped.get("policy", {}).get("name") in RAW_METHODS
            if current == "data":
                if pi05:
                    if shard:
                        raise ValueError("pi0.5 data preparation is task-sharded only")
                    from .pi05_finetune import prepare
                    prepare(scoped, task_dir)
                else:
                    from .data import prepare
                    prepare(scoped, task_dir, shard)
            elif current == "prefit":
                from .bpc import prepare_lookups
                prepare_lookups(scoped, task_dir, scoped_fit)
            elif current == "fit":
                if pi05:
                    from .pi05_finetune import fit as fit_pi05
                    fit_pi05(scoped, task_dir)
                else:
                    from .bpc import fit_policy
                    fit_policy(scoped, task_dir, shard, scoped_fit)
            elif current == "evaluate":
                from .evaluate import evaluate
                evaluate(scoped, task_dir, method, shard)
            elif current == "analyze":
                from .analyze import analyze
                analyze(scoped, task_dir, method)
            else:
                raise ValueError(f"unknown stage {current}")
        if current == "analyze" and benchmark is None:
            aggregate(config, run_dir)
            from .tracking import publish
            publish(config, run_dir)
    return run_dir
