"""Evaluate every selected method on one benchmark task."""

import json
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .data import (benchmark_task, evaluation_seed,
                   full_state, load_bank, make_encoder, make_episode, seed,
                   select_shard)
from .bpc import apply_variant_runtime, load_policy
from .io import atomic_path, write_json
from .pipeline import RAW_METHODS, method_variants


def _warmup_demo(policy, trajectories, full_initial, steps, action_key):
    starts = policy.demo_start_state
    query = policy.scale_state(full_initial)
    scaled_starts = (starts - policy.state_mean) / policy.state_scale
    index = int(torch.square(scaled_starts - query).sum(1).argmin())
    actions = np.asarray(trajectories[index][action_key], dtype=np.float64)
    if actions.shape[1] < steps:
        raise ValueError(f"warmup demonstration {index} has fewer than {steps} actions")
    return index, actions[:, :steps].copy()


def _save_trace(path, trace):
    with atomic_path(path) as temporary:
        np.savez_compressed(temporary, **trace)


def _action_trace(proposed, executed):
    """Preserve the exact controller and env.step action dtypes for replay."""
    return {"proposed_action": np.asarray(proposed),
            "executed_action": np.asarray(executed)}


def _append_video(video, images, cameras):
    if video is not None:
        video.append_data(np.concatenate([images[key] for key in cameras], axis=1))


@contextmanager
def _episode(config, initial_state, episode_seed, video_path):
    """Own simulator/video lifetime for both learned and raw-policy rollouts."""
    seed(episode_seed)
    environment = make_episode(config)
    video, started = None, time.monotonic()
    try:
        initializer = ({**initial_state, "_seed": episode_seed}
                       if isinstance(initial_state, dict) else initial_state)
        proprio, images = environment.reset(initializer)
        if video_path is not None:
            import imageio
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            video = imageio.get_writer(video_path, fps=20, codec="libx264")
        _append_video(video, images, config["data"]["cameras"])
        yield environment, proprio, images, video, started
    finally:
        try:
            if video is not None:
                video.close()
        finally:
            environment.close()


def _finish_video(video_path, success):
    if video_path is None:
        return None
    path = Path(video_path)
    output = path.with_name(f"{path.stem}__{'success' if success else 'failure'}.mp4")
    path.replace(output)
    return output


def _prediction_diagnostics(mechanism, policy, proprio, executed):
    """Compact horizon errors; raw predictions remain available in the trace."""
    if "predicted_action_z" not in mechanism:
        return {}
    action = np.asarray(mechanism["predicted_action_z"])
    state = np.asarray(mechanism["predicted_proprio_z"])
    actual_action = ((np.asarray(executed) - policy.action_mean.cpu().numpy())
                     / policy.action_scale.cpu().numpy())
    actual_state = ((np.asarray(proprio) - policy.state_mean[:policy.proprio_dim].cpu().numpy())
                    / policy.state_scale[:policy.proprio_dim].cpu().numpy())

    def errors(predicted, actual, offset):
        output = np.full(predicted.shape[:2], np.nan, dtype=np.float32)
        for horizon in range(predicted.shape[1]):
            count = min(len(predicted), len(actual) - horizon - offset)
            output[:count, horizon] = np.linalg.norm(
                predicted[:count, horizon] - actual[horizon + offset:horizon + offset + count],
                axis=1)
        return output

    return {
        "future_action_error": errors(action, actual_action, 0),
        "future_proprio_error": errors(state, actual_state, 1),
        "action_plan_shift_error": np.linalg.norm(
            action[:-1, 1:] - action[1:, :-1], axis=2),
        "proprio_plan_shift_error": np.linalg.norm(
            state[:-1, 1:] - state[1:, :-1], axis=2),
    }


def _write_trace(path, policy, warmup_demo, proprio, proposed, executed,
                 mechanism, query_history):
    _save_trace(path, {
        "init_index": np.asarray(int(Path(path).stem), dtype=np.int16),
        "warmup_demo": np.asarray(warmup_demo, dtype=np.int16),
        "proprio": np.asarray(proprio, dtype=np.float32),
        **_action_trace(proposed, executed),
        "query_history_z": np.asarray(query_history, dtype=np.float64),
        **_prediction_diagnostics(mechanism, policy, proprio, executed),
        **{key: np.asarray(value, dtype=np.float32) for key, value in mechanism.items()},
    })


def _rollout(config, policy, trajectories, encode, initial_state,
             episode_seed, trace_path=None, video_path=None):
    evaluation, data = config["evaluation"], config["data"]
    clip_low, clip_high = map(float, evaluation["action_clip"])
    warmup_steps = int(evaluation["warmup_steps"])
    controller_steps = int(evaluation["controller_steps"])
    native = evaluation.get("benchmark") in {"adroit", "dexart"}
    done = final_success = terminated = truncated = False
    total_reward, success_steps, warmup_completed = 0.0, 0, 0
    with _episode(config, initial_state, episode_seed, video_path) as session:
        environment, proprio, images, video, started = session
        state = full_state(proprio, images, data, encode)
        warmup_demo, warmup_actions = _warmup_demo(
            policy, trajectories, state, warmup_steps,
            data["action_key"])
        policy.reset()
        policy._record_history = trace_path is not None
        previous = np.zeros(policy.action_dim, dtype=np.float64)
        success = False
        steps_to_success = None
        for step in range(warmup_steps):
            policy.action_for(state, previous)
            action = np.clip(warmup_actions[:, step], clip_low, clip_high)
            proprio, images, reward, done, info = environment.step(action)
            total_reward += float(reward)
            warmup_completed += 1
            final_success = bool(info["success"]) if native else bool(done)
            success_steps += int(final_success)
            terminated, truncated = bool(info.get("terminated", False)), bool(info.get("truncated", False))
            if final_success and not success:
                steps_to_success = 0
            success |= final_success
            _append_video(video, images, data["cameras"])
            previous = np.asarray(info.get("applied_action", action)).copy()
            state = full_state(proprio, images, data, encode)
            if native and done:
                break

        proprio_trace, proposed, executed, mechanism = [proprio.copy()], [], [], {}
        completed = 0
        for step in range(controller_steps):
            if (native and done) or (not native and success and not evaluation["run_full_horizon_after_success"]):
                break
            action = policy.action_for(state, previous)
            if action.shape != (policy.action_dim,) or not np.isfinite(action).all():
                raise FloatingPointError(f"{policy.method} produced an invalid action")
            clipped = np.clip(action, clip_low, clip_high)
            if trace_path is not None:
                proposed.append(action)
                for key, value in policy.last_trace.items():
                    mechanism.setdefault(key, []).append(value)

            proprio, images, reward, done, info = environment.step(clipped)
            applied = np.asarray(info.get("applied_action", clipped))
            total_reward += float(reward)
            final_success = bool(info["success"]) if native else bool(done)
            success_steps += int(final_success)
            terminated, truncated = bool(info.get("terminated", False)), bool(info.get("truncated", False))
            _append_video(video, images, data["cameras"])
            completed += 1
            if trace_path is not None:
                executed.append(applied)
                proprio_trace.append(proprio.copy())
            previous = {"proposed": action, "executed": applied}[evaluation.get("action_history", "executed")].copy()
            if final_success and not success:
                success, steps_to_success = True, step + 1
            if done and (native or not evaluation["run_full_horizon_after_success"]):
                break
            if step + 1 < controller_steps:
                state = full_state(proprio, images, data, encode)
        elapsed = time.monotonic() - started

    reported_success = final_success if native else success
    video_output = _finish_video(video_path, reported_success)
    if trace_path is not None and (completed or not native):
        start = warmup_steps - policy.history + 1
        action = (torch.stack(policy._action_history[start:])
                  - policy.action_mean) / policy.action_scale
        _write_trace(trace_path, policy, warmup_demo, proprio_trace, proposed, executed,
                     mechanism, torch.cat([
                         action, torch.stack(policy._state_history[start:])], 1).cpu().numpy())
    return {
        "seed": episode_seed,
        "success": reported_success,
        **({"success_final": final_success, "success_any": success,
            "success_steps": success_steps, "return": total_reward,
            "terminated": terminated, "truncated": truncated} if native else {}),
        "steps_to_success": steps_to_success,
        "controller_steps": completed,
        "warmup_steps": warmup_completed,
        "warmup_demo": warmup_demo,
        "seconds": elapsed,
        "video": str(video_output) if video_output else None,
    }


def _packed_trace(trace):
    """One contiguous device copy of every per-world diagnostic, plus its layout."""
    names, values = list(trace), list(trace.values())
    count = len(values[0])
    packed = torch.cat([value.reshape(count, -1).double() for value in values], 1).cpu().numpy()
    layout, start = [], 0
    for name, value in zip(names, values):
        width = value[0].numel()
        layout.append((name, start, start + width, tuple(value.shape[1:])))
        start += width
    return packed, layout


def _split_trace(packed, layout):
    """Cut a packed block into the per-key arrays, one row per leading index."""
    return {name: packed[:, start:stop].reshape(len(packed), *tail)
            for name, start, stop, tail in layout}


def _append_trace(active, mature, proposed, trace, warmup, fused):
    """Copy this tick's diagnostics into the per-episode buffers.

    Reads only device tensors and host arrays the simulators never touch, so it
    may run while a step is in flight.  ``fused`` keeps the packed block whole
    -- one row copy per world instead of one per key -- and leaves the split to
    the episode's own write using its saved layout.
    """
    packed, layout = _packed_trace(trace)
    host_trace = None if fused else _split_trace(packed, layout)
    for position, slot in enumerate(mature):
        row = active[slot]
        if row["step"] < warmup:
            continue
        row["proposed"].append(proposed[slot].copy())
        if fused:
            # The split at episode-write time must use THIS episode's layout, not
            # whatever the final tick happened to produce.  Constant for every
            # policy shipped here, so pin it once and fail loudly if one varies.
            if row.get("layout") is None:
                row["layout"] = layout
            elif row["layout"] != layout:
                raise ValueError(
                    "per-row trace layout changed mid-episode; a fused trace "
                    "cannot be split with a single layout")
            row["packed"].append(packed[position].copy())
        else:
            for key, value in host_trace.items():
                row["mechanism"].setdefault(key, []).append(value[position].copy())


def _fixed_rows_direct(policy, action_past, state_past, rows, recovery=None):
    """Pad readout and episode state together; commit only the live state rows."""
    count = len(action_past)
    def invoke(actions, states, status):
        if status is None:
            return policy.direct_actions(actions, states)
        return policy.direct_actions(actions, states, recovery=status)
    if not rows:
        return invoke(action_past, state_past, recovery)
    if count > rows:
        raise ValueError(f"readout_rows={rows} is below the {count} live rows")
    index = torch.zeros(rows, dtype=torch.long, device=action_past.device)
    index[:count] = torch.arange(count, device=action_past.device)
    padded_recovery = None if recovery is None else tuple(value[index] for value in recovery)
    output, trace = invoke(action_past[index], state_past[index], padded_recovery)
    if recovery is not None:
        for target, value in zip(recovery, padded_recovery):
            target.copy_(value[:count])
    return output[:count], {key: value[:count] for key, value in trace.items()}


@torch.no_grad()
def _rollouts(config, policy, trajectories, encode, initializers, indices, run_dir, method):
    """Compact active policy rows and refill persistent simulator slots."""
    from .episode_pool import EpisodePool

    evaluation, data = config["evaluation"], config["data"]
    record_trace = evaluation["save_compact_trace"]
    workers = min(int(evaluation["env_workers"]), len(indices))
    readout_rows = int(evaluation.get("readout_rows", 0))
    policy.fixed_shape_readout = readout_rows > 0
    async_reset = bool(evaluation.get("async_reset", readout_rows > 0))
    overlap = bool(evaluation.get("overlap_bookkeeping", readout_rows > 0))
    fused_trace = bool(evaluation.get("fused_trace", readout_rows > 0))
    if async_reset and not readout_rows:
        raise ValueError("async_reset requires pinned readout_rows")
    if readout_rows and readout_rows < workers:
        raise ValueError(f"readout_rows={readout_rows} is below env_workers={workers}")
    seed(evaluation_seed(config, indices[0]))
    warmup, horizon = int(evaluation["warmup_steps"]), int(evaluation["controller_steps"])
    low, high = evaluation["action_clip"]
    device, dtype = policy.device, policy.dtype
    states = torch.zeros(workers, policy.history, policy.state_dim, device=device, dtype=dtype)
    actions = torch.zeros(workers, policy.history, policy.action_dim, device=device, dtype=dtype)
    # Recovery state belongs to episode slots, just like histories.
    use_recovery = bool(getattr(policy, "recovery", False))
    demos = int(getattr(policy, "demo_count", 0)) if use_recovery else 0
    recovery_status = torch.zeros(workers, 3, device=device, dtype=dtype)
    recovery_tabu = torch.zeros(workers, max(demos, 1), device=device, dtype=torch.bool)
    previous = np.zeros((workers, policy.action_dim), dtype=np.float64)
    active, resetting, pending = {}, {}, iter(indices)
    with EpisodePool(config, workers) as pool:
        while True:
            free, assigned = [], []
            for slot in range(workers):
                if slot not in active and slot not in resetting:
                    index = next(pending, None)
                    if index is not None:
                        free.append(slot)
                        assigned.append(index)
            if free:
                started = time.monotonic()
                starts = [initializers[index] for index in assigned]
                seeds = [evaluation_seed(config, index) for index in assigned]
                (pool.reset_async if async_reset else pool.reset)(free, starts, seeds)
                for slot, index, start, episode_seed in zip(free, assigned, starts, seeds):
                    resetting[slot] = (index, start, episode_seed, started)
            # Without async_reset every reset has already completed above.
            admitted = (pool.harvest_resets(block=not active) if async_reset and resetting
                        else sorted(resetting))
            for slot in admitted:
                index, start, episode_seed, started = resetting.pop(slot)
                states[slot] = actions[slot] = 0
                recovery_status[slot] = 0
                recovery_tabu[slot] = False
                previous[slot] = 0
                active[slot] = {"index": index, "step": 0, "success": False,
                                "steps_to_success": None, "started": started}
                if record_trace:
                    active[slot].update(proprio=[], proposed=[], executed=[],
                                        mechanism={}, packed=[], layout=None, query=[])
            if not active:
                if resetting:  # every slot is still constructing its environment
                    continue
                break
            slots = sorted(active)
            frames = pool.frames[slots].reshape(-1, *pool.frames.shape[2:])
            if readout_rows:
                # The readout is padded, but the encoder is not: R3M replays a
                # fixed-size graph per microbatch and falls back to an eager pass
                # for a trailing partial chunk, whose rows do not match the graph's
                # bits.  Parity therefore needs every chunk full.  True for any
                # camera count today (the batch is min(configured, cameras) and the
                # frame count is slots x cameras), so this is a guard, not a fix.
                batch = int(getattr(encode, "batch_size", 0) or 0)
                if batch and len(frames) % batch:
                    raise ValueError(
                        f"readout_rows is set but the encoder batch is ragged: "
                        f"{len(frames)} frames is not a multiple of {batch}, so the "
                        "trailing chunk would take the eager path and break parity")
            features = encode.tensor(frames).reshape(len(slots), -1)
            raw = torch.cat([torch.as_tensor(pool.proprio[slots], device=device, dtype=dtype),
                             features.to(dtype)], 1)
            scaled = policy.scale_state(raw)
            for position, slot in enumerate(slots):
                row = active[slot]
                if row["step"] == 0:
                    demo, demonstration = _warmup_demo(
                        policy, trajectories, raw[position], warmup, data["action_key"])
                    row.update(warmup_demo=demo, warmup_actions=demonstration)
                if record_trace and row["step"] == warmup:
                    row["proprio"].append(pool.proprio[slot].copy())
            # Reproduce the history appended by each scalar action_for call.
            states[slots, :-1] = states[slots, 1:]
            states[slots, -1] = scaled
            actions[slots, :-1] = actions[slots, 1:]
            actions[slots, -1] = (torch.as_tensor(previous[slots], device=device, dtype=dtype)
                                  - policy.action_mean) / policy.action_scale
            if record_trace:
                history = torch.cat([actions[slots, -1], states[slots, -1]], 1).cpu().numpy()
                for position, slot in enumerate(slots):
                    active[slot]["query"].append(history[position].copy())
            mature = [slot for slot in slots if active[slot]["step"] >= policy.history]
            proposed = np.zeros((workers, policy.action_dim), dtype=np.float64)
            if mature:
                recovery = ((recovery_status[mature], recovery_tabu[mature])
                            if use_recovery else None)
                output, trace = _fixed_rows_direct(
                    policy, actions[mature], states[mature], readout_rows,
                    recovery=recovery)
                if use_recovery:
                    recovery_status[mature], recovery_tabu[mature] = recovery
                proposed[mature] = output.cpu().numpy()
                if not np.isfinite(proposed[mature]).all():
                    raise FloatingPointError(f"{method} produced an invalid action")
                if record_trace and not overlap:
                    _append_trace(active, mature, proposed, trace, warmup, fused_trace)
            for slot in slots:
                row = active[slot]
                if row["step"] < warmup:
                    proposed[slot] = row["warmup_actions"][:, row["step"]]
            clipped = np.clip(proposed[slots], low, high)
            if overlap:
                # Every value the bookkeeping needs from pool.frames / .proprio /
                # .success was copied above, before this step was issued; the
                # rest of it reads device tensors the workers never touch.
                pool.step_async(slots, clipped)
                if record_trace and mature:
                    _append_trace(active, mature, proposed, trace, warmup, fused_trace)
                pool.wait_steps()
            else:
                pool.step(slots, clipped)
            for position, slot in enumerate(slots):
                row = active[slot]
                step = row["step"]
                if record_trace and step >= warmup:
                    row["executed"].append(clipped[position].copy())
                    row["proprio"].append(pool.proprio[slot].copy())
                if pool.success[slot] and not row["success"]:
                    row["success"] = True
                    row["steps_to_success"] = max(0, step - warmup + 1)
                previous[slot] = {"proposed": proposed[slot], "executed": clipped[position]}[
                    evaluation.get("action_history", "executed") if step >= warmup else "executed"]
                row["step"] += 1
                finished = (row["step"] == warmup + horizon or row["step"] >= warmup
                            and row["success"] and not evaluation["run_full_horizon_after_success"])
                if not finished:
                    continue
                elapsed = time.monotonic() - row["started"]
                index = row["index"]
                if record_trace:
                    trace_path = run_dir / "traces" / method / f"{index:03d}.npz"
                    if not row["proprio"]:
                        row["proprio"].append(pool.proprio[slot].copy())
                    if row["packed"]:
                        row["mechanism"] = _split_trace(np.stack(row["packed"]), row["layout"])
                    _write_trace(trace_path, policy, row["warmup_demo"], row["proprio"],
                                 row["proposed"], row["executed"], row["mechanism"],
                                 row["query"][warmup - policy.history + 1:])
                yield index, {
                    "seed": evaluation_seed(config, index), "success": row["success"],
                    "steps_to_success": row["steps_to_success"],
                    "controller_steps": max(0, row["step"] - warmup), "warmup_steps": warmup,
                    "warmup_demo": row["warmup_demo"],
                    "seconds": elapsed, "video": None,
                }
                del active[slot]


def _read_episode(path, index, method=None):
    if not path.is_file():
        return None
    row = json.loads(path.read_text())
    if row.get("init_index") != index or (method and row.get("method") != method):
        raise ValueError(f"{path} contains the wrong episode")
    return row


def _write_evaluation(config, run_dir, task):
    evaluation = config["evaluation"]
    indices = range(int(evaluation["init_index_start"]), int(evaluation["init_index_stop"]))
    outcomes = {}
    for method in config["methods"]:
        rows = [_read_episode(Path(run_dir) / "episodes" / method / f"{index:03d}.json",
                              index, method) for index in indices]
        present = [row for row in rows if row is not None]
        outcomes[method] = {
            "expected_episodes": len(rows),
            "completed_episodes": len(present),
            "successes": sum(row["success"] for row in present),
        }
    write_json(Path(run_dir) / "evaluation.json", {
        "task": task["name"],
        "language": task["language"],
        "init_source": task["init_source"],
        "methods": outcomes,
    })


def _rollout_pi05(config, policy, initial_state, episode_seed,
                   trace_path=None, video_path=None):
    evaluation, data = config["evaluation"], config["data"]
    clip_low, clip_high = map(float, evaluation["action_clip"])
    with _episode(config, initial_state, episode_seed, video_path) as session:
        environment, proprio, images, video, started = session
        policy.reset(episode_seed)
        queued = np.empty((0, policy.action_dim))
        proprio_trace, pi05_state_trace = [proprio.copy()], [environment.pi05_state.copy()]
        proposed, executed, inference_ms = [], [], []
        completed = 0
        success, steps_to_success = False, None
        for step in range(int(evaluation["controller_steps"])):
            if success and not evaluation["run_full_horizon_after_success"]:
                break
            if not len(queued):
                queued, elapsed = policy.plan(environment.pi05_state, images)
                inference_ms.append(elapsed)
            action, queued = queued[0], queued[1:]
            clipped = np.clip(action, clip_low, clip_high)
            if trace_path is not None:
                proposed.append(action.copy())
                executed.append(clipped.copy())
            proprio, images, _, done, _ = environment.step(clipped)
            _append_video(video, images, data["cameras"])
            completed += 1
            if trace_path is not None:
                proprio_trace.append(proprio.copy())
                pi05_state_trace.append(environment.pi05_state.copy())
            if done and not success:
                success, steps_to_success = True, step + 1
        elapsed = time.monotonic() - started

    video_output = _finish_video(video_path, success)
    if trace_path is not None:
        _save_trace(trace_path, {
            "proprio": np.asarray(proprio_trace, dtype=np.float32),
            "pi05_state": np.asarray(pi05_state_trace, dtype=np.float32),
            **_action_trace(proposed, executed),
            "inference_ms": np.asarray(inference_ms, dtype=np.float32),
        })
    return {
        "seed": episode_seed,
        "success": success,
        "steps_to_success": steps_to_success,
        "controller_steps": completed,
        "warmup_steps": 0,
        "warmup_demo": None,
        "policy_queries": len(inference_ms),
        "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
        "seconds": elapsed,
        "video": str(video_output) if video_output else None,
    }


def _serial_episodes(config, run_dir, method, initializers, indices, rollout):
    evaluation = config["evaluation"]
    for index in indices:
        trace = Path(run_dir) / "traces" / method / f"{index:03d}.npz"
        yield index, rollout(initializers[index], evaluation_seed(config, index),
            trace_path=trace if evaluation["save_compact_trace"] else None,
            video_path=trace.with_suffix(".mp4") if evaluation.get("save_videos") else None)


def _record_episode(config, run_dir, method, index, row, completed):
    run_dir = Path(run_dir)
    trace = run_dir / "traces" / method / f"{index:03d}.npz"
    if row["video"]:
        row["video"] = str(Path(row["video"]).relative_to(run_dir))
    row.update(method=method, init_index=index,
               trace=str(trace.relative_to(run_dir)) if config["evaluation"]["save_compact_trace"] and trace.is_file() else None)
    write_json(run_dir / "episodes" / method / f"{index:03d}.json", row)
    completed[index] = row
    present = [value for value in completed.values() if value is not None]
    successes = sum(value["success"] for value in present)
    print(f"[evaluate] {method} init {index:02d}: "
          f"{'success' if row['success'] else 'failure'}; {successes}/{len(present)}", flush=True)


def _evaluate_pi05(config, run_dir, requested, shard):
    from .pi05_finetune import FineTunedPi05

    name = config["policy"]["name"]
    methods = [requested] if requested else config["methods"]
    if methods != [name]:
        raise ValueError(f"pi0.5 evaluation requires methods: [{name}]")
    evaluation = config["evaluation"]
    indices = select_shard(range(int(evaluation["init_index_start"]),
                                 int(evaluation["init_index_stop"])), shard)
    completed = {
        index: _read_episode(
            Path(run_dir) / "episodes" / name / f"{index:03d}.json",
            index, name) for index in indices
    }
    task = benchmark_task(config)
    if all(completed.values()):
        if not shard:
            _write_evaluation(config, run_dir, task)
        print("[evaluate] selected episodes already complete", flush=True)
        return
    policy = FineTunedPi05(config, run_dir)
    remaining = [index for index in indices if completed[index] is None]
    for index, row in _serial_episodes(config, run_dir, name, task["initializers"], remaining,
                                       partial(_rollout_pi05, config, policy)):
        _record_episode(config, run_dir, name, index, row, completed)
    if not shard:
        _write_evaluation(config, run_dir, task)


def evaluate(config, run_dir, requested=None, shard=None):
    if config.get("policy", {}).get("name") in RAW_METHODS:
        return _evaluate_pi05(config, run_dir, requested, shard)
    run_dir = Path(run_dir)
    evaluation, data = config["evaluation"], config["data"]
    variants = method_variants(config)
    methods = [requested] if requested else variants
    indices = select_shard(range(int(evaluation["init_index_start"]),
                                 int(evaluation["init_index_stop"])), shard)
    completed = {method: {
        index: _read_episode(run_dir / "episodes" / method / f"{index:03d}.json", index, method)
        for index in indices} for method in methods}
    task = benchmark_task(config)
    if all(all(rows.values()) for rows in completed.values()):
        if not shard:
            _write_evaluation(config, run_dir, task)
        print("[evaluate] selected episodes already complete", flush=True)
        return

    trajectories, _ = load_bank(run_dir / "bank.pt", data)
    batched = int(evaluation.get("env_workers", 1)) > 1
    if not batched and evaluation.get("readout_rows"):
        raise ValueError("readout_rows requires env_workers > 1")
    encode = make_encoder(data, evaluation=batched)
    fit_methods = {}
    for method in methods:
        fit_methods.setdefault(variants[method]["fit"], []).append(method)
    for grouped in fit_methods.values():
        policy = load_policy(config, run_dir, grouped[0], trajectories=trajectories)
        for method in grouped:
            variant = variants[method]
            policy.select_method(variant["base"])
            apply_variant_runtime(policy, variant)
            if policy.recovery:
                if (not policy.batched_direct or int(evaluation.get("env_workers", 1)) < 2
                        or evaluation.get("save_videos")):
                    raise ValueError("certificate recovery requires the batched direct evaluator")
                write_json(run_dir / "recovery" / f"{method}-calibration.json", {
                    "options": {"enabled": True, "settle": policy.recovery_settle,
                                "hold": policy.recovery_hold, "tabu_fraction": policy.recovery_tabu,
                                "attempts": policy.recovery_attempts},
                    "aperture": policy.recovery_aperture,
                    "calibration": getattr(policy, "recovery_calibration", None),
                    "temporal_eligibility": "opening at every valid first-action rate",
                    "distance_pass": "shared with normal retrieval",
                })
            print(f"[evaluate] {method}: recovery "
                  + (f"on (aperture {policy.recovery_aperture}, hold {policy.recovery_hold})"
                     if getattr(policy, "recovery", False) else "off"), flush=True)
            remaining = [index for index in indices if completed[method][index] is None]
            if not remaining:
                continue
            if batched and (not policy.batched_direct or data.get("encoder", "r3m") != "r3m"
                            or evaluation.get("benchmark") != "robosuite"
                            or evaluation.get("save_videos")
                            or int(evaluation["warmup_steps"]) < policy.history):
                raise ValueError("env_workers>1 requires shared-support R3M G-IP direct "
                                 "on Robosuite, complete warmup, and no video")

            episodes = (_rollouts(config, policy, trajectories, encode, task["initializers"],
                                  remaining, run_dir, method) if batched else
                        _serial_episodes(config, run_dir, method, task["initializers"], remaining,
                                         partial(_rollout, config, policy, trajectories, encode)))
            for index, row in episodes:
                _record_episode(config, run_dir, method, index, row, completed[method])
        del policy
        torch.cuda.empty_cache()
    if not shard:
        _write_evaluation(config, run_dir, task)
