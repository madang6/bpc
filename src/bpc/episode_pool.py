"""Persistent simulator processes with zero-copy camera and action exchange."""

import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time
import traceback

import numpy as np


def _views(buffers):
    return {name: np.frombuffer(raw, dtype=dtype).reshape(shape)
            for name, (raw, dtype, shape) in buffers.items()}


def _worker(slot, config, buffers, connection, factory):
    os.environ["MUJOCO_GL"] = "egl"
    environment = None
    try:
        import torch
        from .data import seed

        torch.set_num_threads(1)
        views = _views(buffers)
        if hasattr(os, "sched_getaffinity"):
            cpus = sorted(os.sched_getaffinity(0))
            offset = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")) * len(views["success"])
            os.sched_setaffinity(0, {cpus[(offset + slot) % len(cpus)]})
        if factory is None:
            from .robosuite import RobosuiteEpisode
            factory = RobosuiteEpisode
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                initializer, episode_seed = payload
                episode_seed = int(episode_seed)
                if environment is not None:
                    environment.close()
                    environment = None
                # Constructor randomness is part of our independent-seed protocol.
                seed(episode_seed)
                environment = factory(config)
                if isinstance(initializer, dict):
                    initializer = {**initializer, "_seed": episode_seed}
                proprio, images = environment.reset(initializer)
                views["success"][slot] = False
            elif command == "step":
                proprio, images, _, done, _ = environment.step(
                    views["actions"][slot].copy())
                views["success"][slot] = done
            else:
                raise ValueError(f"unknown episode command {command}")
            np.copyto(views["proprio"][slot], proprio)
            for index, name in enumerate(config["data"]["cameras"]):
                np.copyto(views["frames"][slot, index], images[name])
            connection.send(None)
    except EOFError:
        pass
    except BaseException:
        connection.send(traceback.format_exc())
    finally:
        if environment is not None:
            environment.close()
        connection.close()


class EpisodePool:
    """One Classic Robosuite environment per worker; inactive slots do no work.

    reset(slots, initializers, seeds) and step(slots, actions) update the shared
    .proprio, .images, and .success views in place. Every reset constructs a fresh
    environment after seeding, exactly as a standalone rollout does.

    Every command is issued and awaited separately, so a caller that wants to
    overlap work with the workers can use reset_async/harvest_resets and
    step_async/wait_steps; reset and step keep the blocking send-and-await pair.
    A slot's shared views are only meaningful while none of its commands is in
    flight -- the worker writes them during reset and step.
    """

    def __init__(self, config, workers, timeout=300, *, _factory=None):
        self.workers, self.timeout = int(workers), float(timeout)
        if self.workers < 1 or self.timeout <= 0:
            raise ValueError("workers and timeout must be positive")
        context = mp.get_context("spawn")
        data = config["data"]
        size, cameras = int(data["image_size"]), data["cameras"]
        shapes = {
            "actions": ("d", np.float64, (self.workers, int(config["method"]["action_dim"]))),
            "proprio": ("d", np.float64, (self.workers, 9)),
            "frames": ("B", np.uint8, (self.workers, len(cameras), size, size, 3)),
            "success": ("b", np.bool_, (self.workers,)),
        }
        self._buffers = {name: (context.RawArray(code, int(np.prod(shape))), dtype, shape)
                         for name, (code, dtype, shape) in shapes.items()}
        for name, view in _views(self._buffers).items():
            setattr(self, name, view)
        self.images = {name: self.frames[:, index] for index, name in enumerate(cameras)}
        self._processes, self._connections = [], []
        # connection -> (slot, command, deadline) for every command in flight.
        self._pending = {}
        # Spawn imports NumPy before entering _worker; limit its BLAS threads too.
        thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        previous = {key: os.environ.get(key) for key in thread_vars}
        try:
            os.environ.update(dict.fromkeys(thread_vars, "1"))
            for slot in range(self.workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(
                    slot, config, self._buffers, child, _factory),
                    name=f"robosuite-{slot}", daemon=True)
                process.start()
                child.close()
                self._processes.append(process)
                self._connections.append(parent)
        except BaseException:
            self.close()
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _slots(self, slots):
        slots = np.asarray(slots, dtype=np.int64)
        if (slots.ndim != 1 or len(np.unique(slots)) != len(slots)
                or np.any((slots < 0) | (slots >= self.workers))):
            raise ValueError("slots must be unique valid worker indices")
        return slots

    def _send(self, slots, command, payloads):
        """Dispatch one command per slot and record it as in flight."""
        deadline = time.monotonic() + self.timeout
        try:
            for slot, payload in zip(slots, payloads):
                connection = self._connections[slot]
                if connection in self._pending:
                    raise RuntimeError(f"episode worker {slot} is still running "
                                       f"{self._pending[connection][1]}")
                connection.send((command, payload))
                self._pending[connection] = (int(slot), command, deadline)
        except BaseException:
            self.close()
            raise

    def _collect(self, command, *, block):
        """Retire whichever ``command`` replies have arrived; return their slots.

        ``block`` waits for at least one reply.  Deadlines are checked on every
        call, so a worker that hangs is still reported while other slots run.
        """
        waiting = [connection for connection, (_, name, _) in self._pending.items()
                   if name == command]
        if not waiting:
            return []
        try:
            deadline = min(self._pending[connection][2] for connection in waiting)
            remaining = max(0, deadline - time.monotonic())
            slots = []
            for connection in wait(waiting, timeout=remaining if block else 0):
                slot, name, _ = self._pending.pop(connection)
                try:
                    error = connection.recv()
                except EOFError as error:
                    raise RuntimeError(f"episode worker {slot} exited during {name}") from error
                if error is not None:
                    raise RuntimeError(f"episode worker {slot} failed during {name}:\n{error}")
                slots.append(slot)
            if not slots and time.monotonic() >= deadline:
                stalled = [self._pending[connection][0] for connection in waiting
                           if connection in self._pending]
                raise TimeoutError(f"{command} timed out in workers {stalled}")
            return sorted(slots)
        except BaseException:
            self.close()
            raise

    def _drain(self, command):
        while any(name == command for _, name, _ in self._pending.values()):
            self._collect(command, block=True)

    def reset(self, slots, initializers, seeds):
        self.reset_async(slots, initializers, seeds)
        self._drain("reset")

    def reset_async(self, slots, initializers, seeds):
        """Start these resets without waiting; harvest_resets reports arrivals."""
        slots = self._slots(slots)
        if len(slots) != len(initializers) or len(slots) != len(seeds):
            raise ValueError("one initializer and seed are required per slot")
        self._send(slots, "reset", zip(initializers, seeds))

    def harvest_resets(self, block=False):
        """Slots whose pending reset has completed and may now be stepped."""
        return self._collect("reset", block=block)

    def step(self, slots, actions):
        self.step_async(slots, actions)
        self.wait_steps()

    def step_async(self, slots, actions):
        """Start these steps without waiting; the views are stale until wait_steps."""
        slots = self._slots(slots)
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (len(slots), self.actions.shape[1]):
            raise ValueError("one complete action is required per slot")
        self.actions[slots] = actions
        self._send(slots, "step", [None] * len(slots))

    def wait_steps(self):
        self._drain("step")

    def close(self):
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = time.monotonic() + 5
        for process in self._processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in self._connections:
            connection.close()
        self._connections.clear()
        self._processes.clear()
        self._pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
