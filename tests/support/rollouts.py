"""Deterministic rollout simulators and scheduling fixtures, without test discovery."""
import os
import random
import time

import numpy as np
import torch


EPISODE_CONFIG = {"data": {"image_size": 3, "cameras": ["agentview_rgb", "eye_in_hand_rgb"]},
                  "method": {"action_dim": 7}}


TRAJECTORIES = [{"u": np.asarray([[.2, 2., .7]])}, {"u": np.asarray([[-.4, -2., -.7]])}]
INITIALIZERS = {8: {"value": 0, "success_at": 1}, 23: {"value": 30, "success_at": 3},
                5: {"value": 4, "success_at": 99}}
BASE = {"data": {"cameras": ["agent", "wrist"], "feature_keys": ["a", "b"], "action_key": "u"},
        "evaluation": {"env_workers": 2, "warmup_steps": 2, "controller_steps": 3,
                       "action_clip": [-1, 1], "episode_seed": 100000,
                       "reset_protocol": "independent_seed", "action_history": "executed",
                       "run_full_horizon_after_success": False, "save_compact_trace": True}}


class Encoder:
    def tensor(self, frames):
        return torch.as_tensor(frames[:, 0, 0, :1], dtype=torch.float64)

    def __call__(self, frames):
        return self.tensor(frames).numpy()


class Episode:
    def __init__(self, config):
        self.cameras = config["data"]["cameras"]
        self.actions = []

    def reset(self, initializer):
        self.initializer, self.steps = initializer, 0
        return self.observation()

    def observation(self):
        value = self.initializer["value"] + self.steps
        proprio = np.asarray([value + .25 * sum(float(a[0]) for a in self.actions)])
        images = {camera: np.full((1, 1, 3), value + i, dtype=np.uint8)
                  for i, camera in enumerate(self.cameras)}
        return proprio, images

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        proprio, images = self.observation()
        return proprio, images, 0, self.steps >= self.initializer["success_at"], {}

    def close(self):
        pass


class Pool:
    def __init__(self, config, workers):
        self.config, self.closed = config, False
        self.proprio = np.zeros((workers, 1))
        self.frames = np.zeros((workers, 2, 1, 1, 3), dtype=np.uint8)
        self.success = np.zeros(workers, dtype=bool)
        self.episodes, self.resets = {}, []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def reset(self, slots, initializers, seeds):
        self.resets.append((list(slots), list(seeds)))
        for slot, initializer, seed in zip(slots, initializers, seeds):
            episode = self.episodes[slot] = Episode(self.config)
            proprio, images = episode.reset({**initializer, "_seed": seed})
            self._update(slot, proprio, images, False)

    def _update(self, slot, proprio, images, success):
        self.proprio[slot] = proprio
        self.frames[slot] = np.stack(list(images.values()))
        self.success[slot] = success

    def step(self, slots, actions):
        for slot, action in zip(slots, actions):
            proprio, images, _, success, _ = self.episodes[slot].step(action)
            self._update(slot, proprio, images, success)


class Policy:
    device, dtype = torch.device("cpu"), torch.float64
    history, horizon, state_dim, action_dim, proprio_dim = 2, 2, 3, 1, 1
    method = "bpc"

    def __init__(self):
        self.state_mean = torch.tensor([1., 2., 3.], dtype=self.dtype)
        self.state_scale = torch.tensor([2., 3., 4.], dtype=self.dtype)
        self.action_mean, self.action_scale = torch.tensor([.3]), torch.tensor([2.])
        self.demo_start_state = torch.tensor([[0., 0., 1.], [30., 30., 31.]])
        self.calls = []
        self.reset()

    def scale_state(self, state):
        return (torch.as_tensor(state) - self.state_mean) / self.state_scale

    def reset(self):
        self._state_history, self._action_history = [], []

    def direct_actions(self, actions, states):
        self.calls.append((actions.clone(), states.clone()))
        output = 1.4 + .01 * states[:, -1, :1] + .1 * actions.sum((1, 2))[:, None]
        predicted = output[:, None].expand(-1, self.horizon, -1).clone()
        prior = predicted.flatten(1).clone()
        prior[:, 0] += .5
        return output, {
            "base_z": prior[:, :1], "predicted_action_z": predicted,
            "predicted_proprio_z": states[:, -1:, :1].expand(-1, self.horizon, -1),
        }

    def action_for(self, state, previous):
        self._state_history.append(self.scale_state(state))
        self._action_history.append(torch.as_tensor(previous))
        if len(self._state_history) <= self.history:
            return previous.copy()
        actions = (torch.stack(self._action_history[-self.history:])
                   - self.action_mean) / self.action_scale
        states = torch.stack(self._state_history[-self.history:])
        output, trace = self.direct_actions(actions[None], states[None])
        self.last_trace = {key: value[0].numpy() for key, value in trace.items()}
        return output[0].numpy()


class FakeEpisode:
    def __init__(self, config):
        self.constructor = np.random.random() + random.random()
        self.cameras = config["data"]["cameras"]
        self.state = np.zeros(9, dtype=np.float64)

    def _observation(self):
        images = {name: np.arange(27, dtype=np.uint8).reshape(3, 3, 3) + index
                  for index, name in enumerate(self.cameras)}
        return self.state.copy(), images

    def reset(self, initializer):
        if initializer == "raise":
            raise ValueError("intentional simulator failure")
        if initializer == "exit":
            os._exit(19)
        if initializer == "wait":
            time.sleep(30)
        np.random.seed(int(initializer))
        self.state[:] = self.constructor
        self.state[-1] = np.random.random()
        return self._observation()

    def step(self, action):
        self.state[:7] = action
        proprio, images = self._observation()
        return proprio, images, 0.0, bool(action[0] > 0), {}

    def close(self):
        pass


class SchedulePool(Pool):
    """A fake pool with the async API, a late reset, and an in-flight hazard.

    ``delay`` is how many harvests a slot's reset takes to land (slot 0 lands at
    once, so the loop always has something to step).  ``hazard`` fills the shared
    views with garbage while a step is in flight, exactly as a real worker would
    be part-way through writing them; any bookkeeping that reads the pool during
    that window is then visible as a difference.
    """

    def __init__(self, config, workers, delay=0, hazard=False):
        super().__init__(config, workers)
        self.delay, self.hazard = delay, hazard
        self.staged, self.inflight, self.harvests = {}, None, 0
        self.admitted, self.windows = [], []

    def reset_async(self, slots, initializers, seeds):
        for slot, initializer, seed in zip(slots, initializers, seeds):
            self.staged[slot] = [self.delay * (slot % 2), initializer, seed]

    def harvest_resets(self, block=False):
        if not self.staged:
            return []
        self.harvests += 1
        if block and not any(row[0] <= 0 for row in self.staged.values()):
            soonest = min(row[0] for row in self.staged.values())
            for row in self.staged.values():
                row[0] -= soonest
        due = sorted(slot for slot, row in self.staged.items() if row[0] <= 0)
        for row in self.staged.values():
            row[0] -= 1
        for slot in due:
            _, initializer, seed = self.staged.pop(slot)
            super().reset([slot], [initializer], [seed])
            self.admitted.append((self.harvests, slot))
        return due

    def step_async(self, slots, actions):
        self.inflight = (list(slots), np.array(actions, dtype=np.float64))
        if self.hazard:
            for slot in slots:  # what a worker is doing to the views right now
                self.proprio[slot] = np.nan
                self.frames[slot] = 255
                self.success[slot] = True

    def wait_steps(self):
        slots, actions = self.inflight
        self.inflight = None
        self.windows.append(np.array(self.proprio, copy=True))  # what an early read saw
        super().step(slots, actions)
