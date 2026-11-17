"""Native input contracts, policy dimensions, and episode outcomes."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bpc.bpc import Policy
from bpc.data import benchmark_task
from bpc.dp3_dexart import policy_observation
from bpc.evaluate import _rollout
from bpc.pipeline import load_config, task_configs, temporal_method
from tests.support.policies import fisher_recipe


PROJECT = Path(__file__).resolve().parents[1]


class ScriptedEpisode:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.steps = 0
        self.actions = []
        self.closed = False
        self.total_reward = 0.0
        self.success_steps = 0
        self.final_success = False
        self.terminated = False
        self.truncated = False

    def reset(self, initializer):
        self.initializer = initializer
        return np.zeros(2, dtype=np.float64), {}

    def step(self, action):
        if self.steps >= len(self.outcomes):
            raise AssertionError("native episode was stepped after its scripted end")
        self.actions.append(np.asarray(action).copy())
        success, done, reward = self.outcomes[self.steps]
        self.steps += 1
        self.total_reward += reward
        self.success_steps += int(success)
        self.final_success = success
        self.terminated = done
        return (np.full(2, self.steps, dtype=np.float64), {}, reward, done,
                {"success": success, "episode_return": self.total_reward,
                 "terminated": done, "truncated": False})

    def close(self):
        self.closed = True


class ScriptedPolicy:
    action_dim = 1
    method = "bpc"

    def __init__(self):
        self.previous = []

    def reset(self):
        self.previous.clear()

    def action_for(self, state, previous):
        self.previous.append(np.asarray(previous).copy())
        return np.array([0.25], dtype=np.float64)


class DexterousRolloutTest(unittest.TestCase):
    def rollout(self, benchmark, outcomes, *, warmup=0, full_horizon=False):
        episode = ScriptedEpisode(outcomes)
        policy = ScriptedPolicy()
        config = {
            "data": {"cameras": [], "feature_keys": [], "action_key": "Uro"},
            "evaluation": {"benchmark": benchmark, "action_clip": [-1, 1],
                           "warmup_steps": warmup, "controller_steps": 5,
                           "run_full_horizon_after_success": full_horizon,
                           "action_history": "executed"},
        }
        warmup_actions = np.full((1, warmup), 0.5, dtype=np.float64)
        with (patch("bpc.evaluate.make_episode", return_value=episode),
              patch("bpc.evaluate._warmup_demo",
                    return_value=(0, warmup_actions))):
            result = _rollout(config, policy, [], None, {"seed": 17}, 17)
        self.assertTrue(episode.closed)
        return result, episode, policy

    def test_native_termination_after_failure_is_not_success(self):
        for benchmark in ("adroit", "dexart"):
            with self.subTest(benchmark=benchmark):
                result, episode, _ = self.rollout(
                    benchmark, [(False, False, 1.0), (False, True, 2.0)])
                self.assertEqual(episode.steps, 2)
                self.assertFalse(result["success"])
                self.assertFalse(result["success_final"])
                self.assertFalse(result["success_any"])
                self.assertEqual(result["success_steps"], 0)
                self.assertEqual(result["return"], 3.0)
                self.assertTrue(result["terminated"])
                self.assertFalse(result["truncated"])

    def test_native_transient_success_does_not_stop_or_count_as_final(self):
        result, episode, _ = self.rollout(
            "adroit", [(True, False, 1.0), (False, False, 2.0),
                       (False, True, 3.0)])
        self.assertEqual(episode.steps, 3)
        self.assertFalse(result["success"])
        self.assertFalse(result["success_final"])
        self.assertTrue(result["success_any"])
        self.assertEqual(result["success_steps"], 1)
        self.assertEqual(result["return"], 6.0)

    def test_native_final_success_is_reported(self):
        result, episode, _ = self.rollout(
            "dexart", [(False, False, 1.0), (True, True, 2.0)])
        self.assertEqual(episode.steps, 2)
        self.assertTrue(result["success"])
        self.assertTrue(result["success_final"])
        self.assertTrue(result["success_any"])
        self.assertEqual(result["success_steps"], 1)
        self.assertEqual(result["steps_to_success"], 2)

    def test_native_warmup_termination_prevents_controller_steps(self):
        result, episode, policy = self.rollout(
            "adroit", [(False, True, 1.0)], warmup=2)
        self.assertEqual(episode.steps, 1)
        self.assertEqual(len(policy.previous), 1)
        self.assertEqual(result["controller_steps"], 0)
        self.assertFalse(result["success"])
        self.assertEqual(result["return"], 1.0)

    def test_robosuite_keeps_done_as_success(self):
        result, episode, _ = self.rollout(
            "robosuite", [(False, False, 1.0), (False, True, 2.0)])
        self.assertEqual(episode.steps, 2)
        self.assertTrue(result["success"])
        self.assertEqual(result["steps_to_success"], 2)


class DexterousInputTest(unittest.TestCase):
    def test_all_native_tasks_resolve_shapes_and_action_rates(self):
        expected = {
            "adroit": {"door": (88, 28, 190), "hammer": (88, 26, 190),
                       "pen": (88, 24, 190)},
            "dexart": {"toilet": (96, 22, 240), "faucet": (96, 22, 240),
                       "laptop": (96, 22, 240)},
        }
        for suite, tasks in expected.items():
            config = load_config(PROJECT / f"configs/{suite}_pointnet.json")
            resolved = task_configs(config, "/tmp/dexterous-test")
            self.assertEqual({c["evaluation"]["task"] for c, _, _ in resolved}, set(tasks))
            for scoped, _, label in resolved:
                name = scoped["evaluation"]["task"]
                with self.subTest(task=label):
                    self.assertEqual((scoped["method"]["state_dim"],
                                      scoped["method"]["action_dim"],
                                      scoped["evaluation"]["controller_steps"]),
                                     tasks[name])
                    self.assertEqual(scoped["method"]["scaling"]["proprio_dim"],
                                     tasks[name][0] - 64)
                    self.assertEqual(scoped["data"]["cameras"], ["point_cloud"])
                    self.assertEqual(scoped["data"]["feature_keys"], ["dp3_pointnet"])
                    self.assertEqual(scoped["data"]["dp3_pointnet"]["device"], "cuda")
                    rates = temporal_method(scoped["method"])["g"]["rate_scaled_action_dims"]
                    self.assertEqual(rates, [] if suite == "adroit" else list(range(6)))

    def test_task_catalogs_import_without_optional_simulators(self):
        with patch.dict(sys.modules, {"gymnasium_robotics": None, "sapien": None}):
            for suite in ("adroit", "dexart"):
                config = load_config(PROJECT / f"configs/{suite}_pointnet.json")
                scoped = task_configs(config, "/tmp/dexterous-test")[0][0]
                task = benchmark_task(scoped)
                self.assertEqual(task["name"], scoped["evaluation"]["task"])
                self.assertEqual(len(task["initializers"]), 30)

    def test_dexart_cloud_uses_xyz_and_excludes_labels(self):
        observed = np.zeros((1024, 6), dtype=np.float32)
        observed[:, :3] = [1, 2, 3]
        observed[:, 3:] = 999
        imagined = np.zeros((96, 7), dtype=np.float32)
        imagined[:, :3] = [4, 5, 6]
        imagined[:, 3:] = -999
        native = {"state": np.arange(32, dtype=np.float32),
                  "instance_1-point_cloud": observed,
                  "imagination_robot": imagined,
                  "instance_1-rgb": np.full((84, 84, 3), 888, dtype=np.float32)}
        state, images = policy_observation(native, "faucet")
        np.testing.assert_array_equal(state, native["state"])
        self.assertEqual(set(images), {"point_cloud"})
        self.assertEqual(images["point_cloud"].shape, (1120, 3))
        np.testing.assert_array_equal(images["point_cloud"][0], [1, 2, 3])
        np.testing.assert_array_equal(images["point_cloud"][-1], [4, 5, 6])

    def test_cpu_policy_fits_and_queries_adroit_action_widths(self):
        for action_dim in (24, 28):
            with self.subTest(action_dim=action_dim):
                rng = np.random.default_rng(2)
                bank = [{"Xro": rng.normal(size=(24, 12)),
                         "dp3_pointnet": rng.normal(size=(64, 12)),
                         "Uro": rng.uniform(-1, 1, size=(action_dim, 12))}
                        for _ in range(3)]
                data = {"state_key": "Xro", "action_key": "Uro",
                        "feature_keys": ["dp3_pointnet"], "encoder": "dp3_pointnet",
                        "dp3_pointnet": {"checkpoint_sha256": "f" * 64}}
                method = fisher_recipe()
                method.update(state_dim=88, action_dim=action_dim)
                method['g']['rate_scaled_action_dims'] = []
                method['g']['lda']['calibration_holdout'] = 8
                method["scaling"]["proprio_dim"] = 24
                method["rff"]["features"] = 16
                policy = Policy(bank, data, method)
                policy.fit()
                self.assertEqual(policy.g_lookup_stats['anchors'], 16)
                self.assertEqual(policy.g_lookup_stats['support_calibration_histories'], 8)
                self.assertEqual(tuple(policy.residual_weights.shape), (16, action_dim))
                policy.reset()
                for _ in range(4):
                    action = policy.action_for(np.zeros(88), np.zeros(action_dim))
                    self.assertEqual(action.shape, (action_dim,))
                    self.assertTrue(np.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
