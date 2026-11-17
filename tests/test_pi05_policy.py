import tempfile
import unittest
from tests.suites import integration
from pathlib import Path
from types import ModuleType
from unittest import mock

import numpy as np

from bpc.evaluate import _episode, _rollout_pi05
from bpc.pi05_finetune import FineTunedPi05, _configure_image_keys, pi05_frames
from bpc.robosuite import pi05_state
from tests.support.rollouts import BASE, Episode


class Pi05PolicyTest(unittest.TestCase):
    @integration
    def test_rollout_resources_close_on_reset_inference_or_video_failure(self):
        config = {"data": {"cameras": ["camera"]}}
        for failure in ("reset", "inference", "video", None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                environment, video = mock.Mock(), mock.Mock()
                environment.reset.return_value = (np.zeros(9), {"camera": np.zeros((2, 2, 3))})
                if failure == "reset":
                    environment.reset.side_effect = RuntimeError("reset")
                if failure == "video":
                    video.close.side_effect = RuntimeError("video")
                with (mock.patch("bpc.evaluate.seed"),
                      mock.patch("bpc.evaluate.make_episode", return_value=environment),
                      mock.patch("imageio.get_writer", return_value=video)):
                    try:
                        with _episode(config, {"state": 1}, 17, Path(folder) / "trial.mp4"):
                            if failure == "inference":
                                raise RuntimeError("inference")
                    except RuntimeError as error:
                        self.assertEqual(str(error), failure)
                    else:
                        self.assertIsNone(failure)
                environment.close.assert_called_once()
                environment.reset.assert_called_once_with({"state": 1, "_seed": 17})
                self.assertEqual(video.close.call_count, int(failure != "reset"))

    @integration
    def test_optional_trace_preserves_actions_and_episode_summary(self):
        for full_horizon in (False, True):
            config = {**BASE, "evaluation": {**BASE["evaluation"],
                      "run_full_horizon_after_success": full_horizon}}
            outcomes, actions = [], []
            with self.subTest(full_horizon=full_horizon), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "trace.npz"
                for record in (False, True):
                    environment = Episode(config)
                    environment.pi05_state = np.zeros(8)
                    policy = mock.Mock(action_dim=1)
                    policy.plan.return_value = (np.array([[np.pi], [.75]]), 12.5)
                    with mock.patch("bpc.evaluate.make_episode", return_value=environment):
                        row = _rollout_pi05(config, policy, {"value": 0, "success_at": 1}, 17,
                                            trace_path=path if record else None)
                    outcomes.append({key: value for key, value in row.items() if key != "seconds"})
                    actions.append(environment.actions)
                    self.assertEqual(path.exists(), record)
                self.assertEqual(outcomes[0], outcomes[1])
                self.assertEqual(outcomes[0]["controller_steps"], 3 if full_horizon else 1)
                np.testing.assert_array_equal(actions[0], actions[1])
                with np.load(path) as trace:
                    for name in ("proposed_action", "executed_action"):
                        self.assertEqual(trace[name].dtype, np.float64)
                    self.assertEqual(trace["proposed_action"][0, 0], np.pi)
                    np.testing.assert_array_equal(trace["executed_action"], actions[0])

    @integration
    def test_finetuning_configures_openpi_image_keys_without_leaking(self):
        model = ModuleType("openpi.models.model")
        model.IMAGE_KEYS = ("base", "left", "right")
        model.preprocess_observation = mock.Mock()
        models = ModuleType("openpi.models")
        models.model = model
        openpi = ModuleType("openpi")
        openpi.models = models
        modules = {"openpi": openpi, "openpi.models": models,
                   "openpi.models.model": model}
        with mock.patch.dict("sys.modules", modules):
            _configure_image_keys(("base", "left"))
            model.preprocess_observation(None, "observation", train=True)
            _configure_image_keys(None)
            model.preprocess_observation(None, "observation")
        self.assertEqual([call.kwargs for call in
                          model._bpc_preprocess_observation.call_args_list], [
            {"train": True, "image_keys": ("base", "left")},
            {"image_keys": model.IMAGE_KEYS},
        ])

    def test_state_matches_openpi_libero_runtime(self):
        obs = {
            "robot0_eef_pos": np.arange(6).reshape(2, 3),
            "robot0_eef_quat": np.array([
                [0, 0, 0, 1], [0, 0, np.sqrt(.5), np.sqrt(.5)]]),
            "robot0_gripper_qpos": np.array([[.02, -.02], [.01, -.01]]),
        }
        state = pi05_state(obs)
        np.testing.assert_array_equal(state[:, :3], obs["robot0_eef_pos"])
        np.testing.assert_allclose(state[:, 3:6], [[0, 0, 0], [0, 0, np.pi / 2]])
        np.testing.assert_allclose(state[:, 6:], obs["robot0_gripper_qpos"])

    def test_cameras_match_openpi_libero_rotation(self):
        raw = np.arange(18).reshape(2, 3, 3)
        upright = raw[::-1]
        frames = pi05_frames({"agentview_rgb": upright,
                              "eye_in_hand_rgb": upright + 100})
        np.testing.assert_array_equal(frames, np.stack([
            raw[::-1, ::-1], (raw + 100)[::-1, ::-1]]))

    def test_episode_noise_is_shard_order_invariant(self):
        policy = FineTunedPi05.__new__(FineTunedPi05)
        policy.action_horizon = 10
        policy.reset(123)
        first = policy._noise()
        second = policy._noise()
        policy.reset(123)
        repeated = policy._noise()
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, second))

    def test_finetuned_adapter_passes_exact_state_images_and_noise(self):
        class Policy:
            def infer(inner, inputs, noise):
                self.assertEqual(inputs["observation/state"].shape, (8,))
                self.assertEqual(inputs["observation/image"].shape, (224, 224, 3))
                self.assertEqual(noise.shape, (10, 32))
                return {"actions": np.zeros((10, 7)),
                        "policy_timing": {"infer_ms": 12.5}}

        policy = FineTunedPi05.__new__(FineTunedPi05)
        policy.policy = Policy()
        policy._configure({"data": {"prompt": "do it"}}, 10, 5)
        policy.reset(123)
        images = {name: np.zeros((224, 224, 3), dtype=np.uint8)
                  for name in ("agentview_rgb", "eye_in_hand_rgb")}
        actions, elapsed = policy.plan(np.zeros(8), images)
        self.assertEqual(actions.shape, (5, 7))
        self.assertEqual(elapsed, 12.5)


if __name__ == "__main__":
    unittest.main()
