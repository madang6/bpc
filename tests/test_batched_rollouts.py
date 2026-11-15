import copy
import tempfile
import unittest
from tests.suites import integration
from itertools import product
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np


from dpc_nldl.evaluate import _rollout, _rollouts, _write_trace
from tests.support.rollouts import BASE, INITIALIZERS, TRAJECTORIES, Encoder, Episode, Policy, Pool


@integration
class BatchedRolloutsTest(unittest.TestCase):
    def test_saved_history_preserves_float64_bits(self):
        history = np.array([[np.nextafter(1., 2.), -1. - 2**-40, 1e-200, 0.]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "000.npz"
            _write_trace(path, Policy(), 0, [], [], [], {}, history)
            with np.load(path) as trace:
                self.assertEqual(trace["query_history_z"].dtype, np.float64)
                np.testing.assert_array_equal(trace["query_history_z"], history)

    def setUp(self):
        self.config = copy.deepcopy(BASE)
        self.trajectories = copy.deepcopy(TRAJECTORIES)
        self.initializers = copy.deepcopy(INITIALIZERS)

    def test_batched_refill_matches_scalar_outcomes_and_every_trace_field(self):
        for history, full_horizon, warmup, record in product(("executed", "proposed"),
                                                            (False, True), (2, 3), (False, True)):
            with self.subTest(history=history, full_horizon=full_horizon, warmup=warmup, trace=record):
                self.config["evaluation"].update(
                    action_history=history, run_full_horizon_after_success=full_horizon,
                    warmup_steps=warmup, save_compact_trace=record)
                with tempfile.TemporaryDirectory() as temporary:
                    root, policy = Path(temporary), Policy()
                    pool = Pool(self.config, 2)
                    with (patch("dpc_nldl.episode_pool.EpisodePool", return_value=pool),
                          nullcontext() if record else patch("dpc_nldl.evaluate._packed_trace",
                              side_effect=AssertionError("disabled trace copied to host"))):
                        actual = dict(_rollouts(self.config, policy, self.trajectories,
                                               Encoder(), self.initializers, [8, 23, 5],
                                               root / "batched", "test"))
                    self.assertTrue(pool.closed)
                    self.assertEqual(pool.resets[0], ([0, 1], [100008, 100023]))
                    self.assertEqual(pool.resets[1], ([0], [100005]))
                    for index in (8, 23, 5):
                        reference_path = root / "scalar" / f"{index:03d}.npz"
                        with patch("dpc_nldl.evaluate.make_episode", side_effect=Episode):
                            expected = _rollout(self.config, Policy(), self.trajectories,
                                                Encoder(), self.initializers[index],
                                                100000 + index, reference_path if record else None)
                        for key in expected.keys() - {"seconds"}:
                            self.assertEqual(actual[index][key], expected[key], key)
                        if not record:
                            self.assertFalse((root / 'batched/traces').exists())
                            continue
                        with (np.load(reference_path) as reference,
                              np.load(root / "batched/traces/test" / f"{index:03d}.npz")
                              as batched):
                            self.assertEqual(set(reference), set(batched))
                            for trace in (reference, batched):
                                self.assertEqual(trace["query_history_z"].dtype, np.float64)
                            for key in reference:
                                np.testing.assert_allclose(batched[key], reference[key],
                                                           rtol=0, atol=0, err_msg=key)

    def test_early_success_still_executes_entire_warmup(self):
        self.config["evaluation"]["save_compact_trace"] = False
        pool, policy = Pool(self.config, 1), Policy()
        with patch("dpc_nldl.episode_pool.EpisodePool", return_value=pool):
            rows = list(_rollouts(self.config, policy, self.trajectories, Encoder(),
                                  self.initializers, [8], Path("unused"), "test"))
        self.assertEqual(rows[0][1]["controller_steps"], 0)
        self.assertEqual(rows[0][1]["steps_to_success"], 0)
        np.testing.assert_array_equal(pool.episodes[0].actions, [[.2], [1.]])
        self.assertEqual(policy.calls, [])

    def test_invalid_policy_output_closes_pool(self):
        self.config["evaluation"]["save_compact_trace"] = False
        pool, policy = Pool(self.config, 1), Policy()
        original = policy.direct_actions

        def invalid(*args):
            output, trace = original(*args)
            return output.fill_(float("nan")), trace

        with (patch("dpc_nldl.episode_pool.EpisodePool", return_value=pool),
              patch.object(policy, "direct_actions", side_effect=invalid)):
            with self.assertRaisesRegex(FloatingPointError, "invalid action"):
                list(_rollouts(self.config, policy, self.trajectories, Encoder(),
                               self.initializers, [5], Path("unused"), "test"))
        self.assertTrue(pool.closed)


if __name__ == "__main__":
    unittest.main()
