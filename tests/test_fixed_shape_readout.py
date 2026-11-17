"""evaluation.readout_rows: fixed-shape, row-decoupled policy readout (CPU logic checks)."""
import tempfile
import unittest
from tests.suites import integration
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


from bpc.evaluate import _fixed_rows_direct, _rollouts
from tests.support.policies import exact_policy
from tests.support.rollouts import Encoder, Policy, Pool


def _padded(index, rows):
    """Row indices the evaluator feeds the readout: live rows, then copies of row 0."""
    return torch.cat([index, torch.zeros(rows - len(index), dtype=torch.long)])


class FixedShapeSolveTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(3)
        self.design = torch.randn(6, 12, 7, generator=generator, dtype=torch.float64)
        self.query = torch.randn(6, 12, generator=generator, dtype=torch.float64)
        # Repeated supports make the gram singular: rows 1 and 4 take the fallback.
        self.design[1, :, 3] = self.design[1, :, 2]
        self.design[4] = self.design[4, :, :1]
        # Nearly dependent supports converge later than the rest.
        self.design[2, :, 5] = self.design[2, :, 4] + 1e-7 * self.design[2, :, 6]

    def test_fixed_rows_do_not_depend_on_batch_mates(self):
        policy = exact_policy()
        policy.fixed_shape_readout = True
        full = policy._solve_affine(self.design, self.query)
        for live in ([0, 1, 2], [2], [4, 5], [3, 1, 5, 0]):
            index = _padded(torch.tensor(live), 6)
            part = policy._solve_affine(self.design[index], self.query[index])
            torch.testing.assert_close(part[:len(live)], full[live], rtol=0, atol=0)
        # Every row solves the affine problem; fallback rows stay on the simplex plane.
        torch.testing.assert_close(full.sum(1), torch.ones(6, dtype=torch.float64))
        residual = self.query - (self.design @ full[:, :, None]).squeeze(2)
        gradient = self.design.transpose(1, 2) @ residual[:, :, None]
        self.assertLess(float((gradient - gradient.mean(1, keepdim=True)).abs().max()), 1e-6)

    def test_fixed_rows_match_the_default_solve(self):
        policy = exact_policy()
        default = policy._solve_affine(self.design, self.query)
        policy.fixed_shape_readout = True
        fixed = policy._solve_affine(self.design, self.query)
        torch.testing.assert_close(fixed, default, rtol=1e-9, atol=1e-9)
        # Off by default, and the flag never touches float32 queries' fallback selection.
        self.assertFalse(exact_policy().fixed_shape_readout)

    def test_all_rows_singular_uses_fallback_without_raising(self):
        policy = exact_policy()
        policy.fixed_shape_readout = True
        design = self.design[[4, 4, 4]]
        output = policy._solve_affine(design, self.query[[4, 4, 4]])
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.sum(1), torch.ones(3, dtype=torch.float64))


class FixedRowsDirectTest(unittest.TestCase):
    def test_pads_with_row_zero_and_slices_outputs(self):
        policy = Policy()
        generator = torch.Generator().manual_seed(5)
        actions = torch.randn(3, 2, 1, generator=generator, dtype=torch.float64)
        states = torch.randn(3, 2, 3, generator=generator, dtype=torch.float64)
        expected, expected_trace = policy.direct_actions(actions, states)
        output, trace = _fixed_rows_direct(policy, actions, states, 5)
        fed_actions, fed_states = policy.calls[-1]
        self.assertEqual(tuple(fed_actions.shape), (5, 2, 1))
        torch.testing.assert_close(fed_actions[3:], actions[:1].expand(2, -1, -1))
        torch.testing.assert_close(fed_states[3:], states[:1].expand(2, -1, -1))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(set(trace), set(expected_trace))
        for key in trace:
            torch.testing.assert_close(trace[key], expected_trace[key], rtol=0, atol=0)
        output, trace = _fixed_rows_direct(policy, actions, states, 3)
        self.assertEqual(tuple(policy.calls[-1][0].shape), (3, 2, 1))
        with self.assertRaisesRegex(ValueError, "readout_rows=2"):
            _fixed_rows_direct(policy, actions, states, 2)

    def test_zero_rows_is_the_compacted_call(self):
        policy = Policy()
        actions, states = torch.ones(2, 2, 1, dtype=torch.float64), torch.ones(2, 2, 3, dtype=torch.float64)
        _fixed_rows_direct(policy, actions, states, 0)
        self.assertEqual(tuple(policy.calls[-1][0].shape), (2, 2, 1))


@integration
class FixedShapeRolloutsTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "data": {"cameras": ["agent", "wrist"], "feature_keys": ["a", "b"],
                     "action_key": "u"},
            "evaluation": {"env_workers": 2, "warmup_steps": 2, "controller_steps": 3,
                           "action_clip": [-1, 1], "episode_seed": 100000,
                           "reset_protocol": "independent_seed", "action_history": "executed",
                           "run_full_horizon_after_success": True, "save_compact_trace": True},
        }
        self.trajectories = [{"u": np.asarray([[.2, 2., .7]])},
                             {"u": np.asarray([[-.4, -2., -.7]])}]
        self.initializers = {8: {"value": 0, "success_at": 1},
                             23: {"value": 30, "success_at": 3},
                             5: {"value": 4, "success_at": 99}}

    def _run(self, rows, root):
        config = {**self.config, "evaluation": {**self.config["evaluation"], "readout_rows": rows, "async_reset": False,
                                                    "overlap_bookkeeping": False, "fused_trace": False}}
        pool, policy = Pool(config, 2), Policy()
        with patch("bpc.episode_pool.EpisodePool", return_value=pool):
            outcomes = dict(_rollouts(config, policy, self.trajectories, Encoder(),
                                      self.initializers, [8, 23, 5], root, "test"))
        self.assertTrue(pool.closed)
        return outcomes, policy

    def test_readout_rows_pads_every_call_and_keeps_outcomes_and_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compact, policy = self._run(0, root / "compact")
            self.assertFalse(policy.fixed_shape_readout)
            self.assertEqual({len(a) for a, _ in policy.calls}, {1, 2})
            padded, policy = self._run(4, root / "padded")
            self.assertTrue(policy.fixed_shape_readout)
            self.assertEqual({len(a) for a, _ in policy.calls}, {4})
            for index in (8, 23, 5):
                for key in compact[index].keys() - {"seconds"}:
                    self.assertEqual(padded[index][key], compact[index][key], key)
                with (np.load(root / "compact/traces/test" / f"{index:03d}.npz") as expected,
                      np.load(root / "padded/traces/test" / f"{index:03d}.npz") as actual):
                    self.assertEqual(set(expected), set(actual))
                    for key in expected:
                        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def test_readout_rows_below_live_rows_raises(self):
        self.config["evaluation"]["save_compact_trace"] = False
        with self.assertRaisesRegex(ValueError, "readout_rows=1"):
            self._run(1, Path("unused"))


if __name__ == "__main__":
    unittest.main()
