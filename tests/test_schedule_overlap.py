"""evaluation.async_reset / overlap_bookkeeping / fused_trace: schedule-only speedups.

Every check here compares an optimized schedule against the parent schedule on
the same fake simulators and demands identical bytes: these flags may reorder
work, never change arithmetic.
"""
import tempfile
import time
import unittest
from tests.suites import integration
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


from dpc_nldl.episode_pool import EpisodePool
from dpc_nldl.evaluate import _packed_trace, _split_trace, _rollouts
from tests.support.rollouts import (
    BASE, INITIALIZERS, TRAJECTORIES, EPISODE_CONFIG as CONFIG, Encoder, FakeEpisode,
    Policy, SchedulePool,
)


def _outcome(config, flags, pool, root):
    """Run one evaluation over a fake pool and return its episodes and traces."""
    config = {**config, "evaluation": {**config["evaluation"], **flags}}
    with patch("dpc_nldl.episode_pool.EpisodePool", return_value=pool):
        episodes = dict(_rollouts(config, Policy(), TRAJECTORIES, Encoder(),
                                  INITIALIZERS, [8, 23, 5], root, "test"))
    traces = {}
    for path in sorted((root / "traces" / "test").glob("*.npz")):
        with np.load(path) as trace:
            traces[path.name] = {key: (trace[key].dtype, trace[key].shape,
                                       trace[key].tobytes()) for key in trace.files}
    return {index: {key: value for key, value in row.items() if key != "seconds"}
            for index, row in episodes.items()}, traces


@integration
class ScheduleFlagTest(unittest.TestCase):
    def _reference(self, folder, flags=None):
        return _outcome(BASE, dict(flags or {}), SchedulePool(BASE, 2), Path(folder) / "reference")

    def test_each_flag_and_all_three_reproduce_the_blocking_schedule(self):
        """Every flag combination must produce the same episodes and trace bytes."""
        # async_reset requires a pinned readout_rows, so every arm here (including
        # the reference) carries it: readout_rows changes the arithmetic by design,
        # and the flags must be compared against a baseline that already has it.
        pinned = {"readout_rows": 2, "async_reset": False,
                  "overlap_bookkeeping": False, "fused_trace": False}
        combinations = [{"readout_rows": 2}, {**pinned, "async_reset": True},
                        {**pinned, "overlap_bookkeeping": True},
                        {**pinned, "fused_trace": True},
                        {**pinned, "async_reset": True, "overlap_bookkeeping": True,
                         "fused_trace": True}]
        with tempfile.TemporaryDirectory() as folder:
            expected_episodes, expected_traces = self._reference(folder, pinned)
            self.assertEqual(len(expected_traces), 3)
            for position, flags in enumerate(combinations):
                with self.subTest(**flags):
                    pool = SchedulePool(BASE, 2, delay=3 if flags.get("async_reset", True) else 0,
                                        hazard=flags.get("overlap_bookkeeping", True))
                    episodes, traces = _outcome(BASE, flags, pool,
                                                Path(folder) / f"variant{position}")
                    self.assertEqual(episodes, expected_episodes)
                    self.assertEqual(traces, expected_traces)

    def test_async_reset_without_pinned_readout_rows_is_refused(self):
        """The pairing is a correctness precondition, so it must fail loudly.

        Certification showed an unpadded readout diverges at the first controller
        tick when admission changes the live-row count, so the combination must
        never run silently.
        """
        with tempfile.TemporaryDirectory() as folder:
            for flags in ({"async_reset": True}, {"async_reset": True, "readout_rows": 0}):
                with self.subTest(**flags), self.assertRaises(ValueError) as caught:
                    _outcome(BASE, flags, SchedulePool(BASE, 2), Path(folder) / "refused")
                self.assertIn("readout_rows", str(caught.exception))

    def test_readout_rows_below_env_workers_is_refused(self):
        """Padding narrower than the live rows cannot certify anything."""
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError) as caught:
                _outcome(BASE, {"readout_rows": 1, "async_reset": True},
                         SchedulePool(BASE, 2), Path(folder) / "narrow")
            self.assertIn("below env_workers", str(caught.exception))

    def test_async_reset_admits_a_slot_several_ticks_after_it_was_freed(self):
        """A late reset must not stall the live slots, and must still be admitted."""
        with tempfile.TemporaryDirectory() as folder:
            expected_episodes, expected_traces = self._reference(folder, {"readout_rows": 2})
            pool = SchedulePool(BASE, 2, delay=4)
            episodes, traces = _outcome(BASE, {"readout_rows": 2, "async_reset": True}, pool,
                                        Path(folder) / "late")
            late = [harvest for harvest, slot in pool.admitted if slot % 2]
            self.assertTrue(late and min(late) >= 5, pool.admitted)  # 4 ticks after freeing
            self.assertEqual(sorted(slot for _, slot in pool.admitted), [0, 0, 1])
            self.assertEqual(episodes, expected_episodes)
            self.assertEqual(traces, expected_traces)

    def test_overlapped_bookkeeping_reads_nothing_the_workers_are_writing(self):
        """With the views scribbled mid-step, the overlapped run must still match."""
        with tempfile.TemporaryDirectory() as folder:
            expected_episodes, expected_traces = self._reference(folder)
            pool = SchedulePool(BASE, 2, hazard=True)
            episodes, traces = _outcome(BASE, {"overlap_bookkeeping": True}, pool,
                                        Path(folder) / "hazard")
            # The window really was poisoned while the bookkeeping ran, so the
            # match below is evidence that it read none of it.
            self.assertTrue(pool.windows and all(np.isnan(window).any()
                                                 for window in pool.windows))
            self.assertEqual(episodes, expected_episodes)
            self.assertEqual(traces, expected_traces)

    def test_fused_trace_splits_into_exactly_the_unfused_arrays(self):
        """The packed block and the per-key copies must agree bit for bit."""
        generator = torch.Generator().manual_seed(11)
        trace = {"scalar": torch.randn(4, generator=generator, dtype=torch.float64),
                 "vector": torch.randn(4, 3, generator=generator, dtype=torch.float64),
                 "float32": torch.randn(4, 2, generator=generator).float(),
                 "integer": torch.randint(0, 97, (4, 5), generator=generator),
                 "matrix": torch.randn(4, 2, 3, generator=generator, dtype=torch.float64)}
        packed, layout = _packed_trace(trace)
        fused = _split_trace(np.stack([packed[row].copy() for row in range(4)]), layout)
        self.assertEqual(list(fused), list(trace))
        for key, tensor in trace.items():
            value = tensor.double().numpy()
            self.assertEqual((fused[key].dtype, fused[key].shape),
                             (value.dtype, value.shape), key)
            self.assertEqual(fused[key].tobytes(), value.tobytes(), key)


@integration
class AsyncEpisodePoolTest(unittest.TestCase):
    def test_late_reset_is_harvested_and_step_replies_are_awaited(self):
        with EpisodePool(CONFIG, 2, _factory=FakeEpisode) as pool:
            pool.reset_async([0, 1], [101, 202], [501, 502])
            landed, deadline = [], time.monotonic() + 60
            while len(landed) < 2 and time.monotonic() < deadline:
                landed.extend(pool.harvest_resets(block=not landed))
            self.assertEqual(sorted(landed), [0, 1])
            self.assertEqual(pool.harvest_resets(), [])  # nothing is left pending
            action = np.arange(7, dtype=np.float64)[None] + 2 ** -40
            pool.step_async([1], action)
            pool.wait_steps()
            np.testing.assert_array_equal(pool.proprio[1, :7], action[0])
            self.assertTrue(pool.success[1])

    def test_a_second_command_on_a_busy_slot_is_refused(self):
        pool = EpisodePool(CONFIG, 1, _factory=FakeEpisode)
        processes = list(pool._processes)
        pool.reset([0], [101], [501])
        pool.step_async([0], np.zeros((1, 7)))
        with self.assertRaisesRegex(RuntimeError, "still running step"):
            pool.step_async([0], np.zeros((1, 7)))
        self.assertTrue(all(not process.is_alive() for process in processes))

    def test_a_pending_reset_that_never_lands_still_times_out(self):
        with EpisodePool(CONFIG, 2, _factory=FakeEpisode) as pool:
            pool.reset([0, 1], [101, 202], [501, 502])  # pay the workers' import cost
            pool.timeout = .5
            pool.reset_async([0, 1], [101, "wait"], [501, 502])
            landed, deadline = [], time.monotonic() + 30
            with self.assertRaisesRegex(TimeoutError, r"reset timed out in workers \[1\]"):
                while time.monotonic() < deadline:
                    landed.extend(pool.harvest_resets())
            self.assertEqual(landed, [0])  # the healthy slot was harvested first
