import sys
from types import SimpleNamespace
import unittest
from tests.suites import integration
from unittest.mock import Mock, patch

import numpy as np


from bpc.data import seed
from bpc.episode_pool import EpisodePool
from bpc.robosuite import _compact_renderer
from tests.support.rollouts import EPISODE_CONFIG as CONFIG, FakeEpisode


@integration
class EpisodePoolTest(unittest.TestCase):
    def test_reset_recycling_and_shared_step_match_fresh_seeded_episodes(self):
        with EpisodePool(CONFIG, 2, _factory=FakeEpisode) as pool:
            processes = list(pool._processes)
            views = pool.proprio, pool.frames, pool.success
            for initializers, seeds in (([101, 202], [501, 502]), ([303, 404], [503, 504])):
                pool.reset([0, 1], initializers, np.asarray(seeds, dtype=np.int64))
                for slot in range(2):
                    seed(seeds[slot])
                    reference = FakeEpisode(CONFIG)
                    expected, images = reference.reset(initializers[slot])
                    np.testing.assert_array_equal(pool.proprio[slot], expected)
                    for name in images:
                        np.testing.assert_array_equal(pool.images[name][slot], images[name])
                inactive = pool.proprio[0].copy()
                action = np.arange(7, dtype=np.float64)[None] + 2 ** -40
                pool.step([1], action)
                np.testing.assert_array_equal(pool.proprio[0], inactive)
                np.testing.assert_array_equal(pool.proprio[1, :7], action[0])
                self.assertTrue(pool.success[1])
                self.assertFalse(pool.success[0])
                for original, current in zip(views, (pool.proprio, pool.frames, pool.success)):
                    self.assertIs(original, current)
                self.assertEqual([p.pid for p in pool._processes], [p.pid for p in processes])
            self.assertEqual(pool.actions.dtype, np.float64)
            self.assertEqual(pool.proprio.dtype, np.float64)
            self.assertEqual(pool.frames.dtype, np.uint8)
        self.assertTrue(all(not process.is_alive() for process in processes))
        pool.close()

    def test_worker_failure_and_crash_reach_parent_and_close_pool(self):
        for initializer, message in (("raise", "intentional simulator failure"),
                                     ("exit", "worker 0 exited")):
            with self.subTest(initializer=initializer):
                pool = EpisodePool(CONFIG, 1, _factory=FakeEpisode)
                processes = list(pool._processes)
                with self.assertRaisesRegex(RuntimeError, message):
                    pool.reset([0], [initializer], [101])
                self.assertTrue(all(not process.is_alive() for process in processes))

    def test_timeout_and_empty_active_set(self):
        with EpisodePool(CONFIG, 1, _factory=FakeEpisode) as pool:
            pool.reset([0], [101], [101])
            pool.reset([], [], [])
            pool.step([], np.empty((0, 7)))
            pool.timeout = .1
            with self.assertRaisesRegex(TimeoutError, "reset timed out"):
                pool.reset([0], ["wait"], [101])

    def test_invalid_slots_and_lengths_fail_before_dispatch(self):
        with EpisodePool(CONFIG, 1, _factory=FakeEpisode) as pool:
            with self.assertRaisesRegex(ValueError, "slots"):
                pool.reset([0, 0], [1, 2], [1, 2])
            with self.assertRaisesRegex(ValueError, "slots"):
                pool.step([1], np.zeros((1, 7)))
            with self.assertRaisesRegex(ValueError, "initializer"):
                pool.reset([0], [], [1])
            with self.assertRaisesRegex(ValueError, "complete action"):
                pool.step([0], np.zeros((1, 6)))

    def test_compact_renderer_only_reduces_disabled_shadow_allocation(self):
        original = Mock(return_value=object())
        base = SimpleNamespace(MjRenderContextOffscreen=original)
        modules = {
            "robosuite.environments": SimpleNamespace(base=base),
            "robosuite.utils.binding_utils": SimpleNamespace(MjRenderContextOffscreen=original),
        }
        with patch.dict(sys.modules, modules):
            for shadows in ([0, 0], [0, 1]):
                visual = SimpleNamespace(global_=SimpleNamespace(offwidth=640, offheight=480),
                                         quality=SimpleNamespace(shadowsize=4096))
                sim = SimpleNamespace(model=SimpleNamespace(vis=visual, light_castshadow=shadows))
                _compact_renderer(224)
                base.MjRenderContextOffscreen(sim, 1)
                self.assertEqual((visual.global_.offwidth, visual.global_.offheight), (224, 224))
                self.assertEqual(visual.quality.shadowsize, 4096 if any(shadows) else 64)
                original.assert_called_with(sim, device_id=1)
            self.assertEqual(original.call_count, 2)  # repeated installs do not nest wrappers


if __name__ == "__main__":
    unittest.main()
