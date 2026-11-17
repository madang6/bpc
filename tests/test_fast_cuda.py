import unittest
from tests.suites import cuda
from unittest import mock

import torch


from bpc import fast_cuda


class FastCudaDispatchTest(unittest.TestCase):
    def test_cpu_inputs_do_not_load_cuda(self):
        with mock.patch.object(fast_cuda, "extension") as load:
            self.assertIsNone(fast_cuda.call("history_accumulators", torch.zeros(1)))
        load.assert_not_called()

    def test_shared_loader_routes_each_kernel(self):
        reference = mock.Mock(is_cuda=True)
        reference.device.index = 0
        for operation in ("pack_history", "history_accumulators"):
            with self.subTest(operation=operation):
                arguments = (None, None, torch.zeros(1, 1), torch.zeros(1, 1))
                with (mock.patch.object(fast_cuda, "extension", return_value=None) as load,
                      mock.patch.object(torch.cuda, "get_device_properties",
                                        return_value=mock.Mock(shared_memory_per_block=1024))):
                    self.assertIsNone(fast_cuda.call(operation, reference, *arguments))
                load.assert_called_once_with(0, "fast_cuda")

    def test_history_exceeding_shared_memory_uses_torch(self):
        reference = mock.Mock(is_cuda=True)
        candidates = torch.zeros(1, 50, dtype=torch.long)
        states = torch.zeros(1, 100, dtype=torch.float64)
        with (mock.patch.object(fast_cuda, "extension") as load,
              mock.patch.object(torch.cuda, "get_device_properties",
                                return_value=mock.Mock(shared_memory_per_block=1024))):
            self.assertIsNone(fast_cuda.call(
                "history_accumulators", reference, None, None, candidates, states))
        load.assert_not_called()


@cuda
class FastCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = fast_cuda.extension(torch.cuda.current_device())
        if cls.module is None:
            raise unittest.SkipTest("fit CUDA extension unavailable")

    def test_history_pack_is_bit_exact_for_five_and_ten_steps(self):
        generator = torch.Generator(device="cuda").manual_seed(23)
        for dtype in (torch.float32, torch.float64):
            states = torch.randn(31, 1033, generator=generator, device="cuda", dtype=dtype)
            actions = torch.randn(31, 7, generator=generator, device="cuda", dtype=dtype)
            starts = torch.tensor([17, 0, 4, 17, 3], device="cuda")
            for history in (5, 10):
                indices = starts[:, None] + torch.arange(history, device="cuda")
                expected = torch.cat([actions[indices].flatten(1),
                                      states[indices].flatten(1)], dim=1)
                actual = self.module.pack_history(states, actions, starts, history)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                empty = self.module.pack_history(states, actions, starts[:0], history)
                self.assertEqual(empty.shape, (0, history * 1040))

    def test_fused_history_matches_literal_and_topk(self):
        generator = torch.Generator(device="cuda").manual_seed(23)
        lengths_by_demo = (13, 17, 19)
        states = torch.randn(
            sum(lengths_by_demo), 37, generator=generator,
            device="cuda", dtype=torch.float64)
        ages = torch.cat([
            torch.arange(length, device="cuda") for length in lengths_by_demo])
        trajectory = torch.repeat_interleave(
            torch.arange(len(lengths_by_demo), device="cuda"),
            torch.tensor(lengths_by_demo, device="cuda"))
        starts = torch.tensor([0, 13, 30], device="cuda")
        rows = torch.tensor([9, 12, 22, 28, 43], device="cuda")
        candidates = torch.randint(
            len(states), (len(rows), 13), generator=generator, device="cuda")
        for lookback in (1, 4, 10):
            with self.subTest(lookback=lookback):
                back = torch.arange(lookback - 1, -1, -1, device="cuda")
                indices = torch.maximum(
                    rows[:, None] - back, starts[trajectory[rows]][:, None])
                windows = states[indices]
                lengths = (ages[rows] + 1).clamp(max=lookback)
                weights = (torch.arange(1, lookback + 1, device="cuda",
                                        dtype=torch.float64).pow(-.3658387314461997))
                actual = self.module.history_accumulators(
                    windows.contiguous(), lengths.contiguous(),
                    candidates.contiguous(), states.contiguous(), ages.contiguous(),
                    weights.contiguous())
                maximum = torch.minimum(
                    ages[candidates] + 1, lengths[:, None]).clamp(max=lookback)
                expected = torch.zeros_like(actual)
                for offset in range(lookback):
                    distance = torch.linalg.vector_norm(
                        states[(candidates - offset).clamp_min(0)]
                        - windows[:, lookback - 1 - offset, None], dim=2)
                    expected += (offset < maximum) * weights[offset] * distance
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-10)
                self.assertTrue(torch.equal(
                    torch.topk(actual, 5, largest=False).indices,
                    torch.topk(expected, 5, largest=False).indices))


if __name__ == "__main__":
    unittest.main()
