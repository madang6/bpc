import unittest
from tests.suites import cuda
from unittest.mock import patch

import numpy as np
import torch
from torchvision import transforms


from dpc_nldl.data import make_encoder
from dpc_nldl.fast_r3m import R3MEncoder


def original(network, frames, resize, batch_size):
    transform = transforms.Compose([
        transforms.Resize((resize, resize), antialias=False),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])
    with torch.no_grad():
        return torch.cat([
            network(transform((chunk.permute(0, 3, 1, 2) / 255.0).float())).flatten(1)
            for chunk in frames.split(batch_size)])


def network(device):
    torch.manual_seed(8)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                               torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1)
                               ).to(device).eval()


class R3MEncoderTest(unittest.TestCase):
    def test_original_preprocessing_and_numpy_api(self):
        model = network("cpu")
        for size in (7, 8):
            frames = torch.randint(256, (5, size, size, 3), dtype=torch.uint8)
            saved = frames.clone()
            encoder = R3MEncoder(model, 8, 2, graphs=True)
            expected = original(model, frames, 8, 2)
            torch.testing.assert_close(encoder.tensor(frames), expected, rtol=0, atol=0)
            np.testing.assert_array_equal(encoder(frames.numpy()), expected.numpy())
            torch.testing.assert_close(frames, saved, rtol=0, atol=0)
            self.assertFalse(encoder.graphs)

    def test_evaluation_keeps_single_world_camera_batch(self):
        data = {"feature_keys": ["agent", "wrist"],
                "cameras": ["agentview_rgb", "eye_in_hand_rgb"],
                "r3m": {"model": "resnet18", "device": "cuda",
                        "resize": 224, "batch_size": 512}}
        with patch("dpc_nldl.data.make_r3m_encoder") as factory:
            make_encoder(data)
            factory.assert_called_once_with("resnet18", "cuda", 224, 512,
                                            evaluation=False)
            factory.reset_mock()
            make_encoder(data, evaluation=True)
            factory.assert_called_once_with("resnet18", "cuda", 224, 2,
                                            evaluation=True)
            data["r3m"]["batch_size"] = 1
            factory.reset_mock()
            make_encoder(data, evaluation=True)
            factory.assert_called_once_with("resnet18", "cuda", 224, 1,
                                            evaluation=True)

    def test_invalid_frames(self):
        encoder = R3MEncoder(network("cpu"), 8, 2)
        for shape in ((0, 8, 8, 3), (8, 8, 3), (1, 8, 8, 4)):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                encoder.tensor(torch.empty(shape))


@cuda
class R3MGraphTest(unittest.TestCase):
    def test_graphs_match_deployed_camera_batches_and_reuse_buffers(self):
        model = network("cuda")
        for batch_size in (1, 2):
            encoder = R3MEncoder(model, 8, batch_size, graphs=True)
            saved_graphs = None
            for count in (37, 6, 70, 1):
                frames = torch.randint(256, (count, 8, 8, 3),
                                       dtype=torch.uint8, device="cuda")
                actual = encoder.tensor(frames)
                expected = original(model, frames, 8, batch_size)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                graphs = tuple(id(lane[2]) for lane in encoder._lanes)
                if saved_graphs is not None:
                    self.assertEqual(graphs, saved_graphs)
                saved_graphs = graphs
            self.assertEqual(len(encoder._lanes), 16)

    def test_resize_and_independent_asynchronous_outputs(self):
        model = network("cuda")
        encoder = R3MEncoder(model, 8, 2, graphs=True)
        frames = torch.randint(256, (42, 7, 7, 3), dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            first = encoder.tensor(frames[::2])
            second = encoder.tensor(frames[1::2])
        torch.cuda.current_stream().wait_stream(stream)
        resized = torch.randint(256, (4, 8, 8, 3), dtype=torch.uint8, device="cuda")
        torch.testing.assert_close(encoder.tensor(resized), original(model, resized, 8, 2),
                                   rtol=0, atol=0)
        torch.testing.assert_close(first, original(model, frames[::2], 8, 2),
                                   rtol=0, atol=0)
        torch.testing.assert_close(second, original(model, frames[1::2], 8, 2),
                                   rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
