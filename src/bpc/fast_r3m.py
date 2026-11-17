"""Resident R3M encoding with exact per-environment CUDA-graph replay."""

import torch


class R3MEncoder:
    """Retain deployed camera microbatches, scheduling them across 16 streams.

    ``tensor`` returns device features in input order; calling the encoder keeps
    the bank-generation NumPy API. Evaluation uses the original single-world
    camera batch size, not a larger batch that can change convolution rounding.
    Graph buffers are reused, so calls must be sequential, not concurrent.
    """

    def __init__(self, network, resize, batch_size, *, graphs=False):
        from torchvision.transforms import Resize

        if batch_size < 1:
            raise ValueError("R3M batch size must be positive")
        self.network = network.eval().requires_grad_(False)
        self.device = next(network.parameters()).device
        self.resize = Resize((resize, resize), antialias=False)
        self.image_size = (resize, resize)
        self.batch_size = batch_size
        self.graphs = graphs and self.device.type == "cuda"
        self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32,
                                 device=self.device).view(3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], dtype=torch.float32,
                                device=self.device).view(3, 1, 1)
        self._lanes = []
        self._geometry = None

    def _forward(self, frames):
        batch = (frames.permute(0, 3, 1, 2) / 255.0).float()
        if tuple(batch.shape[-2:]) != self.image_size:
            batch = self.resize(batch)
        # torchvision Normalize's original clone/subtract/divide arithmetic.
        batch = batch.clone().sub_(self.mean).div_(self.std)
        return self.network(batch).flatten(1)

    @torch.no_grad()
    def _capture(self, frames):
        caller = torch.cuda.current_stream(self.device)
        for stream, *_ in self._lanes:
            stream.synchronize()
        self._lanes = []
        for _ in range(16):
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                static = frames[:self.batch_size].clone()
                for _ in range(3):
                    self._forward(static)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                features = self._forward(static)
            self._lanes.append((stream, static, graph, features))
        self._geometry = (tuple(frames.shape[1:]), frames.dtype)

    @torch.no_grad()
    def tensor(self, frames):
        """Encode flattened world-major camera frames without a CPU round-trip."""
        frames = torch.as_tensor(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
            raise ValueError("R3M frames must be nonempty [N,H,W,3]")
        if not self.graphs or len(frames) < self.batch_size:
            return torch.cat([
                self._forward(chunk.to(self.device))
                for chunk in frames.split(self.batch_size)])

        frames = frames.to(self.device).contiguous()
        geometry = (tuple(frames.shape[1:]), frames.dtype)
        if self._geometry != geometry:
            self._capture(frames)
        output = torch.empty(len(frames), self._lanes[0][3].shape[1],
                             device=self.device, dtype=self._lanes[0][3].dtype)
        caller = torch.cuda.current_stream(self.device)
        for stream, *_ in self._lanes:
            stream.wait_stream(caller)
            frames.record_stream(stream)
            output.record_stream(stream)
        for index, start in enumerate(range(0, len(frames), self.batch_size)):
            stop = min(start + self.batch_size, len(frames))
            if stop - start != self.batch_size:
                output[start:stop].copy_(self._forward(frames[start:stop]))
                continue
            stream, static, graph, features = self._lanes[index % len(self._lanes)]
            with torch.cuda.stream(stream):
                static.copy_(frames[start:stop])
                graph.replay()
                output[start:stop].copy_(features)
        for stream, *_ in self._lanes:
            caller.wait_stream(stream)
        return output

    def __call__(self, frames):
        return self.tensor(frames).cpu().numpy()
