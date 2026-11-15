"""V79's tile-major pinned-host RFF atlas with bounded CUDA replay storage."""

import torch


class HostPagedAtlas:
    """FP32 host pages, two CUDA staging pages, and one row-major replay page.

    Replay consumers must enqueue their work on the current CUDA stream and
    must not retain the borrowed page.
    """

    ndim = 2
    dtype = torch.float32

    def __init__(self, rows, feature_dim, *, device, page_rows=8192, feature_tile=512):
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0
               for n in (rows, feature_dim, page_rows, feature_tile)):
            raise ValueError("atlas dimensions must be positive integers")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("host-paged replay requires a CUDA device")
        self.rows, self.feature_dim = rows, feature_dim
        self.page_rows, self.feature_tile = page_rows, feature_tile
        self.page_count = (rows + page_rows - 1) // page_rows
        self.feature_tiles = (feature_dim + feature_tile - 1) // feature_tile
        self._page_shape = (self.feature_tiles, page_rows, feature_tile)
        self.page_storage_bytes = self.feature_tiles * page_rows * feature_tile * 4
        self.pinned_storage_bytes = self.page_count * self.page_storage_bytes
        self.logical_storage_bytes = rows * feature_dim * 4
        self.padding_bytes = self.pinned_storage_bytes - self.logical_storage_bytes
        self.pages, self.staging = [], []
        self.row_buffer = self.transfer_stream = self.producer_stream = None
        self._ready, self._reusable = [None] * 2, [None] * 2
        self._pending_copies = []
        self._produced = self._closed = False
        self.producer_copy_calls = self.producer_d2h_bytes = 0
        self.page_loads = self.replay_count = 0
        self.pages = [torch.empty(self._page_shape, dtype=self.dtype, pin_memory=True)
                      for _ in range(self.page_count)]

    @property
    def shape(self):
        return self.rows, self.feature_dim

    def __len__(self):
        return self.rows

    def row_count(self, page_index):
        if not 0 <= page_index < self.page_count:
            raise IndexError("atlas page is out of bounds")
        return min(self.page_rows, self.rows - page_index * self.page_rows)

    def begin_production(self):
        if self._closed or self.producer_stream is not None:
            raise RuntimeError("atlas production was already started or closed")
        self.producer_stream = torch.cuda.Stream(device=self.device)

    def write_tile(self, *, page_index, tile_index, local_start, values):
        """Queue one feature block; retain its storage until the D2H copy finishes."""
        if self._closed or self.producer_stream is None or self._produced:
            raise RuntimeError("begin production before writing atlas tiles")
        rows = self.row_count(page_index)
        width = min(self.feature_tile, self.feature_dim - tile_index * self.feature_tile)
        if (not 0 <= tile_index < self.feature_tiles or values.ndim != 2
                or values.shape[1] != width or local_start < 0
                or local_start + len(values) > rows or values.dtype != self.dtype
                or values.device != self.producer_stream.device):
            raise ValueError("atlas tile has incompatible shape, dtype, device or bounds")
        if len(self._pending_copies) == 2:
            self._pending_copies.pop(0).synchronize()
        if width != self.feature_tile:
            padded = values.new_zeros((len(values), self.feature_tile))
            padded[:, :width].copy_(values)
            values = padded
        destination = self.pages[page_index][
            tile_index, local_start:local_start + len(values)]
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.producer_stream):
            self.producer_stream.wait_event(ready)
            destination.copy_(values, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(self.producer_stream)
        self._pending_copies.append(copied)
        values.record_stream(self.producer_stream)
        self.producer_copy_calls += 1
        self.producer_d2h_bytes += values.numel() * values.element_size()

    def finish_production(self):
        if self._closed or self.producer_stream is None or not self.producer_copy_calls:
            raise RuntimeError("atlas production queued no writes")
        self.producer_stream.synchronize()
        self._pending_copies.clear()
        self._produced = True

    def allocate_replay_buffers(self):
        if self._closed or not self._produced or self.staging:
            raise RuntimeError("finish production before allocating replay buffers once")
        self.staging = [torch.empty(self._page_shape, dtype=self.dtype, device=self.device)
                        for _ in range(2)]
        self.row_buffer = torch.empty(
            (self.page_rows, self.feature_tiles * self.feature_tile),
            dtype=self.dtype, device=self.device)
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.transfer_stream.wait_stream(torch.cuda.current_stream(self.device))
        for page in self.staging:
            page.record_stream(self.transfer_stream)

    def _require_replay(self):
        if self._closed or self.transfer_stream is None:
            raise RuntimeError("allocate replay buffers before reading the atlas")

    def _queue_page(self, page_index, slot):
        self.row_count(page_index)
        with torch.cuda.stream(self.transfer_stream):
            if self._reusable[slot] is not None:
                self.transfer_stream.wait_event(self._reusable[slot])
            self.staging[slot].copy_(self.pages[page_index], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.transfer_stream)
        self._ready[slot] = ready
        self.page_loads += 1

    def _release_page(self, slot):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self._reusable[slot] = event

    def replay(self, consumer):
        """Visit ordered row-major pages while prefetching the next host page."""
        self._require_replay()
        current = torch.cuda.current_stream(self.device)
        self._queue_page(0, 0)
        try:
            for page_index in range(self.page_count):
                slot = page_index % 2
                current.wait_event(self._ready[slot])
                if page_index + 1 < self.page_count:
                    self._queue_page(page_index + 1, (slot + 1) % 2)
                rows = self.row_count(page_index)
                try:
                    self.row_buffer[:rows].view(
                        rows, self.feature_tiles, self.feature_tile).copy_(
                            self.staging[slot][:, :rows].permute(1, 0, 2))
                    start = page_index * self.page_rows
                    consumer(self.row_buffer[:rows, :self.feature_dim], start, start + rows)
                finally:
                    self._release_page(slot)
            self.replay_count += 1
        finally:
            # Also drain on consumer failure before buffers can be reused/freed.
            current.synchronize()
            self.transfer_stream.synchronize()
            self._ready, self._reusable = [None] * 2, [None] * 2

    def close(self):
        """Drain outstanding transfers before releasing pinned and CUDA storage."""
        for stream in (getattr(self, "producer_stream", None),
                       getattr(self, "transfer_stream", None)):
            if stream is not None:
                stream.synchronize()
        for event in getattr(self, "_reusable", ()):
            if event is not None:
                event.synchronize()
        self.pages, self.staging, self.row_buffer = [], [], None
        self._pending_copies = []
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # CUDA may already be shut down during interpreter teardown.
