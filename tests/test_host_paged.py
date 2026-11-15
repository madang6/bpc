import unittest
from tests.suites import cuda

import torch


from dpc_nldl.host_paged import HostPagedAtlas


@cuda
class HostPagedAtlasTest(unittest.TestCase):
    def atlas(self, rows=39, width=12, page_rows=8, feature_tile=4):
        source = torch.arange(rows * width, dtype=torch.float32, device="cuda")
        source = source.reshape(rows, width).div_(7)
        atlas = HostPagedAtlas(rows, width, device=source.device,
                               page_rows=page_rows, feature_tile=feature_tile)
        self.addCleanup(atlas.close)
        atlas.begin_production()
        for tile in range(atlas.feature_tiles):
            for page in range(atlas.page_count):
                for local in range(0, atlas.row_count(page), 3):
                    start = page * page_rows + local
                    stop = min(start + 3, page * page_rows + atlas.row_count(page))
                    values = source[start:stop, tile * feature_tile:(tile + 1) * feature_tile]
                    atlas.write_tile(page_index=page, tile_index=tile,
                                     local_start=local, values=values.clone())
        atlas.finish_production()
        atlas.allocate_replay_buffers()
        return atlas, source

    def test_exact_roundtrip_tail_rows_and_features(self):
        for width in (12, 11, 3):
            with self.subTest(width=width):
                atlas, source = self.atlas(width=width)
                actual = torch.empty_like(source)
                boundaries = []

                def consume(page, start, stop):
                    actual[start:stop].copy_(page)
                    boundaries.append((start, stop))

                atlas.replay(consume)
                self.assertTrue(torch.equal(actual, source))
                self.assertEqual(boundaries, [(0, 8), (8, 16), (16, 24), (24, 32), (32, 39)])
                self.assertEqual(atlas.shape, source.shape)
                self.assertEqual(len(atlas), len(source))
                self.assertTrue(all(page.is_pinned() for page in atlas.pages))
                self.assertEqual(atlas.pinned_storage_bytes,
                                 sum(page.numel() * 4 for page in atlas.pages))
                self.assertEqual(len(atlas.staging), 2)
                self.assertEqual(atlas.row_buffer.numel() * 4, atlas.page_storage_bytes)

    def test_replay_reuses_bounded_buffers(self):
        atlas, source = self.atlas()
        buffers = [page.data_ptr() for page in atlas.staging] + [atlas.row_buffer.data_ptr()]
        for _ in range(3):
            output = torch.empty_like(source)
            atlas.replay(lambda page, start, stop: output[start:stop].copy_(page))
            self.assertTrue(torch.equal(output, source))
        self.assertEqual(atlas.replay_count, 3)
        self.assertEqual(buffers, [page.data_ptr() for page in atlas.staging]
                         + [atlas.row_buffer.data_ptr()])

    def test_async_production_records_source_lifetime(self):
        rows, width = 2048, 512
        atlas = HostPagedAtlas(rows, width, device="cuda", page_rows=512, feature_tile=128)
        self.addCleanup(atlas.close)
        atlas.begin_production()
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            for tile in range(4):
                for page in range(4):
                    torch.cuda._sleep(100_000)
                    values = torch.full((512, 128), tile * 4 + page,
                                        device="cuda", dtype=torch.float32)
                    atlas.write_tile(page_index=page, tile_index=tile,
                                     local_start=0, values=values)
                    del values
                    # Pressure the source allocator while the D2H copy is queued.
                    torch.empty((512, 128), device="cuda").fill_(-99)
        atlas.finish_production()
        atlas.allocate_replay_buffers()
        expected = torch.empty((rows, width), device="cuda")
        for tile in range(4):
            for page in range(4):
                expected[page * 512:(page + 1) * 512, tile * 128:(tile + 1) * 128] = tile * 4 + page
        output = torch.empty_like(expected)
        atlas.replay(lambda page, start, stop: output[start:stop].copy_(page))
        self.assertTrue(torch.equal(output, expected))

    def test_production_bounds_outstanding_device_buffers(self):
        page_rows, width, pages = 512, 512, 24
        atlas = HostPagedAtlas(page_rows * pages, width, device="cuda",
                               page_rows=page_rows, feature_tile=width)
        self.addCleanup(atlas.close)
        atlas.begin_production()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_stats()["active_bytes.all.current"]
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.stream(atlas.producer_stream):
            torch.cuda._sleep(200_000_000)
        for page in range(pages):
            values = torch.full((page_rows, width), page, device="cuda",
                                dtype=torch.float32)
            atlas.write_tile(page_index=page, tile_index=0, local_start=0,
                             values=values)
            self.assertLessEqual(len(atlas._pending_copies), 2)
            del values
        peak = torch.cuda.memory_stats()["active_bytes.all.peak"] - baseline
        self.assertLessEqual(peak, 4 * page_rows * width * 4)
        atlas.finish_production()
        self.assertEqual(atlas._pending_copies, [])
        for page, values in enumerate(atlas.pages):
            self.assertTrue(bool((values == page).all()))

    def test_replay_waits_for_delayed_consumers_on_other_stream(self):
        atlas, source = self.atlas(rows=257, width=128, page_rows=64, feature_tile=32)
        compute = torch.cuda.Stream()
        compute.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(compute):
            output = torch.empty_like(source)

            def consume(page, start, stop):
                torch.cuda._sleep(100_000)
                output[start:stop].copy_(page)

            atlas.replay(consume)
        self.assertTrue(torch.equal(output, source))

    def test_consumer_exception_drains_and_allows_replay(self):
        atlas, source = self.atlas()

        def failing_consumer(page, start, stop):
            page.square_()
            raise ValueError("consumer failed")

        with self.assertRaisesRegex(ValueError, "consumer failed"):
            atlas.replay(failing_consumer)
        output = torch.empty_like(source)
        atlas.replay(lambda page, start, stop: output[start:stop].copy_(page))
        self.assertTrue(torch.equal(output, source))
        atlas.close()
        atlas.close()
        with self.assertRaises(RuntimeError):
            atlas.replay(lambda *_: None)

    def test_validation_and_lifecycle(self):
        for dimensions in ((0, 4), (4, 0), (True, 4)):
            with self.assertRaises(ValueError):
                HostPagedAtlas(*dimensions, device="cuda")
        atlas = HostPagedAtlas(4, 4, device="cuda", page_rows=4, feature_tile=4)
        self.addCleanup(atlas.close)
        with self.assertRaises(RuntimeError):
            atlas.allocate_replay_buffers()
        with self.assertRaises(RuntimeError):
            atlas.finish_production()
        atlas.begin_production()
        with self.assertRaises(ValueError):
            atlas.write_tile(page_index=0, tile_index=0, local_start=1,
                             values=torch.zeros((4, 4), device="cuda"))
        atlas.write_tile(page_index=0, tile_index=0, local_start=0,
                         values=torch.zeros((4, 4), device="cuda"))
        atlas.finish_production()
        atlas.allocate_replay_buffers()
        for index in (-1, atlas.page_count):
            with self.assertRaises(IndexError):
                atlas.row_count(index)


if __name__ == "__main__":
    unittest.main()
