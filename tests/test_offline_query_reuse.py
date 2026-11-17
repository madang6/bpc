"""Full-precision gather reuse against the literal 5eee261 target implementation.

The three reference functions intentionally retain their original repeated gathers.
Projection and solve inputs are compared, including original batch shapes and dtypes.
"""
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from bpc.ragged_supports import RaggedColumns, support_groups
from tests.support.readout import offline_fixture as fixture
from tests.support.readout import ragged_policy


def literal_fixed_g(self, supports, rows, *, rates=None):
    """Solve fixed supports in bounded history tiles."""
    batch, count = supports.shape
    tile = self._g_microbatch(count)
    grams, rhs, coefficients = [], [], []
    for start in range(0, batch, tile):
        stop = min(start + tile, batch)
        selected = rows[start:stop]
        query = self._queries(selected)
        rate_args = {} if rates is None else {"rates": rates[start:stop]}
        design = self._reconstruction_columns(
            supports[start:stop], query, torch.float32, selected,
            **rate_args).transpose(1, 2)
        query = query.float()
        if self.g_solver == "exact":
            coefficients.append(self._solve_affine(design, query))
            continue
        transpose = design.transpose(1, 2)
        grams.append(transpose @ design)
        rhs.append((transpose @ query.unsqueeze(2)).squeeze(2))
    return (torch.cat(coefficients) if coefficients else self._solve_g_normal(
        torch.cat(grams), torch.cat(rhs)))


def literal_g_prior(self, plane, supports, queries, rows=None, rates=None, *, futures=None):
    fixed_rates = future_queries = None
    if (futures is None and rates is None and plane is None and rows is not None
            and self.g_weights != "uniform" and self.g_align_history and self.g_align_future
            and len(self.g_lookup_rates) > 1 and len(rows) <= self._g_microbatch(supports.shape[1])):
        # These two decoders originally projected the identical FP64 query
        # with the identical batch shape. Share only in this case: a split
        # solve keeps its original per-microbatch rate decisions. Explicit
        # caller rates still describe futures only, as before.
        future_queries = self._queries(rows)
        fixed_rates = self._rate_trace(future_queries, supports, rows)["hankel_rates"]
        rates = fixed_rates
    if futures is None:
        if future_queries is None:
            future_queries = self._queries(rows) if self.g_align_future and rows is not None else queries
        futures = self._future_columns(
            supports, future_queries, rows, rates)
    if self.g_weights == "uniform":
        coefficients = queries.new_full(
            supports.shape, 1 / supports.shape[1])
        prior = futures[:, 0].clone()
        prior[:, :self.action_dim] = futures[:, :, :self.action_dim].mean(1)
        return prior, coefficients
    if plane is None:
        coefficients = literal_fixed_g(self, supports, rows,
            **({} if fixed_rates is None else {"rates": fixed_rates}))
        return (torch.einsum(
            "bkh,bk->bh", futures.float(), coefficients).to(self.dtype),
            coefficients)
    futures = futures.to(plane.dtype)
    Z = plane.transpose(1, 2)
    if self.g_solver == "exact":
        coefficients = self._solve_affine(Z, queries)
    else:
        queries = queries.to(Z.dtype)
        coefficients = self._solve_g_normal(
            Z.transpose(1, 2) @ Z,
            (Z.transpose(1, 2) @ queries[:, :, None]).squeeze(2)).to(futures.dtype)
    return (torch.einsum("bkh,bk->bh", futures, coefficients).to(self.dtype),
            coefficients)


def literal_offline_g(self, rows, leave_demo_out=False, atlas=None):
    if getattr(self, "support_selector", None) is not None and atlas is None:
        queries = self._queries(rows)
        selected = self.support_selector.select(
            queries, target_rows=rows,
            target_demos=self.trajectory_of[rows] if leave_demo_out else None)
        prior = queries.new_empty(len(rows), self.horizon * self.action_dim)
        retained = rows.new_empty(len(rows))
        excluded = (torch.bincount(self.hankel_trajectory, minlength=self.demo_count)[
            self.trajectory_of[rows]] if leave_demo_out else
            torch.searchsorted(self.hankel_starts, rows, right=True)
            - torch.searchsorted(self.hankel_starts, rows - self.history))
        for indices, columns, _ in support_groups(selected, self.device):
            values = literal_offline_g(self,
                rows[indices], leave_demo_out, atlas=(columns, excluded[indices]))
            prior[indices], retained[indices] = values[0], values[2]
        neighbors = RaggedColumns.from_rows([
            self._support_endpoints(columns) for columns, _ in selected])
        return prior, excluded, retained, neighbors
    support_count = (min(self.g_supports, len(self.hankel_starts))
                     if atlas is None else atlas[0].shape[1])
    legacy_batch = self._legacy_g_batch(support_count)
    packed = (self.g_native and self.device.type == "cuda"
              and self.g_tolerance is None and self.g_weights != "uniform")
    solve_batch = len(rows) if packed or self.g_weights == "uniform" else legacy_batch
    projected = self.g_lookup == "past" and self.pf is not None
    plane_dtype = self.g_fit_dtype
    priors, exclusions, retained_counts, neighbors = [], [], [], []
    for start in range(0, len(rows), solve_batch):
        stop = min(start + solve_batch, len(rows))
        selected = rows[start:stop]
        queries = self._queries(selected)
        target_demos = self.trajectory_of[selected] if leave_demo_out else None
        if atlas is None:
            supports, excluded = self._g_support_topk(
                queries, support_count, selected, target_demos, legacy_batch)
        else:
            supports, excluded = (value[start:stop] for value in atlas)
        retained = self.hankel_starts[supports]
        if ((selected[:, None] - self.history <= retained)
                & (retained <= selected[:, None])).any():
            raise RuntimeError("target action entered an offline G window")
        if leave_demo_out and (self.hankel_trajectory[supports]
                               == target_demos[:, None]).any():
            raise RuntimeError(
                "target demonstration entered a sequential G support")
        rates = None
        if (not packed and self.g_weights != "uniform" and not projected
                and self.g_align_history and self.g_align_future and len(self.g_lookup_rates) > 1):
            # Reconstruction and future decoding consume the same original
            # query batch and exclusion rows in the materialized path.
            rates = self._rate_trace(queries, supports, selected)["hankel_rates"]
        plane = (None if self.g_weights == "uniform" or packed else
                 self.pf[supports].to(plane_dtype) if projected else
                 self._reconstruction_columns(supports, queries, plane_dtype, selected,
                     **({} if rates is None else {"rates": rates})))
        arguments = (plane, supports, queries.to(plane_dtype),
                     selected)
        prior, _ = literal_g_prior(self, *arguments, **({} if rates is None else {"rates": rates}))
        priors.append(prior)
        exclusions.append(excluded)
        retained_counts.append((self.hankel_trajectory[supports]
                                == self.trajectory_of[selected, None]).sum(1))
        neighbors.append(self._support_endpoints(supports))
    return tuple(map(torch.cat, (
        priors, exclusions, retained_counts, neighbors)))


def copied(value):
    if torch.is_tensor(value):
        return value.clone(), tuple(value.stride()), value.is_contiguous()
    if isinstance(value, (tuple, list)):
        return tuple(copied(item) for item in value)
    if isinstance(value, dict):
        return {key: copied(item) for key, item in value.items()}
    return value


def execution(policy, callback):
    """Record arithmetic inputs, not just final outputs which can hide a wrong rate."""
    work = []
    def traced(name, method):
        def call(*args, **kwargs):
            work.append((name, copied(args), copied(kwargs)))
            return method(*args, **kwargs)
        return call
    with ExitStack() as stack:
        gather = stack.enter_context(patch.object(policy, '_queries', wraps=policy._queries))
        for name in ('_rate_trace', '_reconstruction_columns', '_future_columns', '_solve_affine'):
            stack.enter_context(patch.object(policy, name, side_effect=traced(name, getattr(policy, name))))
        result = callback()
    return result, work, gather.call_count


class OfflineQueryReuseTest(unittest.TestCase):
    def assert_exact(self, actual, expected):
        if isinstance(expected, RaggedColumns):
            self.assertEqual(actual.shape, expected.shape)
            self.assert_exact(actual.payload(), expected.payload())
        elif torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_exact(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_exact(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_fixed_solve_reuses_full_precision_history_at_original_microtiles(self):
        for tile in (1, 2, 8):
            for supplied_rates in (False, True):
                with self.subTest(tile=tile, explicit=supplied_rates):
                    policy, rows, query, supports = fixture()
                    rates = torch.ones_like(supports, dtype=query.dtype) if supplied_rates else None
                    with patch.object(policy, '_g_microbatch', return_value=tile):
                        expected = execution(policy, lambda: literal_fixed_g(
                            policy, supports, rows, rates=rates))
                        cached = execution(policy, lambda: policy._fixed_g(
                            supports, rows, rates=rates, query_history=query))
                        fallback = execution(policy, lambda: policy._fixed_g(
                            supports, rows, rates=rates))
                    self.assert_exact(cached[:2], expected[:2])
                    self.assert_exact(fallback[:2], expected[:2])
                    tiles = (len(rows)+tile-1)//tile
                    self.assertEqual(expected[2], tiles)
                    self.assertEqual(cached[2], 0)
                    self.assertEqual(fallback[2], tiles)

    def test_fp32_solve_query_never_replaces_fp64_temporal_query(self):
        for tile in (1, 2, 8):
            for supplied_rates in (False, True):
                with self.subTest(tile=tile, explicit=supplied_rates):
                    policy, rows, query, supports = fixture()
                    self.assertFalse(torch.equal(query, query.float().double()))
                    rates = torch.ones_like(supports, dtype=query.dtype) if supplied_rates else None
                    with patch.object(policy, '_g_microbatch', return_value=tile):
                        expected = execution(policy, lambda: literal_g_prior(
                            policy, None, supports, query.float(), rows, rates))
                        cached = execution(policy, lambda: policy._g_prior(
                            None, supports, query.float(), rows, rates, query_history=query))
                        fallback = execution(policy, lambda: policy._g_prior(
                            None, supports, query.float(), rows, rates))
                    self.assert_exact(cached[:2], expected[:2])
                    self.assert_exact(fallback[:2], expected[:2])
                    self.assertEqual(cached[2], 0)
                    self.assertEqual(fallback[2], 1)
                    self.assertGreater(expected[2], fallback[2])

    def test_ragged_offline_reuses_one_gather_and_keeps_demo_masks_and_solve_shapes(self):
        policy = ragged_policy()
        rows = torch.tensor([3, 5, 11, 13, 19, 21])
        for leave_demo_out in (False, True):
            for tile in (1, 8):
                with self.subTest(demo=leave_demo_out, tile=tile):
                    with patch.object(policy, '_legacy_g_batch', return_value=tile):
                        expected = execution(policy, lambda: literal_offline_g(
                            policy, rows, leave_demo_out))
                        actual = execution(policy, lambda: policy._offline_g(rows, leave_demo_out))
                    self.assert_exact(actual[:2], expected[:2])
                    self.assertEqual(actual[2], 1)
                    self.assertGreater(expected[2], actual[2])

    def test_materialized_temporal_offline_reuses_queries_without_changing_projection_shapes(self):
        policy, rows, query, supports = fixture()
        excluded = rows.new_zeros(len(rows))
        for tile in (1, 2, 8):
            with self.subTest(tile=tile):
                with patch.object(policy, '_legacy_g_batch', return_value=tile):
                    expected = execution(policy, lambda: literal_offline_g(
                        policy, rows, atlas=(supports, excluded)))
                    actual = execution(policy, lambda: policy._offline_g(
                        rows, atlas=(supports, excluded)))
                    cached = execution(policy, lambda: policy._offline_g(
                        rows, atlas=(supports, excluded), query_history=query))
                self.assert_exact(actual[:2], expected[:2])
                self.assert_exact(cached[:2], expected[:2])
                self.assertEqual(actual[2], (len(rows)+tile-1)//tile)
                self.assertEqual(cached[2], 0)
                self.assertEqual(expected[2], 2*actual[2])

    def test_existing_private_fixed_override_does_not_receive_new_argument(self):
        policy, rows, query, supports = fixture(rates=(1.,))
        def legacy_override(columns, selected):
            return literal_fixed_g(policy, columns, selected)
        expected = literal_g_prior(policy, None, supports, query.float(), rows=rows)
        with patch.object(policy, '_fixed_g', side_effect=legacy_override) as override:
            actual = policy._g_prior(None, supports, query.float(), rows=rows, query_history=query)
        self.assert_exact(actual, expected)
        self.assertEqual(override.call_count, 1)

    def test_existing_private_offline_override_keeps_original_signature(self):
        policy = ragged_policy()
        rows = torch.tensor([3, 5, 11, 13, 19, 21])
        offline = policy._offline_g
        def legacy_override(selected, leave_demo_out=False, atlas=None):
            return offline(selected, leave_demo_out, atlas)
        expected = literal_offline_g(policy, rows)
        with patch.object(policy, '_offline_g', side_effect=legacy_override) as override:
            actual = policy._offline_g(rows)
        self.assert_exact(actual, expected)
        self.assertEqual(override.call_count, 4)


if __name__ == '__main__':
    unittest.main()
