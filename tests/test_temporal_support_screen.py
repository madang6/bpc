"""Temporal screening must preserve ordered supports, exclusions and targets."""
import unittest
from tests.suites import cuda
from unittest.mock import patch
import torch
from bpc.temporal_screen import TemporalTopK
from bpc.bpc import Policy
from tests.support.policies import (
    method_config as _method_config, data_config as _data_config, trajectories as
    _trajectories,
)


class TemporalScreenTest(unittest.TestCase):
    device = 'cpu'

    def fixture(self, ties=False):
        gen = torch.Generator().manual_seed(93)
        banks = [torch.randn(2048, 8, dtype=torch.float64, generator=gen).to(self.device)
                 for _ in range(5)]
        if ties:
            banks = [bank[:1].expand_as(bank).clone() for bank in banks]
        starts = torch.arange(2048, device=self.device)
        valid = starts % 9 != 0
        rates = [(banks[0], None, None, None)] + [
            (bank, valid, starts - i, starts + i) for i, bank in enumerate(banks[1:], 1)]
        queries = torch.randn(23, 8, dtype=torch.float64, generator=gen).to(self.device)
        rows = torch.arange(23, device=self.device) * 17
        return rates, queries, rows, starts, starts // 128

    def test_certified_order_matches_exhaustive_minimum_with_masks(self):
        for demos_enabled in (False, True):
            rates, q, rows, starts, traj = self.fixture()
            demos = rows // 128 if demos_enabled else None
            distance = None
            for bank, valid, first, last in rates:
                d = (q @ bank.T).mul(-2).add(q.square().sum(1)[:, None]).add(bank.square().sum(1)).clamp_min(0)
                if valid is not None:
                    d.masked_fill_(~valid | ((first <= rows[:, None]) & (rows[:, None] <= last)), float('inf'))
                distance = d if distance is None else torch.minimum(distance, d)
            mask = (rows[:, None] - 3 <= starts) & (starts <= rows[:, None])
            if demos is not None:
                mask |= traj == demos[:, None]
            distance.masked_fill_(mask, float('inf'))
            expected = distance.topk(64, largest=False).indices
            got, certified, removed = TemporalTopK(rates).select(q, 64, rows, demos, starts, traj, 3)
            self.assertTrue(certified.all())
            self.assertTrue(torch.equal(got, expected))
            self.assertTrue(torch.equal(removed, mask.sum(1)))

    def test_ties_and_nonfinite_bounds_fail_closed(self):
        rates, q, rows, starts, traj = self.fixture(ties=True)
        _, certified, _ = TemporalTopK(rates).select(q, 64, rows, None, starts, traj, 3)
        self.assertFalse(certified.any())
        rates, q, rows, starts, traj = self.fixture()
        _, certified, _ = TemporalTopK(rates).select(q, 64, rows, None, starts, traj, 3,
                            reference_error=torch.full((len(q),), float('inf'), device=q.device))
        self.assertFalse(certified.any())

    def policy(self):
        m = _method_config(); m['device'] = self.device
        m['g'].update(lookup='action_ridge', supports=64, fit_dtype='float32',
                      solver='exact', constraint='affine', l1=0, future_consistency=0,
                      lookup_rates=[.5,.75,1,1.5,2], align_history=True,
                      align_future=True, state_offset=1, rate_scaled_action_dims=[0])
        m['residual']['lookup'] = 'g_supports'
        p = Policy(_trajectories(demos=40, length=40), _data_config(), m,
                   fit_residuals=False)
        width = p.history * (p.action_dim + p.state_dim)
        p.g_lookup_matrix = torch.eye(width, dtype=p.dtype, device=p.device)
        p._project_hankel()
        return p

    def test_policy_supports_targets_and_original_tile_fallback(self):
        p = self.policy()
        rows = torch.arange(10, 610, 19, device=p.device)
        q = p._queries(rows)
        expected = p._literal_g_supports(q, 64, rows, None, 7)
        screened = p._temporal_support_topk(q, 64, rows, None, 7)
        for one, two in zip(expected, screened):
            self.assertTrue(torch.equal(one, two))
        # An ambiguous query causes its complete original tile to be replayed.
        original = TemporalTopK.select
        def ambiguous(screen, *args, **kwargs):
            support, certified, removed = original(screen, *args, **kwargs)
            certified[2] = False
            return support, certified, removed
        with patch.object(TemporalTopK, 'select', ambiguous), patch.object(
                p, '_literal_g_supports', wraps=p._literal_g_supports) as replay:
            screened = p._temporal_support_topk(q, 64, rows, None, 7)
            self.assertTrue(any(torch.equal(call.args[2], rows[:7]) for call in replay.call_args_list))
        for one, two in zip(expected, screened):
            self.assertTrue(torch.equal(one, two))
        reference = p._offline_g(rows, atlas=expected)
        candidate = p._offline_g(rows, atlas=screened)
        for one, two in zip(reference, candidate):
            self.assertTrue(torch.equal(one, two))


@cuda
class CudaTemporalScreenTest(TemporalScreenTest):
    device = 'cuda'

    def test_complete_rff_fit_matches_after_screening(self):
        import copy
        method = copy.deepcopy(self.policy().config)
        method['residual'].update(lookup='g_supports', target_batch_size=128)
        method['rff'].update(features=32, dtype='float32')
        results = []
        for enabled in (False, True):
            p = Policy(_trajectories(demos=40, length=40), _data_config(), method,
                       fit_residuals=True)
            p.g_temporal_screen = enabled
            width = p.history * (p.action_dim + p.state_dim)
            p.g_lookup_matrix = torch.eye(width, dtype=p.dtype, device=p.device)
            p._project_hankel()
            p.fit()
            results.append([x.clone() for x in (p.residual_weights, p.auxiliary_residual_weights, p.omega, p.phase)])
        for reference, screened in zip(*results):
            self.assertTrue(torch.equal(reference, screened))

    def test_opt_in_and_oom_fallback(self):
        p = self.policy(); rows = torch.arange(10, 100, device=p.device); q = p._queries(rows)
        expected = p._literal_g_supports(q, 64, rows, None, 7)
        p.g_temporal_screen = False
        with patch.object(p, '_temporal_support_topk', side_effect=AssertionError('disabled')):
            p._g_support_topk(q, 64, rows, None, 7)
        p.g_temporal_screen = True
        with patch.object(p, '_temporal_support_topk', side_effect=torch.cuda.OutOfMemoryError):
            got = p._g_support_topk(q, 64, rows, None, 7)
        self.assertTrue(torch.equal(got[0], expected[0]))
