import copy
import tempfile
import unittest
from tests.suites import cuda, integration
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np
import torch

from tests.support.policies import (
    small_ridge_fit as _small_ridge_fit, action_ridge_config as _action_ridge_config,
    data_config as _data_config, method_config as _method_config, trajectories as
    _trajectories,
)
from dpc_nldl.bpc import Policy
from dpc_nldl.artifacts import _target_key, fit_policy, load_policy, prepare_lookups
from dpc_nldl.pipeline import fit_groups


def _method(offset=1):
    method = _method_config()
    method["g"].update(state_offset=offset, constraint="affine", l1=0,
                       future_consistency=0, fit_dtype="float32")
    return method


class StateOffsetTest(unittest.TestCase):
    def test_columns_match_live_queries_and_remain_inside_demonstrations(self):
        for offset in (0, 1):
            p = Policy(_trajectories(), _data_config(), _method(offset),
                       fit_residuals=False)
            columns = torch.arange(len(p.hankel_starts))
            bases = p.hankel_starts[:, None]
            past_rows = bases + torch.arange(p.history)
            future_rows = bases + p.history + torch.arange(p.horizon)
            expected = torch.cat([p.action[past_rows].flatten(1),
                                  p.state[past_rows + offset].flatten(1)], 1)
            torch.testing.assert_close(p._past_columns(columns), expected, rtol=0, atol=0)
            torch.testing.assert_close(p.pf, expected, rtol=0, atol=0)
            torch.testing.assert_close(p.uf.T, p.action[future_rows].flatten(1))
            self.assertEqual(len(columns), 4 * (8 - p.history - p.horizon + 1 - offset))
            self.assertTrue(torch.equal(p.trajectory_of[(future_rows + offset).ravel()],
                                       p.hankel_trajectory.repeat_interleave(p.horizon)))
            if offset:
                torch.testing.assert_close(p._queries(p.hankel_starts + p.history),
                                           expected, rtol=0, atol=0)

    def test_default_is_legacy_and_invalid_offsets_are_rejected(self):
        explicit = _method(0)
        omitted = copy.deepcopy(explicit)
        omitted["g"].pop("state_offset")
        a, b = [Policy(_trajectories(), _data_config(), method, fit_residuals=False)
                for method in (explicit, omitted)]
        for field in ("hankel_starts", "pf", "uf"):
            self.assertTrue(torch.equal(getattr(a, field), getattr(b, field)))
        for value in (-1, 2, .5, "1"):
            with self.assertRaisesRegex(ValueError, "state_offset"):
                Policy(_trajectories(), _data_config(), _method(value), fit_residuals=False)

    def test_packed_history_contract_and_fixed_normal_equations(self):
        p = Policy(_trajectories(), _data_config(), _method(),
                   fit_residuals=False)
        supports, rows = torch.tensor([[0, 5], [2, 8]]), torch.tensor([3, 12])
        expected = p._past_columns(supports)

        def pack(name, reference, states, actions, starts, history):
            self.assertEqual(name, "pack_history")
            self.assertTrue(states.is_contiguous())
            self.assertEqual(len(states), len(actions))
            self.assertEqual(states.data_ptr(), p.state[1:].data_ptr())
            indices = starts[:, None] + torch.arange(history)
            return torch.cat([actions[indices].flatten(1), states[indices].flatten(1)], 1)

        # Exercise native argument packing without accessing a GPU.
        with (patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True),
              patch("dpc_nldl.fast_cuda.call", side_effect=pack) as native):
            torch.testing.assert_close(p._past_columns(supports), expected)
            native.assert_called_once()
        design, query = expected.float(), p._queries(rows, torch.float32)
        coefficients = p._solve_g_normal(design @ design.transpose(1, 2),
                                         (design @ query[:, :, None]).squeeze(2))
        torch.testing.assert_close(p._fixed_g(supports, rows), coefficients)

    @cuda
    def test_native_shifted_views_match_explicit_history_at_terminal_column(self):
        from dpc_nldl.fast_cuda import call, extension
        method = _method()
        method["device"] = "cuda"
        p = Policy(_trajectories(), _data_config(), method,
                   fit_residuals=False)
        if extension(p.state.device.index) is None:
            self.skipTest("native extension unavailable")
        columns = torch.tensor([0, len(p.hankel_starts) - 1], device=p.device)
        rows = p.hankel_starts[columns, None] + torch.arange(p.history, device=p.device)
        for dtype in (torch.float64, torch.float32):
            expected = torch.cat([p.action[rows].to(dtype).flatten(1),
                                  p.state[rows + 1].to(dtype).flatten(1)], 1)
            with patch("dpc_nldl.fast_cuda.call", wraps=call) as native:
                actual = p._past_columns(columns, dtype)
                native.assert_called_once()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @integration
    def test_sharded_fit_reload_preserves_shared_supports_and_scalar_batch_outputs(self):
        method = _action_ridge_config()
        method["g"].update(_method()["g"])
        method["residual"].update(lookup="g_supports", target_batch_size=16)
        with tempfile.TemporaryDirectory(prefix="state-offset-") as folder:
            root = Path(folder)
            config = {
                "experiment": {"name": "offset", "cache_dir": str(root / "cache")},
                "analysis": {"reference_method": "metric"}, "data": _data_config(),
                "method": method, "methods": {
                    "metric": {"base": "gip_direct", "g_lookup": "action_ridge"}},
            }
            legacy = copy.deepcopy(config)
            legacy["method"]["g"].pop("state_offset")
            self.assertNotEqual(fit_groups(config)[0]["key"], fit_groups(legacy)[0]["key"])
            self.assertNotEqual(_target_key(config, method), _target_key(legacy, legacy["method"]))
            torch.save({"data": _trajectories(demos=8, length=12)}, root / "bank.pt")
            with _small_ridge_fit():
                prepare_lookups(config, root)
            fit_policy(config, root, shard="0/2")
            fit_policy(config, root, shard="1/2")
            with patch.object(Policy, "_g_support_topk", side_effect=AssertionError("support refit")):
                fit_policy(config, root)
            p = load_policy(config, root, "metric")
            with self.assertRaisesRegex(ValueError, "different run config"):
                load_policy(legacy, root, "metric")
            for path in (root / "fit_parts").glob("*/*.pt"):
                part = torch.load(path)
                rows = part["rows"]
                prior, _, _, neighbors = p._offline_g(rows)
                torch.testing.assert_close(part["targets"], p.action[rows] - prior[:, :1])
                torch.testing.assert_close(part["g_neighbors"], neighbors)
                starts = neighbors - p.history + 1 - p.g_state_offset
                supports = torch.searchsorted(p.hankel_starts, starts)
                self.assertTrue(torch.equal(p.hankel_starts[supports], starts))
                excluded = p._hankel_exclusion(p.hankel_starts, rows)
                self.assertFalse(bool(excluded.gather(1, supports).any()))
            rows = torch.tensor([3, 17, 29])
            queries = p._queries(rows)
            actions = queries[:, :p.history].reshape(-1, p.history, 1)
            states = queries[:, p.history:].reshape(-1, p.history, p.state_dim)
            output, trace = p.direct_actions(actions, states)
            supports = trace["hankel_supports"]
            endpoints = p.hankel_starts[supports] + p.history
            expected_residual = p._query_features(states[:, -1], endpoints) @ p.W_gip
            torch.testing.assert_close(trace["residual_z"], expected_residual)
            futures = endpoints[:, :, None] + 1 + torch.arange(p.horizon)
            expected_states = torch.einsum("bkhs,bk->bhs", p.state[futures, :p.proprio_dim],
                                           trace["g_coefficients"])
            torch.testing.assert_close(trace["predicted_proprio_z"], expected_states)
            for i in range(len(rows)):
                for scalar in (False, True):
                    p.reset()
                    raw_actions = actions[i] * p.action_scale + p.action_mean
                    raw_state = states[i, -1] * p.state_scale + p.state_mean
                    p._state_history = [states[i, 0], *states[i, :-1]]
                    p._action_history = [raw_actions[0], *raw_actions[:-1]]
                    with patch.object(Policy, "batched_direct", new_callable=PropertyMock,
                                      return_value=not scalar):
                        actual = p.action_for(raw_state.numpy(), raw_actions[-1].numpy())
                    np.testing.assert_allclose(actual, output[i].numpy(), rtol=1e-9, atol=1e-9)
                    np.testing.assert_allclose(p.last_trace["predicted_proprio_z"],
                                               expected_states[i].numpy(), rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
