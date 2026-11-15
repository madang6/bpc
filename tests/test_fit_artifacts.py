import tempfile
import unittest
from tests.suites import integration
from pathlib import Path
from unittest.mock import patch

import torch


from dpc_nldl.bpc import Policy
from dpc_nldl.artifacts import fit_policy, load_policy, prepare_lookups
from dpc_nldl.pipeline import fit_groups, method_variants
from tests.support.policies import (
    small_ridge_fit as _small_ridge_fit, action_ridge_config as _action_ridge_config,
    data_config as _data_config, method_config as _method_config, trajectories as
    _trajectories,
)


class FitArtifactTest(unittest.TestCase):
    @integration
    def test_target_parts_bound_batches_and_recover_overlong_cache(self):
        with tempfile.TemporaryDirectory(prefix="dpc-target-parts-") as folder:
            root = Path(folder)
            method = _action_ridge_config()
            method["g"]["future_consistency"] = 0
            method["residual"]["target_batch_size"] = 6
            method["residual"]["lookup"] = "g_supports"
            config = {
                "experiment": {"name": "parts", "cache_dir": str(root / "cache")},
                "analysis": {"reference_method": "metric"},
                "data": _data_config(),
                "method": method,
                "methods": {"metric": {
                    "base": "gip_direct", "g_lookup": "action_ridge"}},
            }
            torch.save({"data": _trajectories(demos=8, length=12)}, root / "bank.pt")
            with patch("dpc_nldl.fitting.FIT_ROWS", 4):
                with _small_ridge_fit():
                    prepare_lookups(config, root)
                fit_policy(config, root, shard="0/2")
                fit_policy(config, root, shard="1/2")
                paths = sorted((root / "fit_parts").glob("*/*.pt"))
                self.assertTrue(paths)
                self.assertTrue(all(len(torch.load(path)["rows"]) <= 6 for path in paths))
                rows = torch.cat([torch.load(path)["rows"] for path in paths])
                self.assertEqual(len(rows), len(rows.unique()))
                payload = torch.load(paths[0])
                for name, value in payload.items():
                    payload[name] = torch.cat([value, value[-1:]])
                torch.save(payload, paths[0])
                fit_policy(config, root)
            self.assertIsNotNone(load_policy(config, root, "metric").W_gip)

    @integration
    def test_invalid_staged_lookup_is_rebuilt_then_reused(self):
        with tempfile.TemporaryDirectory(prefix="dpc-action-ridge-") as folder:
            root = Path(folder)
            method = _action_ridge_config()
            method["g"]["future_consistency"] = 0
            method["residual"]["lookup"] = "g_supports"
            config = {
                "experiment": {"name": "metric", "cache_dir": str(root / "cache")},
                "analysis": {"reference_method": "metric"},
                "data": _data_config(),
                "method": method,
                "methods": {"metric": {
                    "base": "gip_direct", "g_lookup": "action_ridge"}},
            }
            torch.save({"data": _trajectories(demos=8, length=12)}, root / "bank.pt")
            with _small_ridge_fit():
                prepare_lookups(config, root)
            path = next((root / "fit_parts").glob("*__lookup.pt"))
            valid = torch.load(path, map_location="cpu")
            for missing in (None, "g_lookup_matrix", "state_scale"):
                with self.subTest(missing=missing):
                    if missing is None:
                        path.write_bytes(b"unreadable lookup")
                    else:
                        torch.save({key: value for key, value in valid.items()
                                    if key != missing}, path)
                    with self.assertRaisesRegex(RuntimeError, "prepare learned lookup"):
                        fit_policy(config, root, shard="0/2")
                    with _small_ridge_fit():
                        prepare_lookups(config, root)
                    fit_policy(config, root, shard="0/2")
            with patch.object(
                    Policy, "_fit_action_ridge_lookup",
                    side_effect=AssertionError("metric refitted after preparation")):
                prepare_lookups(config, root)
                fit_policy(config, root, shard="0/2")
                fit_policy(config, root, shard="1/2")
                with patch.object(
                        Policy, "_g_support_topk",
                        side_effect=AssertionError("G supports searched twice")):
                    fit_policy(config, root)
            self.assertTrue(all(
                "g_neighbors" in torch.load(path, map_location="cpu")
                for path in (root / "fit_parts").glob("*/*.pt")))
            policy = load_policy(config, root, "metric")
            self.assertEqual(policy.g_lookup_matrix.shape[1], 2)

    def test_action_ridge_variant_gets_distinct_fit(self):
        config = {
            "data": _data_config(),
            "method": _action_ridge_config(select=False),
            "methods": {
                "past": {"base": "gip_direct"},
                "metric": {"base": "gip_direct", "g_lookup": "action_ridge"},
                "uniform": {"base": "gip_direct", "g_weights": "uniform"},
            },
        }
        variants = method_variants(config)
        self.assertEqual(variants["past"]["fit"], "default")
        self.assertEqual(variants["metric"]["fit"], "action_ridge")
        self.assertEqual(variants["uniform"]["fit"], "default_uniform")
        groups = {group["name"]: group for group in fit_groups(config)}
        self.assertEqual(groups["default"]["method"]["g"].get("lookup", "past"), "past")
        self.assertEqual(groups["default"]["method"]["g"].get("weights", "optimized"), "optimized")
        self.assertEqual(
            groups["action_ridge"]["method"]["g"]["lookup"], "action_ridge")
        self.assertEqual(
            groups["default_uniform"]["method"]["g"]["weights"], "uniform")
        self.assertNotEqual(groups["default"]["key"], groups["default_uniform"]["key"])

    @integration
    def test_action_ridge_fits_once_then_shards_targets_and_residual(self):
        with tempfile.TemporaryDirectory(prefix="dpc-action-ridge-") as folder:
            root = Path(folder)
            config = {
                "experiment": {"name": "metric", "cache_dir": str(root / "cache")},
                "analysis": {"reference_method": "metric"},
                "data": _data_config(),
                "method": _action_ridge_config(select=False),
                "methods": {"metric": {
                    "base": "gip_direct", "g_lookup": "action_ridge"}},
            }
            torch.save({"data": _trajectories()}, root / "bank.pt")
            with _small_ridge_fit():
                prepare_lookups(config, root)
            with patch.object(
                    Policy, "_fit_action_ridge_lookup",
                    side_effect=AssertionError("metric refitted after preparation")):
                fit_policy(config, root, shard="0/2")
                fit_policy(config, root, shard="1/2")
                fit_policy(config, root)
            self.assertTrue(any((root / "fit_parts").glob("*/*.pt")))
            policy = load_policy(config, root, "metric")
            self.assertIsNotNone(policy.g_lookup_matrix)
            self.assertIsNotNone(policy.W_gip)

    @integration
    def test_fit_round_trip_and_reuses_cache(self):
        with tempfile.TemporaryDirectory(prefix="dpc-fit-variants-") as folder:
            root = Path(folder)
            config = {
                "experiment": {"name": "smoke", "cache_dir": str(root / "cache")},
                "analysis": {"reference_method": "gip_direct"},
                "data": _data_config(),
                "method": _method_config(),
                "methods": ["gip_direct"],
            }
            torch.save({"data": _trajectories()}, root / "bank.pt")
            fit_policy(config, root)
            self.assertEqual(len(list((root / "policies").glob("*.pt"))), 1)
            load_policy(config, root, "gip_direct")
            for path in (root / "policies").glob("*.pt"):
                path.unlink()
            (root / "policy.pt").unlink()
            with patch.object(Policy, "fit", side_effect=AssertionError("cache miss")):
                fit_policy(config, root)


if __name__ == "__main__":
    unittest.main()
