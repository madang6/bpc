import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from tests.suites import integration
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PROJECT = Path(__file__).resolve().parents[1]
from scripts import run_experiment as RUNNER

from dpc_nldl.data import evaluation_seed, prepare
from dpc_nldl.pipeline import (ROBOSUITE_RENDER_CONVENTION, aggregate, fit_groups,
                               load_config, run, task_configs)
from dpc_nldl.robosuite import task
from dpc_nldl.tracking import publish


class DistributedFitTest(unittest.TestCase):
    @integration
    @unittest.skipUnless(shutil.which("rsync"), "rsync unavailable")
    def test_remote_bank_staging_reuses_storage_and_preserves_prior_runs(self):
        execute = subprocess.run
        def local_rsync(command, **kwargs):
            command = list(command)
            command[-1] = command[-1].removeprefix("remote:")
            return execute(command, **kwargs, stdout=subprocess.DEVNULL)
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                RUNNER.subprocess, "run", side_effect=local_rsync):
            base = Path(folder)
            bank = base / "bank.pt"
            bank.write_bytes(b"original bank")
            staged = []
            for name in ("one", "two", "changed"):
                if name == "changed":
                    bank.write_bytes(b"replacement bank")
                run = base / "local" / name
                run.mkdir(parents=True)
                (run / "bank.pt").symlink_to(bank)
                root = base / "remote" / name
                (root / "experiments" / name).mkdir(parents=True)
                RUNNER._stage_run("remote", str(root), run)
                staged.append(root / "experiments" / name / "bank.pt")
            self.assertEqual(staged[0].stat().st_ino, staged[1].stat().st_ino)
            self.assertEqual(staged[0].read_bytes(), b"original bank")
            self.assertEqual(staged[2].read_bytes(), b"replacement bank")
            self.assertEqual(staged[2].stat().st_ino,
                             (base / "remote/banks/bank.pt").stat().st_ino)

    def test_headline_preserves_eight_task_protocol_and_active_dependencies(self):
        config = load_config(PROJECT / "configs/robosuite_fisher_sparsemax_local.json")
        jobs = RUNNER._jobs(config)
        self.assertEqual(len(jobs), 8)
        for scoped, _, _ in jobs:
            self.assertTrue(scoped["methods"]["gip_lda"]["recovery"])
            self.assertEqual(scoped["method"]["g"]["lda"]["teacher"], "sparsemax")
            self.assertEqual(scoped["method"]["g"]["lookup_rates"], [.5, .75, 1, 1.5, 2])
            self.assertEqual(scoped["data"]["demonstrations"], 1000)
            self.assertEqual(scoped["evaluation"]["init_index_stop"], 300)
            square = scoped["evaluation"]["task"] == "square_d0"
            self.assertEqual(scoped["method"]["history"], 5 if square else 10)

    def test_robosuite_tasks_require_collision_hidden_banks(self):
        config = {
            "experiment": {"name": "render"}, "runtime": {}, "methods": [],
            "method": {}, "evaluation": {},
            "analysis": {"reference_method": "gip_direct"}, "data": {},
            "benchmarks": [{"kind": "robosuite", "dataset_dir": "/data",
                            "bank_dir": "/cache",
                            "construction": "robosuite_recorded_state_rerender_v1",
                            "tasks": ["square_d0"]}],
        }
        scoped, _, _ = task_configs(config, "/run")[0]
        self.assertEqual(scoped["data"]["render_convention"],
                         ROBOSUITE_RENDER_CONVENTION)
        self.assertEqual(scoped["data"]["construction"],
                         "robosuite_recorded_state_rerender_v1")

    @integration
    def test_collision_visible_bank_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bank = root / "old.pt"
            torch.save({"data": [], "metadata": {}}, bank)
            config = {
                "data": {"bank": str(bank), "render_convention":
                         ROBOSUITE_RENDER_CONVENTION},
                "evaluation": {}, "method": {},
            }
            with self.assertRaisesRegex(ValueError, "regenerate the bank"):
                prepare(config, root / "run")

    @integration
    def test_wrong_trajectory_construction_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = Path(folder) / "old.pt"
            torch.save({"data": [], "metadata": {
                "render_convention": ROBOSUITE_RENDER_CONVENTION,
                "trajectory_construction": "robosuite_action_replay_v1"}}, bank)
            config = {"data": {"bank": str(bank), "render_convention":
                                ROBOSUITE_RENDER_CONVENTION,
                                "construction":
                                "robosuite_recorded_state_rerender_v1"},
                      "evaluation": {}, "method": {}}
            with self.assertRaisesRegex(ValueError, "trajectory_construction"):
                prepare(config, Path(folder) / "run")

    def test_fit_allocation_uses_current_variant_schema(self):
        jobs = [({"methods": ["gip_direct"]}, None, None)]
        self.assertEqual(RUNNER._fit_allocation(jobs, [("endor", "0", 1),
                                                       ("local", "0", 1)]),
                         [["default"], []])

        jobs = [({"methods": {
            "past": {"base": "gip_direct"},
            "metric": {"base": "gip_direct", "g_lookup": "action_ridge"},
            "fisher": {
                "base": "gip_direct", "g_lookup": "lda"},
        }}, None, None)]
        self.assertEqual(RUNNER._fit_allocation(
            jobs, [("endor", "0", 1), ("local", "0", 1)]),
            [["default", "lda"], ["action_ridge"]])
        self.assertEqual(RUNNER._fit_order(jobs, lookups_only=True),
                         ["action_ridge", "lda"])

    def test_internal_fit_selection_stays_in_single_runner(self):
        command = RUNNER._command("/tmp/run", "fit", fit="a,b")
        self.assertEqual(command[-2:], ["--fit", "a,b"])
        command = RUNNER._command("/tmp/run", "prefit", fit="action_ridge")
        self.assertEqual(command[-2:], ["--fit", "action_ridge"])

    def test_fit_allocation_distributes_task_and_group_pairs(self):
        variants = {
            "past": {"base": "gip_direct"},
            "learned": {"base": "gip_direct", "g_lookup": "lda"},
        }
        jobs = [({"methods": variants}, None, "robosuite/square_d0"),
                ({"methods": {"learned": variants["learned"]}}, None, "robosuite/stack_d0")]
        workers = [("local", "0", 1), ("local", "1", 1)]
        self.assertEqual(RUNNER._fit_allocation(jobs, workers), [
            ["robosuite/square_d0:default", "robosuite/stack_d0:lda"],
            ["robosuite/square_d0:lda"]])
        self.assertEqual(RUNNER._fit_order(jobs, lookups_only=True), [
            "robosuite/square_d0:lda", "robosuite/stack_d0:lda"])
        self.assertEqual(RUNNER._fit_allocation(jobs, workers, "lda"), [
            ["robosuite/square_d0:lda"],
            ["robosuite/stack_d0:lda"]])

    @integration
    def test_task_qualified_prefit_and_final_fit_preserve_target_shards(self):
        config = load_config(PROJECT / "configs/robosuite_fisher_sparsemax_local.json")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps(config))
            for stage, target in (("prefit", "prepare_lookups"), ("fit", "fit_policy")):
                with mock.patch(f"dpc_nldl.bpc.{target}") as fitted:
                    run(run_dir=root, stage=stage,
                        fit="robosuite/stack_d0:lda")
                self.assertEqual(fitted.call_count, 1)
                self.assertEqual(fitted.call_args.args[0]["evaluation"]["task"], "stack_d0")
                self.assertEqual(fitted.call_args.args[-1], "lda")
            with mock.patch("dpc_nldl.bpc.fit_policy") as fitted:
                run(run_dir=root, stage="fit", shard="1/2", fit="lda")
            self.assertEqual(fitted.call_count, 8)
            self.assertTrue(all(call.args[-2:] == ("1/2", "lda")
                                for call in fitted.call_args_list))
            with self.assertRaisesRegex(ValueError, "unknown fit tasks"):
                run(run_dir=root, stage="fit", fit="robosuite/missing:lda")

    @integration
    def test_remote_fit_collection_uses_only_owned_task_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = [(None, root / name, f"robosuite/{name}")
                    for name in ("square_d0", "stack_d0", "threading_d0")]
            selected = {"endor": ["robosuite/stack_d0:default",
                                   "robosuite/stack_d0:lda"]}
            for products in (("fit_parts",), ("policies",)):
                with mock.patch.object(RUNNER.subprocess, "run") as transfer:
                    RUNNER._collect({"endor": "/remote"}, jobs, root, products, selected)
                self.assertEqual(transfer.call_count, 1)
                source = transfer.call_args.args[0][2]
                self.assertIn("/stack_d0/", source)
                self.assertNotIn("policy.pt", source)

    @integration
    def test_fit_dispatch_assigns_both_gpus_and_preserves_lora_early_return(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(run=folder, stage="fit", shard=None,
                                   benchmark=None, detach=False, workers="local=0,local=1",
                                   method=None, fit=None)
            for lora in (False, True):
                recipe = "pi05_lora" if lora else "fisher_sparsemax_local"
                config = load_config(PROJECT / f"configs/robosuite_{recipe}.json")
                with (mock.patch.object(RUNNER, "_run_config", return_value=config),
                      mock.patch.object(RUNNER, "_launch") as launch,
                      mock.patch.object(RUNNER, "_wait"),
                      mock.patch.object(RUNNER.subprocess, "run") as assembly):
                    RUNNER.distribute(args)
                self.assertEqual(launch.call_count, 2 if lora else 6)
                self.assertEqual(assembly.call_count, 0 if lora else 1)
                if not lora:
                    for call in [*launch.call_args_list[:2], *launch.call_args_list[-2:]]:
                        self.assertEqual(len(call.kwargs["fit"].split(",")), 4)

    def test_fit_cache_mounts_only_for_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {
                "experiment": {"output_dir": str(root / "runs"),
                               "cache_dir": str(root / "cache")}}
            runtime = {"gpus": "all", "shm_size": "1g", "mujoco_gl": "egl",
                       "readonly_data_root": "/data", "image": "runtime"}
            jobs = [({"runtime": runtime, "data": {"feature_keys": []}}, None, None)]
            for stage, mounted in (("fit", True), ("evaluate", False)):
                args = SimpleNamespace(run=None, stage=stage, shard=None, benchmark=None)
                with mock.patch.object(RUNNER, "_run_config", return_value=config), \
                        mock.patch.object(RUNNER, "_jobs", return_value=jobs), \
                        mock.patch.object(RUNNER.subprocess, "run") as launch, \
                        mock.patch.dict("os.environ", {}, clear=True):
                    RUNNER.host(args, ["true"])
                command = launch.call_args.args[0]
                self.assertIn(f"{root / 'runs'}:{root / 'runs'}", command)
                self.assertEqual(f"{root / 'cache'}:{root / 'cache'}" in command,
                                 mounted)

    @integration
    def test_serial_reset_catalog_pins_seed_and_initializer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "resets.pt"
            torch.save({"model": "xml", "states": {
                0: torch.tensor([1]), 1: torch.tensor([2])}}, path)
            config = {
                "data": {"reset_catalog": str(path)},
                "evaluation": {"task": "square_d0", "reset_protocol":
                               "serial_seed_catalog", "episode_seed": 0,
                               "init_index_start": 0, "init_index_stop": 2},
            }
            self.assertEqual(evaluation_seed(config, 1), 0)
            self.assertEqual(int(task(config)["initializers"][1]["states"][0]), 2)

    def test_independent_resets_use_all_configured_seeds(self):
        config = {"evaluation": {"reset_protocol": "independent_seed",
                                 "episode_seed": 100000}}
        self.assertEqual([evaluation_seed(config, index) for index in range(300)],
                         list(range(100000, 100300)))

    @integration
    def test_coordinator_stage_sequence(self):
        config = load_config(PROJECT / "configs/robosuite_pi05_lora.json")
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(run=folder, detach=False, method=None)
            with mock.patch.object(RUNNER, "_run_config", return_value=config), \
                    mock.patch.object(RUNNER.subprocess, "run") as launch:
                RUNNER.run_all(args)
            self.assertEqual([call.args[0][-1] for call in launch.call_args_list],
                             ["data", "fit", "evaluate"])

    @integration
    def test_lora_remote_staging_uses_policy_assets_without_an_encoder_selector(self):
        config = load_config(PROJECT / "configs/robosuite_pi05_lora.json")
        config["runtime"]["readonly_data_root"] = "/inputs"
        config["benchmarks"][0]["dataset_dir"] = "/inputs/demos"
        config["data"]["pi05"].update(checkpoint="/inputs/weights", tokenizer="/inputs/tokenizer")
        self.assertNotIn("encoder", config["data"])
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(run=folder, stage="fit", shard=None, benchmark=None,
                                   detach=False, workers="remote=0", method=None, fit=None)
            with (mock.patch.object(RUNNER, "_run_config", return_value=config),
                  mock.patch.object(RUNNER, "_stage_run"),
                  mock.patch.object(RUNNER, "_launch"), mock.patch.object(RUNNER, "_wait"),
                  mock.patch.object(RUNNER.subprocess, "check_output", return_value="/home/worker"),
                  mock.patch.object(RUNNER.subprocess, "run") as transfer):
                RUNNER.distribute(args)
            commands = [call.args[0] for call in transfer.call_args_list]
            sources = [command[2] for command in commands if command[:2] == ["rsync", "-a"]]
            self.assertIn("/inputs/weights", sources)
            self.assertIn("/inputs/tokenizer", sources)
            self.assertFalse(any(".r3m" in source for source in sources))

    def test_pi05_lora_is_one_config_with_exact_training_contract(self):
        config = json.loads((PROJECT / "configs" /
                             "robosuite_pi05_lora.json").read_text())
        self.assertEqual([task["name"] for task in config["benchmarks"][0]["tasks"]],
                         ["square_d0", "stack_d0", "threading_d0"])
        self.assertEqual(config["methods"], ["pi05_lora"])
        self.assertEqual(config["policy"]["training"], {
            "batch_size": 32, "steps": 30000, "seed": 42, "workers": 2,
            "save_interval": 1000, "wandb_project": "dpc-nldl-pi05-lora"})
        self.assertEqual(config["policy"]["image_keys"],
                         ["base_0_rgb", "left_wrist_0_rgb"])
        self.assertEqual((config["policy"]["action_horizon"],
                          config["evaluation"]["episode_seed"],
                          config["evaluation"]["init_index_stop"]), (10, 100000, 300))
        self.assertEqual(config["runtime"]["fit_workers"], [
            "coruscant.stanford.edu=3", "coruscant.stanford.edu=3"])

    def test_native_workers_remap_readonly_data_paths(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(
                "os.environ", {"DPC_NATIVE": "1", "DPC_DATA_ROOT": "/scratch/data"}):
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({
                "runtime": {"readonly_data_root": "/media/admin/data"},
                "data": {"raw_hdf5": "/media/admin/data/demo.hdf5"},
            }))
            config = load_config(path)
        self.assertEqual(config["runtime"]["readonly_data_root"], "/scratch/data")
        self.assertEqual(config["data"]["raw_hdf5"], "/scratch/data/demo.hdf5")

    def test_native_coordinator_and_workers_use_configured_runtime(self):
        config = load_config(PROJECT / "configs/robosuite_fisher_sparsemax_local.json")
        config["runtime"].update(native_python="/opt/pytorch/bin/python",
                                 native_env={"PYTHONPATH": "{project}/src"})
        args = SimpleNamespace(run="/tmp/run", config=None, stage="analyze",
                               benchmark=None, method=None, fit=None, shard=None)
        for environment, coordinator in (({}, True),
                ({"DPC_WORKER": "1", "CUDA_VISIBLE_DEVICES": "1"}, False)):
            config["runtime"]["native_coordinator"] = coordinator
            with (mock.patch.dict("os.environ", environment, clear=True),
                  mock.patch.object(RUNNER, "_run_config", return_value=config),
                  mock.patch.object(RUNNER.subprocess, "run") as launch):
                RUNNER.host(args)
            self.assertEqual(launch.call_args.args[0][0], "/opt/pytorch/bin/python")
            actual = launch.call_args.kwargs["env"]
            self.assertEqual(actual["DPC_NATIVE"], "1")
            self.assertEqual(actual["PYTHONPATH"], str(PROJECT / "src"))
            self.assertEqual(actual.get("MUJOCO_EGL_DEVICE_ID"), environment.get("CUDA_VISIBLE_DEVICES"))

    def test_evaluator_parallelism_does_not_change_fit_identity(self):
        config = load_config(PROJECT / "configs/robosuite_fisher_sparsemax_local.json")
        jobs = RUNNER._jobs(config)
        self.assertEqual(len(jobs), 8)
        for scoped, _, _ in jobs:
            expected = fit_groups(scoped)
            scoped["evaluation"]["env_workers"] = 1
            self.assertEqual(fit_groups(scoped), expected)

    @integration
    def test_pi05_lora_data_shards_tasks_not_demonstrations(self):
        config = json.loads((PROJECT / "configs" /
                             "robosuite_pi05_lora.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps(config))
            with mock.patch("dpc_nldl.pi05_finetune.prepare") as prepare:
                run(run_dir=root, stage="data", shard="1/3")
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(prepare.call_args.args[0]["evaluation"]["task"], "stack_d0")

    @integration
    def test_single_task_gets_canonical_root_summary(self):
        config = {
            "experiment": {"name": "one-task"},
            "evaluation": {"benchmark": "robosuite", "task": "stack_d0"},
        }
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "summary.json").write_text(json.dumps({"methods": {}}))
            aggregate(config, run)
            self.assertEqual(json.loads((run / "summary.json").read_text()), {
                "experiment": "one-task",
                "tasks": {"robosuite/stack_d0": {"methods": {}}},
            })

    @integration
    def test_wandb_publishes_only_config_and_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / "tasks" / "robosuite" / "square"
            task.mkdir(parents=True)
            for path in (root / "config.json", root / "summary.json",
                         task / "results.csv"):
                path.write_text("{}")
            run = mock.Mock()
            wandb = SimpleNamespace(
                login=mock.Mock(), init=mock.Mock(return_value=run),
                Artifact=mock.Mock(side_effect=lambda name, type: SimpleNamespace(
                    name=name, type=type, add_file=mock.Mock())))
            with mock.patch.dict(sys.modules, {"wandb": wandb}), \
                    mock.patch.dict("os.environ", {"WANDB_API_KEY": "key"}, clear=True):
                publish({}, root)
            self.assertEqual([call.kwargs["type"]
                              for call in wandb.Artifact.call_args_list],
                             ["config", "results"])
            self.assertEqual(run.log_artifact.call_count, 2)
            run.log.assert_not_called()
            run.finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
