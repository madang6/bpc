"""Small CPU integration: auxiliary encoder -> bank -> online representation."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from bpc.data import full_state, load_bank, make_encoder, prepare
from bpc.dexterous import contract_digest
from bpc.dp3_pointnet import sha256
from bpc.pipeline import load_config, task_configs
from scripts.fit_dp3_pointnet import fit
from tests.suites import integration

PROJECT = Path(__file__).resolve().parents[1]


@integration
class PointNetBankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        config = load_config(PROJECT / "configs/adroit_pointnet.json")
        cls.config = task_configs(config, cls.root)[0][0]
        data = cls.config["data"]
        data.update(demonstrations=2, bank=str(cls.root / "bank.pt"),
                    pointcloud_hdf5=str(cls.root / "capture.hdf5"))
        data["dp3_pointnet"]["device"] = "cpu"
        rng = np.random.default_rng(40)
        with h5py.File(data["pointcloud_hdf5"], "w") as capture:
            capture.attrs.update(suite="adroit", task="door", fitting_admissible=True,
                                 source_contract_sha256=contract_digest(data))
            for i in range(2):
                episode = capture.create_group(f"episode_{i}")
                episode["state"] = rng.normal(size=(5, 24)).astype(np.float32)
                episode["point_cloud"] = rng.normal(size=(5, 512, 3)).astype(np.float32)
                episode["actions"] = rng.uniform(-1, 1, (4, 28)).astype(np.float32)
        settings = json.loads((PROJECT / "configs/pointnet_fit.json").read_text())
        settings["data"].update(hdf5=data["pointcloud_hdf5"])
        settings["fitting"].update(device="cpu", steps=2, batch_size=2, validation_samples=2)
        settings["output_dir"] = str(cls.root / "encoder")
        fit(settings)
        checkpoint = cls.root / "encoder/final.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert "bc_state_dict" not in payload
        assert payload["fitting_purpose"] == "deployment"
        data["dp3_pointnet"].update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.temporary.cleanup()

    def test_bank_encodes_pre_action_rows_and_matches_online(self):
        bank = prepare(self.config, self.root / "run")
        trajectories, metadata = load_bank(bank)
        encode = make_encoder(self.config["data"], evaluation=True)
        with h5py.File(self.config["data"]["pointcloud_hdf5"], "r") as capture:
            for index, trajectory in enumerate(trajectories):
                episode = capture[f"episode_{index}"]
                np.testing.assert_array_equal(trajectory["Xro"].T, episode["state"][:-1])
                np.testing.assert_array_equal(trajectory["Uro"].T, episode["actions"][:])
                state = full_state(episode["state"][0], {"point_cloud": episode["point_cloud"][0]},
                                   self.config["data"], encode)
                np.testing.assert_array_equal(state, np.concatenate([
                    trajectory["Xro"][:, 0], trajectory["dp3_pointnet"][:, 0]]))
        self.assertEqual(metadata["capture_sha256"], sha256(self.config["data"]["pointcloud_hdf5"]))
        self.assertEqual(prepare(self.config, self.root / "reuse").resolve(), bank.resolve())
        manifest = json.loads((self.root / "reuse/data.json").read_text())
        self.assertTrue(manifest["reused_bank"])
        self.assertEqual(manifest["state_layout"], ["Xro", "dp3_pointnet"])
        self.assertEqual(manifest["action_layout"], "Uro")
        self.assertEqual(manifest["encoder"], metadata["encoder"])
        self.assertEqual(manifest["state_convention"], "dp3_adroit_robot_state")
        self.assertEqual(manifest["representation_check"], {})

    def test_changed_checkpoint_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["data"]["dp3_pointnet"]["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint SHA mismatch"):
            prepare(config, self.root / "wrong-checkpoint")

    def test_archived_capture_and_encoder_preserve_features(self):
        config = copy.deepcopy(self.config)
        data = config['data']
        capture = self.root / 'archived-capture.hdf5'
        shutil.copyfile(data['pointcloud_hdf5'], capture)
        with h5py.File(capture, 'r+') as source:
            source.attrs['training_admissible'] = source.attrs['fitting_admissible']
            del source.attrs['fitting_admissible']
            clouds = np.asarray(source['episode_0']['point_cloud'][:1])
        payload = torch.load(data['dp3_pointnet']['checkpoint'], weights_only=True)
        payload['format'] = 'our_dp3_pointnet_bc_v1'
        for current, old in (
                ('fitted_steps', 'trained_steps'), ('fitting_purpose', 'training_purpose'),
                ('fitting_admissible', 'training_admissible'), ('fitting_episode_ids', 'training_episode_ids')):
            payload[old] = payload.pop(current)
        payload['config']['training'] = payload['config'].pop('fitting')
        normalizers = payload['normalizers']
        normalizers['training_episode_ids'] = normalizers.pop('fitting_episode_ids')
        normalizers['point_scheme'] = 'train_limits'
        payload['source']['sha256'] = sha256(capture)
        payload['config']['data']['hdf5'] = str(capture)
        checkpoint = self.root / 'archived-pointnet.pt'
        torch.save(payload, checkpoint)
        data.update(pointcloud_hdf5=str(capture), bank=str(self.root / 'archived-bank.pt'))
        data['dp3_pointnet'].update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        np.testing.assert_array_equal(make_encoder(data)(clouds), make_encoder(self.config['data'])(clouds))
        actual = load_bank(prepare(config, self.root / 'archived-run'))[0]
        expected = load_bank(prepare(self.config, self.root / 'current-run'))[0]
        for actual_episode, expected_episode in zip(actual, expected):
            for key in expected_episode:
                np.testing.assert_array_equal(actual_episode[key], expected_episode[key])

    def test_wrong_task_checkpoint_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["data"].update(env_task="hammer", adroit_task="hammer")
        config["evaluation"]["task"] = "hammer"
        with self.assertRaisesRegex(ValueError, "does not match live input contract"):
            make_encoder(config["data"])

    def test_bank_for_another_capture_is_rejected(self):
        bank = prepare(self.config, self.root / "source")
        payload = torch.load(bank, map_location="cpu")
        payload["metadata"]["capture_sha256"] = "0" * 64
        bad = self.root / "wrong-bank.pt"
        torch.save(payload, bad)
        config = copy.deepcopy(self.config)
        config["data"]["bank"] = str(bad)
        with self.assertRaisesRegex(ValueError, "capture_sha256"):
            prepare(config, self.root / "wrong-bank")
