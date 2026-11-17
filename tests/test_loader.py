"""Archived names are accepted only when reading saved runs and checkpoints."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from bpc.artifacts import load_policy
from bpc.bpc import Policy
from bpc.loader import compatible_spec, load_run_config, normalize_payload
from bpc.pipeline import fit_groups
from tests.suites import integration
from tests.support.policies import data_config, method_config, trajectories


def archived_spec(spec):
    spec = deepcopy(spec)
    spec.update(fit_version=3, fit_posterior=False)
    method = spec['method']
    method['spc'] = dict(horizon=method.pop('continuation_horizon'), ridge=1,
                         tracking_weight=1, prior_weight=10, dynamics_state_dims=1)
    method['g']['selector'] = 'legacy_full'
    method['temporal_neighbors']['selector'] = 'legacy_temporal'
    return spec


class LoaderTest(unittest.TestCase):
    def setUp(self):
        self.config = dict(data=data_config(), method=method_config(), methods=['bpc'])

    def test_payload_upgrade_preserves_tensor_identity_and_input(self):
        weights = torch.ones(2, 1)
        payload = {'W': weights, 'W_knn': weights, 's2_gip': 1., 's2_knn': 2.,
                   'fit': {'gip_residual_variance': 1., 'knn_residual_variance': 2.,
                           'training_relative_error': .1, 'g_lookup': {'training_r2': .8}}}
        before = deepcopy(payload)
        loaded = normalize_payload(payload)
        self.assertIs(loaded['residual_weights'], weights)
        self.assertIs(loaded['auxiliary_residual_weights'], weights)
        self.assertEqual(loaded['residual_variance'], 1.)
        self.assertEqual(loaded['auxiliary_residual_variance'], 2.)
        self.assertEqual(loaded['fit'], {'residual_variance': 1., 'auxiliary_residual_variance': 2.,
                                       'fitting_relative_error': .1, 'g_lookup': {'fitting_r2': .8}})
        torch.testing.assert_close(payload, before)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            normalize_payload({**payload, 'residual_weights': weights})

    def test_spec_upgrade_rejects_different_fits_and_unknown_metadata(self):
        expected = fit_groups(self.config)[0]['spec']
        archived = archived_spec(expected)
        before = deepcopy(archived)
        self.assertTrue(compatible_spec(archived, expected))
        self.assertEqual(archived, before)
        for section, field, value in (
                ((), 'fit_version', 99), ((), 'fit_posterior', True),
                (('method', 'spc'), 'horizon', 9),
                (('method', 'g'), 'selector', 'unknown'),
                (('method', 'residual'), 'target_execution', 'batched')):
            with self.subTest(field=field):
                changed = deepcopy(archived)
                target = changed
                for key in section:
                    target = target[key]
                target[field] = value
                self.assertFalse(compatible_spec(changed, expected))

    def test_live_methods_do_not_accept_archived_names(self):
        for name in ('gip_direct', 'deepc_g_direct', 'gip_spc', 'deepc_g_spc'):
            with self.subTest(name=name):
                self.config['methods'] = [name]
                with self.assertRaisesRegex(ValueError, 'unknown configured method'):
                    fit_groups(self.config)
                with self.assertRaisesRegex(ValueError, 'unknown method'):
                    Policy.__new__(Policy).select_method(name)

    @integration
    def test_run_and_episode_loading_preserve_completed_rollouts(self):
        from bpc.analyze import _rows
        from bpc.evaluate import _read_episode
        from bpc.pipeline import load_config
        from run_experiment import _run_config
        from types import SimpleNamespace

        archived = deepcopy(self.config)
        archived['method'] = archived_spec(fit_groups(self.config)[0]['spec'])['method']
        archived.update(methods={'gip_lda': {'base': 'gip_direct', 'g_lookup': 'lda'}},
                        analysis={'reference_method': 'gip_lda'},
                        runtime={'image': 'dpc-nldl-robosuite:latest'},
                        evaluation={'init_index_start': 0, 'init_index_stop': 1},
                        benchmarks=[{'methods': {'gip_nnc': {'base': 'deepc_g_direct'}},
                                     'method_base': 'deepc_g_direct'}])
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'config.json'
            path.write_text(json.dumps(archived))
            loaded = load_run_config(path)
            self.assertEqual(loaded['methods']['bpc_fisher']['base'], 'bpc')
            self.assertEqual(loaded['analysis']['reference_method'], 'bpc_fisher')
            self.assertEqual(loaded['method'], self.config['method'])
            self.assertEqual(loaded['runtime']['image'], 'bpc-robosuite:latest')
            self.assertEqual(loaded['benchmarks'][0]['methods']['bpc_nnc']['base'], 'bpc_nnc')
            self.assertEqual(loaded['benchmarks'][0]['method_base'], 'bpc_nnc')
            self.assertEqual(load_config(path, recorded=True), loaded)
            self.assertEqual(_run_config(SimpleNamespace(run=root)), loaded)
            self.assertEqual(json.loads(path.read_text()), archived)
            episode = root / 'episodes/gip_lda/000.json'
            episode.parent.mkdir(parents=True)
            episode.write_text(json.dumps(dict(method='gip_lda', init_index=0, success=True)))
            expected = dict(method='bpc_fisher', init_index=0, success=True)
            self.assertEqual(_read_episode(root / 'episodes/bpc_fisher/000.json', 0, 'bpc_fisher'), expected)
            self.assertEqual(_rows(loaded, root, 'bpc_fisher', required=True), [expected])

    @integration
    def test_old_checkpoints_load_in_single_and_multiple_fit_runs(self):
        config = deepcopy(self.config)
        config['methods'] = {'bpc': {'base': 'bpc'},
                             'bpc_nnc': {'base': 'bpc_nnc', 'g_weights': 'uniform'}}
        bank = trajectories()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'policies').mkdir()
            for group, name in zip(fit_groups(config), config['methods']):
                with self.subTest(method=name):
                    model = Policy(bank, config['data'], group['method'],
                                   fit_residuals=group['spec']['fit_residuals']).fit()
                    payload = model.fitted_payload(group['spec'])
                    payload['spec'] = archived_spec(payload['spec'])
                    for current, old in (
                            ('residual_weights', 'W'), ('auxiliary_residual_weights', 'W_knn'),
                            ('residual_variance', 's2_gip'), ('auxiliary_residual_variance', 's2_knn')):
                        if current in payload:
                            payload[old] = payload.pop(current)
                    path = root / 'policies' / f"{group['name']}__old_hash.pt"
                    torch.save(payload, path)
                    actions, states = model.action[:2][None], model.state[:2][None]
                    expected = model.direct_actions(actions, states)
                    restored = load_policy(config, root, name, trajectories=bank)
                    torch.testing.assert_close(restored.direct_actions(actions, states), expected, rtol=0, atol=0)
                    fresh_payload = restored.fitted_payload(group['spec'])
                    self.assertFalse({'W', 'W_knn', 's2_gip', 's2_knn'} & fresh_payload.keys())
                    if name == 'bpc':
                        payload['spec'] = {'data': config['data'], 'method': payload['spec']['method']}
                        torch.save(payload, root / 'policy.pt')
                        path.unlink()
                        restored = load_policy(self.config, root, trajectories=bank)
                        torch.testing.assert_close(restored.direct_actions(actions, states), expected, rtol=0, atol=0)
