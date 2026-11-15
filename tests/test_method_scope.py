"""The runner must reject retired methods before it starts fitting or evaluation."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from tests.suites import integration
from unittest.mock import patch

from dpc_nldl.data import benchmark_task, encoder_spec, make_episode
from dpc_nldl.bpc import Policy
from dpc_nldl.pipeline import _validate_methods, fit_groups, load_config, run, task_configs
from scripts import run_experiment as runner


PROJECT = Path(__file__).resolve().parents[1]


class MethodScopeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(PROJECT / 'configs/robosuite_fisher_sparsemax_local.json')

    def test_manuscript_variants_preserve_their_distinct_fit_contracts(self):
        self.config['method']['residual']['target_retrieval_precision'] = 'fp64'
        self.config['methods'] = {
            'fisher': {'base': 'gip_direct', 'g_lookup': 'lda'},
            'ridge': {'base': 'gip_direct', 'g_lookup': 'action_ridge'},
            'nhc': {'base': 'gip_direct', 'g_lookup': 'lda', 'g_weights': 'uniform'},
            'nnc': {'base': 'deepc_g_direct', 'g_lookup': 'lda'},
            'nlr': {'base': 'gip_direct', 'g_lookup': 'past'},
        }
        self.config['analysis']['reference_method'] = 'fisher'
        _validate_methods(self.config, None)
        groups = {group['name']: group for group in fit_groups(self.config)}
        self.assertEqual(set(groups), {'lda', 'action_ridge', 'lda_uniform', 'default'})
        self.assertEqual(groups['lda_uniform']['method']['g']['weights'], 'uniform')
        self.assertNotIn('lookup', groups['default']['method']['g'])
        self.config['methods'] = {'nnc': self.config['methods']['nnc']}
        self.assertFalse(fit_groups(self.config)[0]['spec']['fit_residuals'])

    def test_retired_method_names_and_selectors_cannot_form_fit_groups(self):
        for base in ('gip_spc', 'deepc_g_spc', 'selected_knn_rff_direct',
                     'selected_knn_rff_spc', 'active_fusion', 'first_order_fusion', 'pi05_base'):
            with self.subTest(base=base):
                config = deepcopy(self.config)
                config['methods']['gip_lda']['base'] = base
                with self.assertRaisesRegex(ValueError, 'unknown configured method'):
                    fit_groups(config)
                with self.assertRaisesRegex(ValueError, 'unknown method'):
                    Policy.__new__(Policy).select_method(base)
        for lookup in ('action_neighborhood', 'sparsemax_qp'):
            with self.subTest(lookup=lookup):
                config = deepcopy(self.config)
                config['methods']['gip_lda']['g_lookup'] = lookup
                with self.assertRaisesRegex(ValueError, 'unknown G lookup'):
                    fit_groups(config)

    def test_retired_benchmarks_and_encoders_are_rejected_at_entrypoints(self):
        scoped = task_configs(self.config, '.')[0][0]
        for kind in ('libero', 'robocasa'):
            with self.subTest(benchmark=kind):
                config = deepcopy(self.config)
                config['benchmarks'][0]['kind'] = kind
                with self.assertRaisesRegex(ValueError, 'unknown benchmark'):
                    task_configs(config, '.')
                direct = deepcopy(scoped)
                direct['evaluation']['benchmark'] = kind
                for call in (lambda: task_configs(direct, '.'),
                             lambda: benchmark_task(direct), lambda: make_episode(direct)):
                    with self.assertRaisesRegex(ValueError, 'benchmark'):
                        call()
        for name in ('sdp', 'pi05'):
            with self.subTest(encoder=name):
                config = deepcopy(self.config)
                config['data'].update(encoder=name, feature_keys=[])
                with self.assertRaisesRegex(ValueError, 'BPC encoder must be r3m or dp3_pointnet'):
                    _validate_methods(config, None)
                with self.assertRaisesRegex(ValueError, 'visual encoder'):
                    encoder_spec(config['data'])
        config = deepcopy(self.config)
        config['policy'] = {'name': 'pi05_base'}
        with self.assertRaisesRegex(ValueError, 'unknown policy'):
            _validate_methods(config, None)

    @integration
    def test_launcher_rejects_retired_options_before_spawning_workers(self):
        for extra in ('policy', 'encoder', 'benchmark', 'task_encoder'):
            config = deepcopy(self.config)
            if extra == 'policy':
                config['policy'] = {'name': 'pi05_base'}
            elif extra == 'encoder':
                config['data']['encoder'] = 'sdp'
            elif extra == 'benchmark':
                config['benchmarks'][0]['kind'] = 'robocasa'
            else:
                config['benchmarks'][0]['tasks'][0]['data'] = {'encoder': 'pi05'}
            with self.subTest(extra=extra), TemporaryDirectory() as folder:
                args = SimpleNamespace(run=folder, stage='fit', shard=None, benchmark=None,
                                       workers='remote=0', detach=True)
                with (patch.object(runner, '_run_config', return_value=config),
                      patch.object(runner.subprocess, 'Popen') as spawn,
                      patch.object(runner.subprocess, 'run') as stage):
                    with self.assertRaises(ValueError):
                        runner.distribute(args)
                    spawn.assert_not_called()
                    stage.assert_not_called()

    def test_retired_algorithm_options_are_rejected(self):
        options = (
            ('g', 'svd_rcond', .03), ('g', 'future_consistency', 1),
            ('g', 'constraint', 'nonnegative'), ('residual', 'model', 'mlp'),
            ('residual', 'lookup', 'temporal'), ('lda', 'teacher', 'entropy'),
            ('lda', 'teacher', 'radius'), ('lda', 'teacher_distance_scale', 'unit'),
            ('lda', 'fisher_rank', None), ('lda', 'adaptive_support', None),
            ('lda', 'retime', True), ('lda', 'retime', .5),
        )
        for section, field, value in options:
            with self.subTest(section=section, field=field, value=value):
                config = deepcopy(self.config)
                target = config['method']['g']['lda'] if section == 'lda' else config['method'][section]
                target[field] = value
                with self.assertRaises(ValueError):
                    _validate_methods(config, None)
                with self.assertRaises(ValueError):
                    fit_groups(config)
                direct = deepcopy(config['method'])
                direct['g']['lookup'] = 'lda'
                with self.assertRaises(ValueError), patch.object(
                        Policy, '_build_bank', side_effect=AssertionError('validation came too late')):
                    Policy([], {}, direct)

    @integration
    def test_task_override_cannot_reenable_a_removed_method(self):
        self.config['benchmarks'][0]['tasks'] = [{
            'name': 'square_d0', 'method_base': 'active_fusion'}]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps(self.config))
            with patch('dpc_nldl.data.prepare') as prepare:
                with self.assertRaisesRegex(ValueError, 'unknown configured method'):
                    run(run_dir=root, stage='data')
                prepare.assert_not_called()

    def test_task_override_cannot_reenable_a_removed_teacher(self):
        method = deepcopy(self.config['method'])
        method['g']['lda']['teacher'] = 'entropy'
        self.config['benchmarks'][0]['tasks'] = [{'name': 'square_d0', 'method': method}]
        scoped, _, _ = task_configs(self.config, '/unused')[0]
        with self.assertRaisesRegex(ValueError, 'teacher must be sparsemax'):
            _validate_methods(scoped, None)


if __name__ == '__main__':
    unittest.main()
