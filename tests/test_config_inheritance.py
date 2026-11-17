"""Config overlays resolve once and preserve recorded recipes and fit isolation."""
import importlib.util
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from tests.suites import integration
from unittest.mock import patch

from bpc.io import read_config, write_json
from bpc.analyze import analyze
from bpc.pipeline import load_config, task_configs, fit_groups, _validate_methods


ROOT = Path(__file__).resolve().parents[1]


@integration
class ConfigInheritanceTest(unittest.TestCase):
    def test_relative_recursive_objects_lists_and_frozen_resume(self):
        spec=importlib.util.spec_from_file_location('runner_overlay',ROOT/'scripts/run_experiment.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'recipes').mkdir()
            write_json(root/'base.json',{'method':{'seed':7,'rates':[1,2]},'runtime':{'readonly_data_root':'/data'}})
            write_json(root/'recipes/child.json',{'extends':'../base.json','method':{'rates':[1],'ridge':.2}})
            child=root/'recipes/child.json'
            expected={'method':{'seed':7,'rates':[1],'ridge':.2},'runtime':{'readonly_data_root':'/data'}}
            self.assertEqual(runner._run_config(SimpleNamespace(run=None,config=str(child))),expected)
            self.assertEqual(load_config(child),expected)
            self.assertEqual(read_config(root/'base.json')['method']['rates'],[1,2])
            with patch.dict('os.environ',{'BPC_NATIVE':'1','BPC_DATA_ROOT':'/new'}):
                self.assertEqual(load_config(child)['runtime']['readonly_data_root'],'/new')
            run=root/'run';run.mkdir();write_json(run/'config.json',expected)
            (root/'base.json').unlink();child.unlink()
            self.assertEqual(runner._run_config(SimpleNamespace(run=str(run),config=None)),expected)

    def test_invalid_bases_and_cycles_fail_early(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/'config.json'
            for value in ({'extends':'config.json'},{'extends':[]},{'extends':''},[]):
                write_json(p,value)
                with self.assertRaises(ValueError):read_config(p)
            write_json(p,{'extends':'missing.json'})
            with self.assertRaises(FileNotFoundError):read_config(p)

    def test_nested_task_overrides_preserve_settings_and_isolate_mutation(self):
        config = load_config(ROOT/'configs/robosuite_fisher_sparsemax_local.json')
        benchmark = config['benchmarks'][0]
        benchmark['data'] = {'r3m': {'batch_size': 16}}
        task = benchmark['tasks'][0]
        task.update(method={'g': {'supports': 32}},
                    data={'r3m': {'device': 'cpu'}},
                    evaluation={'action_clip': [-.5, .5]},
                    methods=deepcopy(config['methods']), method_base='bpc_nnc')
        original = deepcopy(config)
        jobs = task_configs(config, '.')
        scoped, sibling = jobs[0][0], jobs[1][0]
        expected = deepcopy(config['method']['g'])
        expected['supports'] = 32
        self.assertEqual(scoped['method']['g'], expected)
        self.assertEqual(scoped['data']['r3m'], {
            **config['data']['r3m'], 'batch_size': 16, 'device': 'cpu'})
        self.assertEqual(scoped['evaluation']['action_clip'], [-.5, .5])
        self.assertEqual(scoped['methods']['bpc_fisher']['base'], 'bpc_nnc')
        fit_groups(scoped)
        scoped['method']['g']['lda']['fisher_rank'] = 1
        scoped['evaluation']['action_clip'].append(1)
        scoped['methods']['bpc_fisher']['retrieval_precision'] = 'fp64'
        self.assertEqual(config, original)
        self.assertEqual(sibling['method']['g'], config['method']['g'])
        self.assertEqual(sibling['evaluation']['action_clip'], config['evaluation']['action_clip'])

    def test_tabletop_fisher_ablations_share_full_history_protocol(self):
        baseline = task_configs(load_config(ROOT/'configs/robosuite_fisher_sparsemax_local.json'), '.')
        for name, uniform, residual in (('nhc', True, True), ('nnc', False, False)):
            jobs = task_configs(load_config(ROOT/f'configs/robosuite_{name}_local.json'), '.')
            self.assertEqual(len(jobs), len(baseline))
            for (config, _, label), (reference, _, reference_label) in zip(jobs, baseline):
                with self.subTest(method=name, task=label):
                    self.assertEqual(label, reference_label)
                    group, expected = fit_groups(config)[0], fit_groups(reference)[0]
                    expected_method = deepcopy(expected['method'])
                    if uniform:
                        expected_method['g']['weights'] = 'uniform'
                    self.assertEqual(group['method'], expected_method)
                    self.assertTrue(group['method']['g']['lda']['fit_all_histories'])
                    self.assertEqual(group['spec']['fit_residuals'], residual)
                    self.assertEqual(config['evaluation'], reference['evaluation'])
                    self.assertEqual(config['evaluation']['episode_seed'], 5140000)

    def test_unknown_variant_options_rejected_at_fit_boundary(self):
        config = load_config(ROOT/'configs/robosuite_fisher_sparsemax_local.json')
        config['methods']['bpc_fisher']['unsupported_option'] = True
        with self.assertRaisesRegex(ValueError, 'unknown method fields'):
            fit_groups(config)

    def test_every_experiment_config_can_report_completed_episodes(self):
        for path in sorted((ROOT/'configs').glob('*.json')):
            if path.stem in {'bpc_common', 'robosuite_common', 'dexterous_common', 'pointnet_fit'}:
                continue
            with self.subTest(config=path.name), TemporaryDirectory() as folder:
                config, run, _ = task_configs(load_config(path), folder)[0]
                config['evaluation'].update(init_index_start=0, init_index_stop=3)
                for method in config['methods']:
                    for index in range(3):
                        write_json(run/'episodes'/method/f'{index:03d}.json',
                            dict(method=method, init_index=index, success=index > 0,
                                 steps_to_success=10 if index else None, seconds=1.0))
                summary = analyze(config, run)
                for score in summary['methods'].values():
                    self.assertEqual(score['success_rate'], 2/3)



if __name__=='__main__':unittest.main()
