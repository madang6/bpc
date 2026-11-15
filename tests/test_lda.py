import copy
import unittest
from tests.suites import integration
import torch
from dpc_nldl.bpc import Policy
from dpc_nldl.lda import fit_discriminant
from dpc_nldl.lda_selector import Selector
from tests.support.policies import data_config, fisher_recipe, trajectories


class LDATest(unittest.TestCase):
    def test_fisher_metric_matches_full_generalized_eigensystem(self):
        for n, width, columns in ((7, 19, 23), (19, 7, 23), (17, 31, 8195)):
            torch.manual_seed(31)
            x = torch.randn(n, width, dtype=torch.float64)
            target = torch.randn(n, columns, dtype=x.dtype).softmax(1)
            fitted = fit_discriminant(x, target, .7, fisher_rank=3)
            z, mass = x-x.mean(0), target.sum(0)
            means = z.T@target/mass
            between = (means*(mass/n))@means.T
            within = z.T@z/n-between
            within.diagonal().add_(.7*within.trace()/width)
            factor = torch.linalg.cholesky(within)
            whitened = torch.linalg.solve_triangular(factor, between, upper=False)
            whitened = torch.linalg.solve_triangular(factor, whitened.T, upper=False).T
            values, vectors = torch.linalg.eigh((whitened+whitened.T)*.5)
            expected = torch.linalg.solve_triangular(factor.T, vectors[:, -3:], upper=True)
            actual = fitted['transform']
            torch.testing.assert_close(actual@actual.T, expected@expected.T, rtol=1e-9, atol=1e-9)
            torch.testing.assert_close(actual.T@within@actual, torch.eye(3, dtype=x.dtype))
            torch.testing.assert_close(between@actual, (within@actual)*values[-3:])
            self.assertLess(fitted['relative_residual'], 1e-10)

    @integration
    def test_fisher_fit_reload_temporal_retrieval_and_exclusions(self):
        for storage in ('device', 'host'):
            with self.subTest(teacher_storage=storage):
                method = fisher_recipe()
                method['g']['lda']['teacher_storage'] = storage
                bank = trajectories(demos=4, length=16)
                p = Policy(bank, data_config(), method, fit_residuals=False)
                stats = p.fit_g_lookup()
                self.assertLess(stats['relative_residual'], 1e-10)
                anchors = torch.tensor([1, 17, 32])
                query = p._past_columns(anchors)
                selected = p.support_selector.select(query)
                self.assertTrue(all(len(columns) > 0 for columns, _ in selected))
                rows = p.hankel_starts[anchors] + p.history
                selected = p.support_selector.select(query, target_rows=rows)
                excluded = p._hankel_exclusion(p.hankel_starts, rows, None)
                for i, (columns, _) in enumerate(selected):
                    self.assertFalse(excluded[i, columns].any())
                allowed = ~excluded
                allowed[:, :5] = False
                for i, (columns, _) in enumerate(p.support_selector.select(query, allowed=allowed)):
                    self.assertTrue(allowed[i, columns].all())
                reloaded = Selector(p, p.support_selector.payload)
                for left, right in zip(selected, reloaded.select(query, target_rows=rows)):
                    torch.testing.assert_close(left[0], right[0])
                trained = Policy(bank, data_config(), method)
                trained.fit()
                restored = Policy(bank, data_config(), method,
                    fitted=trained.fitted_payload({'test': True}))
                rows = torch.tensor([4, 20])
                actions = trained._offline_g(rows)[0]
                torch.testing.assert_close(actions, restored._offline_g(rows)[0])

    @integration
    def test_removed_teachers_support_modes_and_artifacts_are_rejected(self):
        bank = trajectories(demos=4, length=16)
        for key, value in (('teacher', 'entropy'), ('teacher', 'radius'),
                           ('teacher_distance_scale', 'unit'), ('fisher_rank', None),
                           ('adaptive_support', True), ('adaptive_support', False)):
            method = fisher_recipe()
            method['g']['lda'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                Policy(bank, data_config(), method, fit_residuals=False)
        policy = Policy(bank, data_config(), fisher_recipe(), fit_residuals=False)
        policy.fit_g_lookup()
        original = policy.support_selector.payload
        for target, key, value in (('config', 'teacher', 'entropy'),
                                    ('config', 'adaptive_support', False),
                                    ('config', 'retime', .5),
                                    ('diagnostics', 'teacher_kind', 'entropy')):
            payload = copy.deepcopy(original)
            payload[target][key] = value
            with self.subTest(artifact=(target, key)), self.assertRaises(ValueError):
                Selector(policy, payload)
        for key, value in (('prototypes', torch.zeros(1)), ('support_scale', 0)):
            payload = dict(original, **{key: value})
            with self.subTest(artifact=key), self.assertRaises(ValueError):
                Selector(policy, payload)


if __name__ == '__main__':
    unittest.main()
