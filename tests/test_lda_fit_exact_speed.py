"""Independent FP64 reference for blocked Fisher covariance accumulation."""
import unittest
import torch
from bpc.lda import fit_discriminant


class ExactFitSpeedTest(unittest.TestCase):
    def test_blocked_fisher_fit_matches_ambient_metric_with_unnormalized_labels(self):
        generator = torch.Generator().manual_seed(819)
        for rows, width in ((23, 41), (41, 23)):
            with self.subTest(rows=rows, width=width):
                x = torch.randn(rows, width, generator=generator, dtype=torch.float64)
                x[:, -1] = 2*x[:, 0]  # Include a dependent ambient direction.
                target = torch.randn(rows, 8207, generator=generator,
                                     dtype=torch.float64).exp_()
                target *= torch.logspace(-3, 3, rows, dtype=torch.float64)[:, None]
                fitted = fit_discriminant(x, target, shrinkage=.7, fisher_rank=5)
                normalized = target/target.sum(1, keepdim=True)
                z = x-x.mean(0)
                mass = normalized.sum(0)
                means = z.T@normalized/mass
                between = (means*(mass/rows).sqrt()) @ (means*(mass/rows).sqrt()).T
                within = z.T@z/rows-between
                within = (within+within.T)*.5
                ridge = .7*within.trace()/width
                within.diagonal().add_(ridge)
                factor = torch.linalg.cholesky(within)
                whitened = torch.linalg.solve_triangular(factor, between, upper=False)
                whitened = torch.linalg.solve_triangular(factor, whitened.T, upper=False).T
                values, vectors = torch.linalg.eigh((whitened+whitened.T)*.5)
                expected = torch.linalg.solve_triangular(factor.T, vectors[:, -5:], upper=True)
                transform = fitted['transform']
                torch.testing.assert_close(transform@transform.T, expected@expected.T,
                                           rtol=2e-9, atol=2e-9)
                torch.testing.assert_close(torch.tensor(fitted['eigenvalues'], dtype=torch.float64),
                                           values[-5:], rtol=2e-10, atol=1e-12)
                self.assertAlmostEqual(fitted['ridge'], float(ridge), places=12)
                self.assertLess(fitted['relative_residual'], 1e-10)
