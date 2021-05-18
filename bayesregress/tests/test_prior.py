import unittest

import numpy as np

from bayesregress.prior import GaussianLogPrior, UniformDistLogitLogPrior


class TestGaussianLogPrior(unittest.TestCase):
    def test_log_prior_scalar(self):
        prior = GaussianLogPrior()
        p = prior(np.ones(5))
        self.assertEqual(np.size(p), 1)

    def test_log_prior_decreases_as_paramvalues_increase(self):
        scales = np.arange(10)
        np.random.seed(202)
        params = np.random.randn(3)
        log_prior = GaussianLogPrior()
        log_priors = [log_prior(s * params) for s in scales]
        self.assertTrue(np.all(np.diff(log_priors) < 0))

    def test_log_prior_quadratic(self):
        log_prior = GaussianLogPrior()
        prior0 = log_prior(np.zeros(1))
        prior3 = log_prior(np.zeros(1) + 3)
        self.assertEqual(prior0 - prior3, 0.5 * 3**2)

    def test_log_prior_includes_gaussian_normalization(self):
        n_terms = np.arange(1, 10, dtype='int')
        for std in [1.0, 1.2]:
            log_prior = GaussianLogPrior(std=std)
            priors = [log_prior(np.zeros(n)) for n in n_terms]
            log_normalization = np.log(1.0 / np.sqrt(2 * np.pi * std**2))
            for dp in np.diff(priors):
                with self.subTest(std=std):
                    self.assertAlmostEqual(dp, log_normalization, places=10)


class TestUniformDistLogitLogPrior(unittest.TestCase):
    def test_log_prior_scalar(self):
        prior = UniformDistLogitLogPrior()
        p = prior(np.ones(5))
        self.assertEqual(np.size(p), 1)

    def test_log_prior_decreases_as_paramvalues_increase(self):
        np.random.seed(1715)
        scales = np.arange(10)
        params = np.random.randn(3)
        log_prior = UniformDistLogitLogPrior()
        log_priors = [log_prior(s * params) for s in scales]
        self.assertTrue(np.all(np.diff(log_priors) < 0))

    def test_dist_is_normalized(self):
        t = np.linspace(-3e2, 3e2, int(1e3))
        log_prior = UniformDistLogitLogPrior()
        rho = np.exp([log_prior(ti) for ti in t])

        normalization = np.trapz(rho, t)
        self.assertAlmostEqual(normalization, 1.0, places=10)

    def test_dist_mean_prob_from_logit_is_correct(self):
        # <p> = 0.5, where p = np.exp(z) / (1 + np.exp(z))
        t = np.linspace(-3e2, 3e2, int(1e3))
        log_prior = UniformDistLogitLogPrior()
        rho = np.exp([log_prior(ti) for ti in t])
        p = 1.0 / (1 + np.exp(t))

        mean = np.trapz(p * rho, t)
        self.assertAlmostEqual(mean, 0.5, places=10)

    def test_dist_var_prob_from_logit_is_correct(self):
        # really test <p^2> = 1/3, where p = np.exp(z) / (1 + np.exp(z))
        t = np.linspace(-3e2, 3e2, int(1e3))
        log_prior = UniformDistLogitLogPrior()
        rho = np.exp([log_prior(ti) for ti in t])
        p = 1.0 / (1 + np.exp(t))

        var = np.trapz(p**2 * rho, t)
        self.assertAlmostEqual(var, 1.0 / 3, places=10)


if __name__ == '__main__':
    unittest.main()
