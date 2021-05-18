import unittest

import numpy as np

from bayesregress.prior import GaussianLogPrior
from bayesregress.posterior import LogPosterior
from bayesregress.tests.common import make_gaussian_log_likelihood


class TestLogPosterior(unittest.TestCase):
    def test_log_posterior(self):
        np.random.seed(218)
        params = np.random.randn(4)
        log_prior = GaussianLogPrior()
        log_likelihood = make_gaussian_log_likelihood()
        log_posterior = LogPosterior(log_prior, log_likelihood)

        tocheck = log_posterior(params)
        correct = (log_prior(params) +
                   log_likelihood(params))
        self.assertAlmostEqual(tocheck, correct, places=12)


if __name__ == '__main__':
    unittest.main()
