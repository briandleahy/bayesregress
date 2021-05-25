import numpy as np

from bayesregress import likelihood, regress
from bayesregress.prior import GaussianLogPrior
from bayesregress.posterior import LogPosterior


class MockLL(likelihood.LogLikelihood):
    def __init__(self, *args, **kwargs):
        super(MockLL, self).__init__(*args, **kwargs)
        self._log_likelihood_called = False

    def log_likelihood(self, params):
        self._log_likelihood_called = True


def make_binomial_log_likelihood():
    npts = 100
    x = np.random.randn(npts)
    trials = np.random.randint(1, 30, npts)
    successes = np.random.randint(0, trials, npts)
    return likelihood.BinomialLogLikelihood(x, trials, successes)


def make_bernoulli_log_likelihood():
    npts = 100
    x = np.random.randn(npts)
    successes = np.random.choice([True, False], size=npts)
    return likelihood.BernoulliLogLikelihood(x, successes)


def make_gaussian_log_likelihood():
    x, y = generate_correlated_pair_of_data()
    return likelihood.GaussianLogLikelihood(x, y)


def generate_correlated_pair_of_data(npts=100, correlation=0.5, seed=72):
    # What is a reasonable empirical correlation?
    # <sum_i x_i y_i> / N = r
    # <(sum_i x_i y_i)^2 > / N^2 = r^2 + (1 + 2r - r^2) / N
    # so the empirical correlation is ~N(r, sqrt((1+2r-r^2)/N) )
    np.random.seed(seed)
    x = np.random.randn(npts)
    y = correlation * x + np.sqrt(1 - correlation**2) * np.random.randn(npts)
    return x, y


def make_regressor():
    posterior = make_gaussian_log_posterior()
    return regress.BayesianRegressor(posterior)


def make_gaussian_log_posterior():
    likelihood = make_gaussian_log_likelihood()
    prior = GaussianLogPrior(mean=0, std=1)
    posterior = LogPosterior(prior, likelihood)
    return posterior


def raise_convergence_error(x):
    raise regress.ConvergenceError


