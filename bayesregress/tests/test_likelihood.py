import unittest
import warnings

import numpy as np

from bayesregress import likelihood
from bayesregress.tests.common import *


TOLS = {"atol": 1e-12, "rtol": 1e-12}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}




class TestLogLikelihood(unittest.TestCase):
    def test_additional_params_is_0(self):
        self.assertEqual(likelihood.LogLikelihood.additional_params, 0)

    def test_default_predictor_is_chebval(self):
        ll = likelihood.LogLikelihood(predictor=None)
        self.assertEqual(ll.predictor, np.polynomial.chebyshev.chebval)

    def test_log_likelihood_not_implemented(self):
        ll = likelihood.LogLikelihood(predictor=None)
        self.assertRaises(NotImplementedError, ll.log_likelihood, 0)

    def test_call_calls_log_likelihood(self):
        ll = MockLL(predictor=None)
        self.assertFalse(ll._log_likelihood_called)
        ll(0)
        self.assertTrue(ll._log_likelihood_called)


class TestGaussianLogLikelihood(unittest.TestCase):
    def test_additional_params_is_1(self):
        self.assertEqual(likelihood.GaussianLogLikelihood.additional_params, 1)

    def test_init_stores_xy(self):
        x, y = generate_correlated_pair_of_data()
        ll = likelihood.GaussianLogLikelihood(x, y)
        self.assertIs(ll.x, x)
        self.assertIs(ll.y, y)

    def test_chebpoly_is_chebpolys_f_x(self):
        ll = make_gaussian_log_likelihood()
        np.random.seed(206)
        x = np.random.rand(100)
        coeffs = np.random.rand(3)
        check = ll.predictor(x, coeffs)
        correct = np.polynomial.chebyshev.chebval(x, coeffs)
        self.assertTrue(np.allclose(check, correct, **TOLS))

    def test_evaluate_conditional_mean_of_y(self):
        ll = make_gaussian_log_likelihood()
        coeffs = np.zeros(2)
        check = ll.evaluate_conditional_mean_of_y(coeffs)
        correct = ll.predictor(ll.x, coeffs)
        self.assertTrue(np.allclose(check, correct, **TOLS))

    def test_evaluate_residuals(self):
        ll = make_gaussian_log_likelihood()
        coeffs = np.zeros(2)
        check = ll.evaluate_residuals(coeffs)
        correct = ll.y - ll.predictor(ll.x, coeffs)
        self.assertTrue(np.allclose(check, correct, **TOLS))

    def test_call_calls_log_likelihood(self):
        ll = make_gaussian_log_likelihood()
        np.random.seed(1703)
        params = np.random.randn(3)
        v0 = ll(params)
        v1 = ll.log_likelihood(params)
        self.assertEqual(v0, v1)

    def test_log_likelihood_uses_log_std_as_first_params(self):
        # We test by making data with no noise, then using the fact that
        # the ll should increase as the log of the noise parameter
        # decreases
        x = np.zeros(10)
        y = 0 * x
        ll = likelihood.GaussianLogLikelihood(x, y)
        log_noise_std = np.linspace(-1, 1, 10)
        for i in range(log_noise_std.size - 1):
            params_low_noise = np.array([log_noise_std[i], 0.0])
            params_high_noise = np.array([log_noise_std[i + 1], 0.0])
            ll_low_noise = ll(params_low_noise)
            ll_high_noise = ll(params_high_noise)
            self.assertGreater(ll_low_noise, ll_high_noise)

    def test_log_likelihood_uses_coeffs_as_last_params(self):
        # We test by making data with no noise, then using the fact that
        # the ll should increase as the params move away from zero
        x = np.zeros(10)
        y = 0 * x
        ll = likelihood.GaussianLogLikelihood(x, y)
        log_noise_std = -1.0
        calc_ll = lambda c: ll(np.array([log_noise_std, c]))
        best_ll = calc_ll(0)

        np.random.seed(1710)
        coeffs = np.random.randn(10)  # positive and negative
        for c in coeffs:
            self.assertLess(calc_ll(c), best_ll)


class TestBinomialLogLikelihood(unittest.TestCase):
    def test_init_stores_x_trials_successes(self):
        np.random.seed(1727)
        ntrials = 100
        x = np.random.randn(ntrials, 3)
        trials = np.random.randint(1, 30, ntrials)
        successes = np.random.randint(0, trials, ntrials)
        ll = likelihood.BinomialLogLikelihood(x, trials, successes)
        self.assertIs(ll.x, x)
        self.assertIs(ll.trials, trials)
        self.assertIs(ll.successes, successes)

    def test_logit_function_is_chebpoly(self):
        np.random.seed(1731)
        ll = make_binomial_log_likelihood()
        params = np.random.randn(4)
        logits = ll.predictor(ll.x, params)
        correct = np.polynomial.chebyshev.chebval(ll.x, params)
        self.assertTrue(np.all(correct == logits))

    def test_calculate_trial_probabilities_returns_on_01(self):
        np.random.seed(1735)
        ll = make_binomial_log_likelihood()
        params = np.random.randn(4)
        probs = ll.calculate_trial_probabilities(params)
        for p in probs:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 1)

    def test_log_likelihood(self):
        n = 70
        tried = np.full(n, 10)
        success = np.full(n, 7)
        x = np.zeros(n)
        calc = likelihood.BinomialLogLikelihood(x, tried, success)

        prob = success.sum() / tried.sum()
        logit = np.log(prob / (1 - prob))
        best_params = np.array([logit])

        total_ll = calc.log_likelihood(best_params)
        per_trial_ll = calc._per_event_log_likelihood(
            tried[0], success[0], prob)
        self.assertAlmostEqual(per_trial_ll * n, total_ll, places=12)

    def test_per_event_log_likelihood(self):
        tried = 10
        success = 7
        best_prob = success / tried
        calc = make_binomial_log_likelihood()
        best_ll = calc._per_event_log_likelihood(tried, success, best_prob)
        for dx in [1e-3, -1e-3]:
            worse_prob = best_prob + dx
            worse_ll = calc._per_event_log_likelihood(
                tried, success, worse_prob)
            self.assertLess(worse_ll, best_ll)

    def test_call_calls_log_likelihood(self):
        np.random.seed(1041)
        ll = make_binomial_log_likelihood()
        params = np.random.randn(4) * 1e-1
        correct = ll.log_likelihood(params)
        check = ll(params)
        self.assertEqual(correct, check)

    def test_probs_clipped_away_from_0_1(self):
        np.random.seed(1041)
        ll = make_binomial_log_likelihood()
        params = np.random.randn(4) * 1e2  # crazy big, so we get 0s and 1s
        probs = ll.calculate_trial_probabilities(params)
        self.assertGreater(probs.min(), 1e-11)
        self.assertLess(probs.max(), 1 - 1e-11)


class TestBernoulliLogLikelihood(unittest.TestCase):
    def test_init_stores_x_successes(self):
        x = np.random.randn(10, 3)
        successes = np.random.randint(low=0, high=2, size=10)
        ll = likelihood.BernoulliLogLikelihood(x, successes)
        self.assertIs(ll.x, x)
        self.assertIs(ll.successes, successes)

    def test_logit_function_is_chebpoly(self):
        np.random.seed(1731)
        ll = make_bernoulli_log_likelihood()
        params = np.random.randn(4)
        logits = ll.predictor(ll.x, params)
        correct = np.polynomial.chebyshev.chebval(ll.x, params)
        self.assertTrue(np.all(correct == logits))

    def test_calculate_trial_probabilities_returns_on_01_clipped(self):
        np.random.seed(1735)
        ll = make_bernoulli_log_likelihood()
        params = np.random.randn(4) * 10
        probs = ll.calculate_trial_probabilities(params)
        for p in probs:
            self.assertGreaterEqual(p, 1e-10)
            self.assertLessEqual(p, 1 - 1e-10)

    def test_log_likelihood(self):
        # we test with 10 trials, 5 successes, p = 0.5, that it
        # gives the correct answer
        successes = np.array([True] * 5 + [False] * 5)
        x = np.zeros(10)
        params = np.zeros(1)
        ll = likelihood.BernoulliLogLikelihood(x, successes)

        probs = ll.calculate_trial_probabilities(params)
        assert np.allclose(probs, 0.5, **TOLS)

        out = ll.log_likelihood(params)
        correct = x.size * np.log(0.5)
        self.assertAlmostEqual(out, correct, places=13)

    def test_call_calls_ll(self):
        np.random.seed(137)
        ll = make_bernoulli_log_likelihood()
        params = np.random.randn(3)
        self.assertEqual(ll(params), ll.log_likelihood(params))


if __name__ == '__main__':
    unittest.main()
