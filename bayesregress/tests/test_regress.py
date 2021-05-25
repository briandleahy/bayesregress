import unittest
import warnings

import numpy as np

from bayesregress import regress, prior, predictors
from bayesregress.likelihood import GaussianLogLikelihood
from bayesregress.tests.common import *
from bayesregress.posterior import LogPosterior


MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}


class TestBayesRegressor(unittest.TestCase):
    def test_init_stores_posterior(self):  # FIXME
        ll = make_gaussian_log_posterior()
        regressor = regress.BayesianRegressor(ll)
        self.assertIs(regressor.log_posterior, ll)

    def test_negative_log_posterior(self):
        np.random.seed(218)
        regressor = make_regressor()
        params = np.random.randn(4)
        tocheck = regressor.negative_log_posterior(params)
        correct = -regressor.log_posterior(params)
        self.assertAlmostEqual(tocheck, correct, places=12)

    def test_cls_minimize_finds_minimum(self):
        np.random.seed(157)
        p0 = np.random.randn(10)
        out = regress.BayesianRegressor.minimize(np.linalg.norm, p0)
        self.assertTrue(np.allclose(out.x, np.zeros(p0.size), **MEDTOLS))

    def test_find_max_a_posteriori_params_for_given_order(self):
        # We just check that, for the found maximum x*, x* + dx and
        # x* -dx both have lower log posteriors.
        # Use 3 params for noise, offset, slope
        nparams = 3
        np.random.seed(17817)
        regressor = make_regressor()
        p0 = np.zeros(nparams)
        pbest = regressor.find_max_a_posteriori_params_for_given_order(p0)
        best_x = pbest.x
        best_posterior = regressor.log_posterior(best_x)
        dx = np.random.randn(nparams) * 1e-2
        for sign in [+1, -1]:
            log_posterior = regressor.log_posterior(best_x + sign * dx)
            self.assertLess(log_posterior, best_posterior)

    def test_calculate_negative_posterior_hessian_gives_correct(self):
        np.random.seed(1444)

        d = 4
        a = np.random.standard_normal((d, d))
        m = a.dot(a.T)
        log_posterior = lambda x: -0.5 * m.dot(x).dot(x)
        regressor = regress.BayesianRegressor(log_posterior)

        hess = regressor._calculate_negative_posterior_hessian(np.zeros(d))
        correct = m

        for a, b in zip(hess.ravel(), correct.ravel()):
            self.assertAlmostEqual(a, b, places=7)

    def _test_calculate_negative_posterior_hessian_raises_ill_conditioned(self):
        d = 2

        def ill_conditioned_function(coeffs):
            # Like a steep booth function, from curve fitting
            dx, dy = coeffs
            x = dx + 3.5
            y = dy + 3.5
            booth_degenerate = (x + 1 * y - 7)**2 + (1 * x + y - 7)**2
            return -1e8 * booth_degenerate

        regressor = regress.BayesianRegressor(ill_conditioned_function)

        self.assertRaises(
            regress.HessianIllConditionedError,
            regressor._calculate_negative_posterior_hessian,
            np.zeros(d))


class TestRegressorFindModelEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # raised by scipy.optimize
        warnings.simplefilter('ignore', RuntimeWarning)

    @classmethod
    def tearDownClass(cls):
        warnings.simplefilter('default', RuntimeWarning)

    def _test_rescales_data_for_hessian_calculation(self):
        # FIXME this is very hard to test, as it is hard to make
        # a test case that fails repeatedly.
        # This test currently fails, but it passes if you change the
        # seed to 1119 instead of 1019, or if you use logistic
        # regression, or if you use a lower order
        np.random.seed(1019)

        # regressor = make_regressor(), but with more variation in
        # x to make the polynomials more unstable.
        npts = 200
        # x = np.random.standard_cauchy((npts,))
        x = np.random.standard_normal((npts,))
        y = 0.5 * x + np.sqrt(0.75) * np.random.randn(npts)
        regressor = regress.BayesianRegressor(
            GaussianLogLikelihood(x, y))

        # y = np.random.choice([True, False], size=npts)
        # regressor = regress.BayesianRegressor(
            # regress.BernoulliLogLikelihood(100 * x, y))

        # regressor = make_regressor()
        # We just care that this runs without throwing a ConvergenceError
        evidence = regressor.find_model_evidence(np.zeros(18))
        self.assertTrue(evidence is not None)
        pass

    def test_linear_model_produces_best_evidence_for_linear_data(self):
        # 3 params is noise, mean, slope
        np.random.seed(227)
        regressor = make_regressor()
        nparams = [2, 3, 4, 5, 6, 7, 8, 9]
        results = [regressor.find_model_evidence(np.zeros(n)) for n in nparams]
        evidences = [r['log_evidence'] for r in results]
        best_evidence = np.argmax(evidences)
        self.assertEqual(nparams[best_evidence], 3)

    def test_quadratic_model_produces_best_evidence_for_quadratic_data(self):
        np.random.seed(1014)
        x = np.random.randn(100)
        noise = np.random.randn(x.size) * 0.5
        a, b, c = np.random.randn(3)
        y = a * x**2 + b * x + c + noise
        y -= y.mean(); y /= y.std()

        likelihood = GaussianLogLikelihood(x, y)
        regressor = regress.BayesianRegressor(likelihood)

        nparams = [2, 3, 4, 5, 6, 7, 8, 9]
        results = [regressor.find_model_evidence(np.zeros(n)) for n in nparams]
        evidences = [r['log_evidence'] for r in results]
        best_evidence = np.argmax(evidences)
        # 4 params is noise, mean, slope, quadratic
        self.assertEqual(nparams[best_evidence], 4)

    def test_plays_with_NoninteractingMultivariatePredictor(self):
        np.random.seed(1057)
        npts = 200
        slope1 = 0.3
        slope2 = 0.2
        x = np.random.randn(npts, 2)
        y = np.random.randn(npts) * 0.1 + slope1 * x[:, 0] + slope2 * x[:, 1]

        order = (3, 2)
        nparams = 2 + sum(order)
        predictor = predictors.NoninteractingMultivariatePredictor(order)
        ll = GaussianLogLikelihood(x, y, predictor)
        regressor = regress.BayesianRegressor(ll)
        evidence = regressor.find_model_evidence(np.zeros(nparams))
        best_params = evidence['result'].x

        self.assertAlmostEqual(best_params[2], slope1, places=2)
        self.assertAlmostEqual(best_params[2 + order[0]], slope2, places=2)

    def test_find_model_evidence_gives_reproducible_answer(self):
        x = np.array([
            3.18545499e-01, -6.79559435e-01, -2.20838847e+00, 5.90240163e-01,
            1.67476604e+00, -1.66595597e+00, -8.24338737e-01, 5.45260342e-01,
            1.18619318e-03, -7.57030522e-01, 6.29545192e-01, -9.56317655e-02,
            -1.50786065e+00, -5.00013015e-01, 1.71401027e+00, -1.22376727e+00,
            1.39590508e+00, 4.14476343e-01, 1.51966677e+00, 1.39006340e+00])
        y = np.array([
            0.80153164, -1.82006924, -2.35668858, 1.54125661, -0.59445441,
            -0.16141933, 0.42192413, 1.0820727, -1.08302787, -1.28133131,
            0.86466443, 0.23203177, -0.40320498, -1.58266396, 0.89672305,
            -0.99160254, 0.4601744, 0.64791449, -0.32455241, 2.42983592])
        ll = GaussianLogLikelihood(x, y)
        posterior = LogPosterior(prior.GaussianLogPrior(), ll)
        regressor = regress.BayesianRegressor(posterior)

        p0 = np.zeros(4)
        prediction = regressor.find_model_evidence(p0)
        log_evidence = prediction['log_evidence']
        best_x = prediction['result'].x
        best_posterior = prediction['result'].fun
        x_var = np.diag(prediction['posterior_covariance'])

        correct_log_evidence = -34.76022288347065
        correct_best_x = np.array(
            [-0.07370544,  0.03699663,  0.58463388, -0.07269822])
        correct_best_posterior = 30.720504872888203
        correct_x_var = np.array(
            [0.02450403, 0.05749395, 0.03187759, 0.00624688])

        self.assertAlmostEqual(log_evidence, correct_log_evidence, places=6)
        self.assertAlmostEqual(best_posterior, correct_best_posterior, places=6)
        for a, b in zip(x_var, correct_x_var):
            self.assertAlmostEqual(a, b, places=6)
        for a, b in zip(best_x, correct_best_x):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == '__main__':
    unittest.main()
