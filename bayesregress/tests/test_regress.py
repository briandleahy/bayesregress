import unittest
import warnings
import itertools

import numpy as np

from bayesregress import regress, prior, predictors
from bayesregress.likelihood import GaussianLogLikelihood
from bayesregress.tests.common import *
from bayesregress.posterior import LogPosterior


TOLS = {'atol': 1e-13, 'rtol': 1e-13}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}


class TestBayesRegressor(unittest.TestCase):
    def test_init_stores_posterior(self):
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


class TestEvidenceFunctionLogger(unittest.TestCase):
    def test_init_creates_orders_and_results(self):
        f = lambda x: np.cos(x).sum()
        caller = MockEvidenceCaller(f)
        logger = regress.EvidenceFunctionLogger(caller)

        self.assertIsInstance(logger.orders_and_results, dict)

    def test_call_updates_stored_results(self):
        f = lambda x: np.cos(x).sum()
        caller = MockEvidenceCaller(f)
        logger = regress.EvidenceFunctionLogger(caller)

        order = (1, 2, 3)
        evidence = logger(order)
        self.assertIn(order, logger.orders_and_results)
        self.assertEqual(
            logger.orders_and_results[order]['log_evidence'], evidence)

    def test_call_looks_up_if_possible(self):
        f = lambda x: np.cos(x).sum()
        caller = MockEvidenceCaller(f)
        logger = regress.EvidenceFunctionLogger(caller)
        order = (1, 2, 3)

        assert caller.counter == 0
        _ = logger(order)
        assert caller.counter == 1
        _ = logger(order)
        self.assertEqual(caller.counter, 1)
        _ = logger.function(order)
        self.assertEqual(caller.counter, 2)

    def test_catches_convergence_errors_as_warnings(self):
        caller = MockEvidenceCaller(raise_convergence_error)
        logger = regress.EvidenceFunctionLogger(caller)
        order = (1, 2, 3)
        self.assertWarnsRegex(
            UserWarning,
            "ConvergenceError, order=.[0-9]",
            logger,
            order)

    def test_treats_convergence_errors_as_minus_np_inf(self):
        caller = MockEvidenceCaller(raise_convergence_error)
        logger = regress.EvidenceFunctionLogger(caller)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            f = logger((1, 2, 3))
        self.assertEqual(f, -np.inf)

    def test_logs_convergence_errors(self):
        caller = MockEvidenceCaller(raise_convergence_error)
        logger = regress.EvidenceFunctionLogger(caller)
        np.random.seed(1138)
        order = tuple(np.random.randint(low=0, high=10, size=3))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            f = logger(order)
        self.assertIn(order, logger.orders_and_results)


class TestOptimizeDiscreteExhaustively(unittest.TestCase):
    def test_univariate(self):
        np.random.seed(1023)
        correct = np.random.rand() * 9

        f = lambda order: -(order[0] - correct)**2
        guess = (3,)

        best = regress.maximize_discrete_exhaustively(f, guess)
        self.assertEqual(best, (round(correct),))

    def test_multivariate(self):
        np.random.seed(1023)
        nvars = 4
        correct = np.random.rand(nvars) * 9

        f = lambda order: -np.sum((np.array(order) - correct)**2)
        guess = (3,) * nvars

        best = regress.maximize_discrete_exhaustively(f, guess)
        self.assertEqual(best, tuple(np.round(correct).astype('int')))

    def test_evaluates_all_points(self):
        np.random.seed(1044)
        nvars = 4
        correct = np.random.rand(nvars) * 5
        max_order = 5

        f = lambda order: -np.sum((np.array(order) - correct)**2)
        guess = (3,) * nvars
        logger = regress.EvidenceFunctionLogger(
            MockEvidenceCaller(f))

        _ = regress.maximize_discrete_exhaustively(
            logger, guess, max_order=max_order)

        for o in itertools.product(range(max_order), repeat=nvars):
            self.assertIn(o, logger.orders_and_results)


class TestOptimizeDiscreteRelevant(unittest.TestCase):
    def test_univariate(self):
        np.random.seed(1023)
        correct = np.random.rand() * 9

        f = lambda order: -(order[0] - correct)**2
        guess = (3,)

        best = regress.maximize_discrete_relevant(f, guess)
        self.assertEqual(best, (round(correct),))

    def test_multivariate(self):
        np.random.seed(1023)
        nvars = 4
        correct = np.random.rand(nvars) * 9

        f = lambda order: -np.sum((np.array(order) - correct)**2)
        guess = (3,) * nvars

        best = regress.maximize_discrete_relevant(f, guess)
        self.assertEqual(best, tuple(np.round(correct).astype('int')))

    def test_evaluates_multiple_points(self):
        np.random.seed(1044)
        nvars = 4
        correct = np.random.rand(nvars) * 5

        f = lambda order: -np.sum((np.array(order) - correct)**2)
        guess = (3,) * nvars
        logger = regress.EvidenceFunctionLogger(
            MockEvidenceCaller(f))

        _ = regress.maximize_discrete_relevant(
            logger, guess)

        # We should have evaluated at least 2 * nvars + 1 orders, since
        # in the best case the initial guess is correct, and we need to
        # check +- 1 order around it:
        self.assertGreater(len(logger.orders_and_results), 2 * nvars + 1)

    def test_respects_max_order(self):
        nvars = 4
        correct = (8,) * nvars
        f = lambda order: -np.sum((np.array(order) - correct)**2)
        guess = (0,) * nvars

        for max_order in range(6):
            best = regress.maximize_discrete_relevant(
                f, guess, max_order=max_order)
            self.assertEqual(max(best), max_order)


class TestRegressionResultsGetter(unittest.TestCase):
    def test_init_stores_x_offset_scale_if_set_as_list(self):
        np.random.seed(1622)
        x = np.random.randn(100) * 10
        y = np.random.randn(x.size)

        offset = 0
        scale = 1
        x_offset_scale = [(offset, scale)]

        getter = regress.RegressionResultsGetter(
            x, y, x_offset_scale=x_offset_scale)
        self.assertEqual(getter.x_offset_scale.shape[0], len(x_offset_scale))
        for check in getter.x_offset_scale:
            self.assertEqual(check[0], offset)
            self.assertEqual(check[1], scale)

    def test_find_x_offset_scale(self):
        np.random.seed(1401)

        npts = int(1e5)  # sqrt(N) = gives almostequal to 1 place
        nvars = 3
        scales = [np.exp(np.random.randn()) for _ in range(nvars)]
        offsets = [5 * np.random.randn() for _ in range(nvars)]
        x = np.transpose([
            scale * np.random.randn(npts) + offset
            for scale, offset in zip(scales, offsets)])
        getter = make_regression_results_getter(x=x)
        offset_scale = getter._find_x_offset_and_scale()
        for i in range(nvars):
            self.assertAlmostEqual(offset_scale[i][0], offsets[i], places=1)
            self.assertAlmostEqual(offset_scale[i][1], scales[i], places=1)

    def test_find_y_offset_scale_when_gaussian(self):
        np.random.seed(1401)

        getter = make_regression_results_getter()
        offset, scale = getter._find_y_offset_and_scale()
        y = getter.y.copy()

        self.assertEqual(offset, y.mean())
        self.assertEqual(scale, y.std())

    def test_find_y_offset_scale_when_bernoulli(self):
        np.random.seed(1401)

        getter = make_regression_results_getter(regression_type='bernoulli')
        self.assertIs(getter._find_y_offset_and_scale(), None)

    def test_normalize_x(self):
        np.random.seed(1410)

        npts = int(1e5)
        nvars = 3
        scales = [np.exp(np.random.randn()) for _ in range(nvars)]
        offsets = [5 * np.random.randn() for _ in range(nvars)]
        x = np.transpose([
            scale * np.random.randn(npts) + offset
            for scale, offset in zip(scales, offsets)])
        getter = make_regression_results_getter(x=x)

        z = getter._normalize_x()

        self.assertEqual(z.shape, (npts, nvars))
        self.assertTrue(np.allclose(z.mean(axis=0), 0, **TOLS))
        self.assertTrue(np.allclose(z.std(axis=0), 1, **TOLS))

    def test_make_prior_returns_gaussian(self):
        getter = make_regression_results_getter()
        prior = getter._make_prior()
        self.assertIsInstance(prior, GaussianLogPrior)

    def test_normalize_y_when_gaussian(self):
        np.random.seed(1427)

        npts = int(1e2)
        y = 10 + np.random.randn(npts) * 3
        getter = make_regression_results_getter(y=y, npts=npts)

        z = getter._normalize_y()

        self.assertEqual(z.shape, (npts,))
        self.assertTrue(np.allclose(z.mean(), 0, **TOLS))
        self.assertTrue(np.allclose(z.std(), 1, **TOLS))

    def test_normalize_y_when_bernoulli(self):
        np.random.seed(1427)

        npts = int(1e2)
        y = np.random.choice([0, 1], size=npts, p=[0.1, 0.9])
        getter = make_regression_results_getter(
            y=y, npts=npts, regression_type='bernoulli')

        z = getter._normalize_y()

        self.assertEqual(z.shape, y.shape)
        self.assertTrue(np.all(z == y))

    def test_get_orders_and_results_returns_dict_with_correct_elements(self):
        getter = make_regression_results_getter(max_order=1)
        out = getter._get_orders_and_results()

        self.assertIsInstance(out, dict)
        for entry in out.values():
            self.assertIsInstance(entry, dict)
            keys = {'log_evidence', 'result', 'posterior_covariance'}
            for k in keys:
                self.assertIn(k, entry)

    def test_strip_convergence_errors_from(self):
        getter = make_regression_results_getter()

        failed_orders = [(k,) for k in range(0, 10, 2)]
        success_orders = [(k,) for k in range(1, 10, 2)]
        failed_evidence = (
            regress.EvidenceFunctionLogger._failed_convergence)

        orders_and_results = dict()
        for k in failed_orders:
            orders_and_results.update({k: {"log_evidence": failed_evidence}})
        for k in success_orders:
            orders_and_results.update({k: {"log_evidence": 10}})

        stripped = getter._strip_convergence_errors_from(orders_and_results)

        self.assertEqual(set(stripped.keys()), set(success_orders))
        for result in stripped.values():
            self.assertNotEqual(result['log_evidence'], failed_evidence)

    def test_make_predictor(self):
        for nvars in [2, 3, 4, 9]:
            npts = 55
            x = np.ones((npts, nvars))
            getter = make_regression_results_getter(x=x)

            order = (1,) * nvars
            predictor = getter._make_predictor(order)

            self.assertIsInstance(
                predictor,
                predictors.NoninteractingMultivariatePredictor)
            self.assertEqual(predictor.order, order)

    def test_find_prediction_gives_correct_keys(self):
        np.random.seed(1222)
        getter = make_regression_results_getter()

        order = (1,) * getter.x.shape[1]
        prediction = getter.find_prediction(order)
        keys = {'log_evidence', 'result', 'posterior_covariance'}
        self.assertEqual(set(prediction.keys()), keys)

    def test_find_prediction_gives_correct_answer(self):
        # moderately correlated x-data
        x = np.array([
            [ 0.17491131, -1.7010468 ],
            [ 2.71542135,  0.76340537],
            [ 1.15516316, -1.13147175],
            [ 0.13247331, -0.88684698],
            [-0.60474454, -0.64565894],
            [ 0.02829037,  0.3861254 ],
            [ 2.29199515, -0.59320056],
            [-0.28914556,  0.06797931],
            [-1.13247949,  0.54257044],
            [-0.71725676, -0.54346153],
            [ 0.8041879 ,  0.51441891],
            [ 1.0750634 ,  0.06184495],
            [ 1.05186973, -0.11528416],
            [ 0.8297467 , -1.90672781],
            [ 0.96021319, -0.13664491],
            [-1.56828591, -0.08195446],
            [ 1.31930697,  0.32005902],
            [-0.86527756,  0.57971185],
            [ 1.55604893, -0.04987975],
            [-0.83317568,  0.07618379]])
        y = np.array([
            -3.40253289, 2.149867, 1.13975773, 1.10042304, -2.08643635,
            2.09155433, 0.06771938, -0.5573923, -2.74462204, 0.68713746,
            1.75997636, -1.42007755, 0.28118766, 1.09699657, 0.60650905,
            -3.62795094, -3.38429385, 1.36664075, -0.22670035, 1.81833603])
        getter = regress.RegressionResultsGetter(
                x, y, regression_type='gaussian')
        order = (2, 1)
        prediction = getter.find_prediction(order)

        correct_log_evidence = -36.01122741724744
        correct_best_post = 31.708203662711924
        correct_best_x = np.array([
            -0.06515793, 0.07376477, 0.30000747,-0.07700238, 0.18797525])
        correct_post_var = np.array([
            0.02448534, 0.05146509, 0.04318893, 0.01026102, 0.04850524])

        self.assertAlmostEqual(
            prediction['log_evidence'],
            correct_log_evidence, places=6)
        self.assertAlmostEqual(
            prediction['result'].fun,
            correct_best_post, places=6)
        tols = {'atol': 1e-6, 'rtol': 1e-6}
        self.assertTrue(np.allclose(
            prediction['result'].x,
            correct_best_x, **tols))
        self.assertTrue(np.allclose(
            np.diag(prediction['posterior_covariance']),
            correct_post_var, **tols))

    def test_respects_max_order_for_multivariate(self):
        max_order = 3

        nvars = 2
        npts = 3100
        x = np.random.standard_normal((npts, nvars))
        y = np.cos(2*x).prod(axis=1)

        getter = regress.RegressionResultsGetter(
            x, y, max_order=max_order, regression_type='gaussian')
        rr = getter.fit_data()

        for order in rr.orders_and_results:
            self.assertLessEqual(max(order), max_order)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                              Helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def make_regression_results_getter(
        x=None, y=None, npts=31, regression_type='gaussian', max_order=3):
    if x is None:
        x = np.random.standard_normal((npts, 2))
    if y is None:
        y = np.random.standard_normal((x.shape[0],))
    getter = regress.RegressionResultsGetter(
            x, y, regression_type=regression_type, max_order=max_order)
    return getter


class MockEvidenceCaller(object):
    def __init__(self, function):
        self.function = function
        self.counter = 0

    def __call__(self, *args):
        self.counter += 1
        return {'log_evidence': self.function(*args)}


if __name__ == '__main__':
    unittest.main()
