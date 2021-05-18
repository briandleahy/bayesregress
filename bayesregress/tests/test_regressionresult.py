import unittest
import warnings
import itertools

import numpy as np

from bayesregress import regress
from bayesregress import regressionresult
from bayesregress.prior import GaussianLogPrior


TOLS = {'atol': 1e-13, 'rtol': 1e-13}


class TestRegressionResult(unittest.TestCase):
    def test_stores_init_args(self):
        np.random.seed(1243)
        x_offset_scale = np.random.rand(3, 2)
        x_names = ['x_{}'.format(i) for i in range(x_offset_scale.shape[0])]
        y_name = 'y'
        orders_and_results = {
            (k,): {'log_evidence': np.random.randn()}
            for k in range(10)}
        predictor = np.cos

        rr = regressionresult.RegressionResult(
            x_offset_scale,
            x_names,
            y_name,
            orders_and_results,
            predictor)

        self.assertEqual(rr.x_names, x_names)
        self.assertEqual(rr.y_name, y_name)
        self.assertEqual(rr.orders_and_results, orders_and_results)
        self.assertEqual(rr.predictor, predictor)

        self.assertTrue(np.all(rr.x_offset_scale == x_offset_scale))

    def test_find_map_model_order(self):
        orders = [
            (1, 2, 3),
            (4, 5, 6),
            (1, 3, 2),
            (0, 0, 0),
            ]
        best_order = orders[2]
        orders_and_results = {k: {"log_evidence": 0} for k in orders}
        orders_and_results[best_order]["log_evidence"] = 1

        rr = regressionresult.RegressionResult(
            orders_and_results=orders_and_results)

        found = rr._find_map_model_order()
        self.assertEqual(found, best_order)

    def test_repr(self):
        x_names = ['var1', 'var2', 'var3']
        y_name = 'dependent-var'
        rr = regressionresult.RegressionResult(
            x_names=x_names,
            y_name=y_name,
            orders_and_results=make_orders_and_results())

        should_be_in = x_names + [y_name] + ['RegressionResult']
        the_repr = repr(rr)
        for word in should_be_in:
            self.assertIn(word, the_repr)

    def test_unnormalize_prediction_raises_error(self):
        rr = make_minimal_regressionresult()
        y = np.random.randn(10)
        self.assertRaises(NotImplementedError, rr._unnormalize_prediction, y)

    def test_unnormalize_error_raises_error(self):
        rr = make_minimal_regressionresult()
        y = np.random.randn(10)
        self.assertRaises(NotImplementedError, rr._unnormalize_error, y)

    def test_errors_predictor_scaled(self):
        # We check by taking a constant model (0th order polynomial),
        # giving it a variance v,
        # and checking that the returned error is sqrt(v).
        predictor = regress.NoninteractingMultivariatePredictor

        np.random.seed(1307)
        var = np.exp(np.random.randn())
        orders_and_results = make_orders_and_results()
        orders_and_results[(0,)]['posterior_covariance'][:] = var

        rr = regressionresult.RegressionResult(
            predictor=predictor,
            orders_and_results=orders_and_results,
            )

        x = np.random.standard_normal((100, 1))
        errs = rr._errors_predictor_scaled(x, (0,))
        correct = np.sqrt(var)
        for err in errs:
            self.assertAlmostEqual(err, correct, places=10)

    def test_errors_map_scaled_returns_map_errors(self):
        predictor = regress.NoninteractingMultivariatePredictor

        np.random.seed(1311)
        best_order = (4,)
        orders_and_results = make_orders_and_results()
        orders_and_results[best_order]['log_evidence'] = 3e3

        rr = regressionresult.RegressionResult(
            predictor=predictor,
            orders_and_results=orders_and_results,
            )

        x = np.random.standard_normal((100, 1))
        err_map = rr._errors_map_scaled(x)
        err_best = rr._errors_predictor_scaled(x, best_order)
        for e1, e2 in zip(err_map, err_best):
            self.assertEqual(e1, e2)

    def test_errors_all_models_scaled_larger_than_min_err(self):
        rr = make_minimal_regressionresult()
        x = 1e-3 * np.random.standard_normal((100, 1))

        errs_all_models = rr._errors_all_models_scaled(x)
        errs_per_order = np.array(
            [rr._errors_predictor_scaled(x, o) for o in rr.orders_and_results])
        lower_bounds = np.min(errs_per_order, axis=0)

        for err, lower_bound in zip(errs_all_models, lower_bounds):
            self.assertGreaterEqual(err, lower_bound)

    def test_predict_model_scaled(self):
        np.random.seed(1346)
        rr = make_minimal_regressionresult()

        x = np.random.standard_normal((10, 1))
        for order in rr.orders_and_results:
            from_rr = rr._predict_model_scaled(x, order)

            coeffs = rr.orders_and_results[order]['result'].x
            correct = rr.predictor(order)(x, coeffs)
            for y1, y2 in zip(from_rr, correct):
                self.assertEqual(y1, y2)

    def test_normalize_x(self):
        scale = 13.50
        offset = 50.13

        x_offset_scale = np.reshape([offset, scale], (1, 2))
        rr = make_minimal_regressionresult(x_offset_scale=x_offset_scale)

        correct = np.random.standard_normal((100, 1))
        descaled = correct * scale + offset
        scaled = rr._normalize_x(descaled)

        for v1, v2 in zip(correct.ravel(), scaled.ravel()):
            self.assertAlmostEqual(v1, v2, places=11)


# to test in subclasses:
# predict_for_map_model(self, x):
# errors_averaged_over_models(self, x):
# errors_for_map_model(self, x):


class TestGaussianRegressionResult(unittest.TestCase):
    def test_irrelevant_params(self):
        self.assertEqual(
            regressionresult.GaussianRegressionResult.irrelevant_params, 1)

    def test_stores_y_offset_scale(self):
        np.random.seed(1243)

        x_offset_scale = np.random.rand(3, 2)
        y_offset_scale = np.random.rand(2)
        x_names = ['x_{}'.format(i) for i in range(x_offset_scale.shape[0])]
        y_name = 'y'
        orders_and_results = {
            (k,): {'log_evidence': np.random.randn()}
            for k in range(10)}
        predictor = np.cos

        rr = regressionresult.GaussianRegressionResult(
            x_offset_scale=x_offset_scale,
            x_names=x_names,
            y_offset_scale=y_offset_scale,
            y_name=y_name,
            orders_and_results=orders_and_results,
            predictor=predictor)

        self.assertEqual(rr.x_names, x_names)
        self.assertEqual(rr.y_name, y_name)
        self.assertEqual(rr.orders_and_results, orders_and_results)
        self.assertEqual(rr.predictor, predictor)

        self.assertTrue(np.all(rr.x_offset_scale == x_offset_scale))
        self.assertTrue(np.all(rr.y_offset_scale == y_offset_scale))

    def test_unnormalize_prediction(self):
        np.random.seed(1634)
        scale, offset = np.exp(5 * np.random.rand(2))
        y_offset_scale = np.array([offset, scale])
        rr = regressionresult.GaussianRegressionResult(
            orders_and_results=make_orders_and_results(),
            y_offset_scale=y_offset_scale)

        z = np.random.randn(31)
        out = rr._unnormalize_prediction(z)
        correct = z * scale + offset
        for y1, y2 in zip(out, correct):
            self.assertAlmostEqual(y1, y2, places=10)

    def test_unnormalize_error(self):
        np.random.seed(1634)
        scale, offset = np.exp(5 * np.random.rand(2))
        y_offset_scale = np.array([offset, scale])
        rr = regressionresult.GaussianRegressionResult(
            orders_and_results=make_orders_and_results(),
            y_offset_scale=y_offset_scale)

        z = np.random.randn(31)
        out = rr._unnormalize_error(z)
        correct = z * scale
        for y1, y2 in zip(out, correct):
            self.assertAlmostEqual(y1, y2, places=10)

    def test_predict_for_map_model(self):
        np.random.seed(1002)
        npts = 13
        rr = make_univariate_gaussian_regressionresult()
        x = np.random.randn(npts, len(rr.x_names))

        y0 = rr.predict_for_map_model(x)
        y1 = rr.predict_for_model(x, rr._map_model_order)

        for a, b in zip(y0, y1):
            self.assertEqual(a, b)

    def test_predict_for_model_calls_separate_models(self):
        np.random.seed(1002)
        npts = 13
        rr = make_univariate_gaussian_regressionresult()
        x = np.random.randn(npts, len(rr.x_names))

        y0 = rr.predict_for_model(x, (0,))
        y1 = rr.predict_for_model(x, (1,))

        for a, b in zip(y0, y1):
            self.assertNotEqual(a, b)


class TestLogisticRegressionResult(unittest.TestCase):
    def test_unnormalize_prediction_returns_probabilities(self):
        rr = regressionresult.LogisticRegressionResult(
            orders_and_results=make_orders_and_results())

        np.random.seed(1640)
        z = np.random.standard_normal((50,)) * 3
        probs = rr._unnormalize_prediction(z)

        for p in probs:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 1)

    def test_errors_for_map_model_returns_twosided(self):
        rr = regressionresult.LogisticRegressionResult(
            x_offset_scale=np.ones([1, 2]),
            orders_and_results=make_orders_and_results(),
            predictor=regress.NoninteractingMultivariatePredictor,
            )

        np.random.seed(1640)
        x = np.random.standard_normal((20, 1))
        probs = rr.predict_for_map_model(x)
        errs = rr.errors_for_map_model(x)

        lower = probs - errs[0]
        upper = probs + errs[1]

        for ar in [lower, upper]:
            self.assertGreaterEqual(ar.min(), 0)
            self.assertLessEqual(ar.max(), 1)


class TestEvidenceFunctionLogger(unittest.TestCase):
    def test_init_creates_orders_and_results(self):
        f = lambda x: np.cos(x).sum()
        caller = MockEvidenceCaller(f)
        logger = regressionresult.EvidenceFunctionLogger(caller)

        self.assertIsInstance(logger.orders_and_results, dict)

    def test_call_updates_stored_results(self):
        f = lambda x: np.cos(x).sum()
        caller = MockEvidenceCaller(f)
        logger = regressionresult.EvidenceFunctionLogger(caller)

        order = (1, 2, 3)
        evidence = logger(order)
        self.assertIn(order, logger.orders_and_results)
        self.assertEqual(
            logger.orders_and_results[order]['log_evidence'], evidence)

    def test_call_looks_up_if_possible(self):
        f = lambda x: np.cos(x).sum()
        caller = MockEvidenceCaller(f)
        logger = regressionresult.EvidenceFunctionLogger(caller)
        order = (1, 2, 3)

        assert caller.counter == 0
        _ = logger(order)
        assert caller.counter == 1
        _ = logger(order)
        self.assertEqual(caller.counter, 1)
        _ = logger.function(order)
        self.assertEqual(caller.counter, 2)

    def test_catches_convergence_errors_as_warnings(self):
        caller = MockEvidenceCaller(_raise_convergence_error)
        logger = regressionresult.EvidenceFunctionLogger(caller)
        order = (1, 2, 3)
        self.assertWarnsRegex(
            UserWarning,
            "ConvergenceError, order=.[0-9]",
            logger,
            order)

    def test_treats_convergence_errors_as_minus_np_inf(self):
        caller = MockEvidenceCaller(_raise_convergence_error)
        logger = regressionresult.EvidenceFunctionLogger(caller)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            f = logger((1, 2, 3))
        self.assertEqual(f, -np.inf)

    def test_logs_convergence_errors(self):
        caller = MockEvidenceCaller(_raise_convergence_error)
        logger = regressionresult.EvidenceFunctionLogger(caller)
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

        best = regressionresult.maximize_discrete_exhaustively(f, guess)
        self.assertEqual(best, (round(correct),))

    def test_multivariate(self):
        np.random.seed(1023)
        nvars = 4
        correct = np.random.rand(nvars) * 9

        f = lambda order: -np.sum((np.array(order) - correct)**2)
        guess = (3,) * nvars

        best = regressionresult.maximize_discrete_exhaustively(f, guess)
        self.assertEqual(best, tuple(np.round(correct).astype('int')))

    def test_evaluates_all_points(self):
        np.random.seed(1044)
        nvars = 4
        correct = np.random.rand(nvars) * 5
        max_order = 5

        f = lambda order: -np.sum((np.array(order) - correct)**2)
        guess = (3,) * nvars
        logger = regressionresult.EvidenceFunctionLogger(
            MockEvidenceCaller(f))

        _ = regressionresult.maximize_discrete_exhaustively(
            logger, guess, max_order=max_order)

        for o in itertools.product(range(max_order), repeat=nvars):
            self.assertIn(o, logger.orders_and_results)


class TestOptimizeDiscreteRelevant(unittest.TestCase):
    def test_univariate(self):
        np.random.seed(1023)
        correct = np.random.rand() * 9

        f = lambda order: -(order[0] - correct)**2
        guess = (3,)

        best = regressionresult.maximize_discrete_relevant(f, guess)
        self.assertEqual(best, (round(correct),))

    def test_multivariate(self):
        np.random.seed(1023)
        nvars = 4
        correct = np.random.rand(nvars) * 9

        f = lambda order: -np.sum((np.array(order) - correct)**2)
        guess = (3,) * nvars

        best = regressionresult.maximize_discrete_relevant(f, guess)
        self.assertEqual(best, tuple(np.round(correct).astype('int')))

    def test_evaluates_multiple_points(self):
        np.random.seed(1044)
        nvars = 4
        correct = np.random.rand(nvars) * 5

        f = lambda order: -np.sum((np.array(order) - correct)**2)
        guess = (3,) * nvars
        logger = regressionresult.EvidenceFunctionLogger(
            MockEvidenceCaller(f))

        _ = regressionresult.maximize_discrete_relevant(
            logger, guess)

        # We should have evaluated at least 2 * nvars + 1 orders, since
        # in the best case the initial guess is correct, and we need to
        # check +- 1 order around it:
        self.assertGreater(len(logger.orders_and_results), 2 * nvars + 1)


class TestRegressionResultsGetter(unittest.TestCase):
    def test_init_stores_x_names(self):
        np.random.seed(1347)
        npts = 31

        x_names = ('a', 'b', 'c', 'd')
        getter = make_regression_results_getter(x_names=x_names)

        self.assertIsInstance(getter.x_names, (list, tuple))
        self.assertEqual(set(getter.x_names), set(x_names))

    def test_init_stores_y_name(self):
        np.random.seed(1347)
        npts = 31
        y_name = 'the-y-name'
        getter = make_regression_results_getter(y_name=y_name)
        self.assertEqual(y_name, getter.y_name)

    def test_init_sets_x_offset_scale_if_not_set(self):
        getter = make_regression_results_getter()
        self.assertIsInstance(getter.x_offset_scale, dict)

    def test_init_stores_x_offset_scale_if_set(self):
        np.random.seed(1622)
        x = np.random.randn(100) * 10
        y = np.random.randn(x.size)

        x_dict = {'x': x}
        y_dict = {'y': y}
        x_offset_scale = {k: (0, 1) for k in x_dict}

        getter = regressionresult.RegressionResultsGetter(
            x_dict,
            y_dict,
            x_offset_scale=x_offset_scale.copy())
        self.assertEqual(getter.x_offset_scale, x_offset_scale)

    def test_find_x_offset_scale(self):
        np.random.seed(1401)

        npts = int(1e5)  # sqrt(N) = gives almostequal to 1 place
        x_names = {'a', 'b', 'c'}
        scales = {k: np.exp(np.random.randn()) for k in x_names}
        offsets = {k: 5 * np.random.randn() for k in x_names}
        x_dict = {
            k: scales[k] * np.random.randn(npts) + offsets[k]
            for k in x_names}
        getter = make_regression_results_getter(x_dict=x_dict)
        offset_scale = getter._find_x_offset_and_scale()
        for k in getter.x_names:
            self.assertAlmostEqual(offset_scale[k][0], offsets[k], places=1)
            self.assertAlmostEqual(offset_scale[k][1], scales[k], places=1)

    def test_find_y_offset_scale_when_gaussian(self):
        np.random.seed(1401)

        getter = make_regression_results_getter()
        offset, scale = getter._find_y_offset_and_scale()
        y = list(getter.y_dict.values())[0]

        self.assertEqual(offset, y.mean())
        self.assertEqual(scale, y.std())

    def test_find_y_offset_scale_when_bernoulli(self):
        np.random.seed(1401)

        getter = make_regression_results_getter(regression_type='bernoulli')
        self.assertIs(getter._find_y_offset_and_scale(), None)

    def test_normalize_x(self):
        np.random.seed(1410)

        npts = int(1e5)
        x_names = {'a', 'b', 'c'}
        scales = {k: np.exp(np.random.randn()) for k in x_names}
        offsets = {k: 5 * np.random.randn() for k in x_names}
        x_dict = {
            k: scales[k] * np.random.randn(npts) + offsets[k]
            for k in x_names}
        getter = make_regression_results_getter(x_dict=x_dict)

        z = getter._normalize_x()
        self.assertEqual(z.shape, (npts, len(x_names)))

        self.assertTrue(np.allclose(z.mean(axis=0), 0, **TOLS))
        self.assertTrue(np.allclose(z.std(axis=0), 1, **TOLS))

    def test_make_prior_returns_gaussian(self):
        getter = make_regression_results_getter()
        prior = getter._make_prior()
        self.assertIsInstance(prior, GaussianLogPrior)

    def test_normalize_y_when_gaussian(self):
        np.random.seed(1427)

        npts = int(1e2)
        y_dict = {'y': 10 + np.random.randn(npts) * 3}
        getter = make_regression_results_getter(y_dict=y_dict, npts=npts)

        z = getter._normalize_y()

        self.assertEqual(z.shape, (npts,))
        self.assertTrue(np.allclose(z.mean(), 0, **TOLS))
        self.assertTrue(np.allclose(z.std(), 1, **TOLS))

    def test_normalize_y_when_bernoulli(self):
        np.random.seed(1427)

        npts = int(1e2)
        y = np.random.choice([0, 1], size=npts, p=[0.1, 0.9])
        y_dict = {'y': y}
        getter = make_regression_results_getter(
            y_dict=y_dict, npts=npts, regression_type='bernoulli')

        z = getter._normalize_y()

        self.assertEqual(z.shape, y.shape)
        self.assertTrue(np.all(z == y))

    def test_get_orders_and_results_returns_dict_with_correct_elements(self):
        getter = make_regression_results_getter(x_names=('a', 'b'))
        out = getter._get_orders_and_results()

        self.assertIsInstance(out, dict)
        for entry in out.values():
            self.assertIsInstance(entry, dict)
            keys = {'log_evidence', 'result', 'posterior_covariance'}
            for k in keys:
                self.assertIn(k, entry)

    def test_make_regression_result_when_gaussian(self):
        getter = make_regression_results_getter(x_names=('a', 'b'))
        rr = getter.make_regression_result()

        self.assertIsInstance(rr, regressionresult.GaussianRegressionResult)

    def test_make_regression_result_when_bernoulli(self):
        npts = 10
        getter = make_regression_results_getter(
            x_names=('a', 'b'),
            y_dict={'y': np.random.choice([True, False], size=npts)},
            npts=npts,
            regression_type='bernoulli')
        rr = getter.make_regression_result()

        self.assertIsInstance(rr, regressionresult.LogisticRegressionResult)

    def test_strip_convergence_errors_from(self):
        getter = make_regression_results_getter()

        failed_orders = [(k,) for k in range(0, 10, 2)]
        success_orders = [(k,) for k in range(1, 10, 2)]
        failed_evidence = (
            regressionresult.EvidenceFunctionLogger._failed_convergence)

        orders_and_results = dict()
        for k in failed_orders:
            orders_and_results.update({k: {"log_evidence": failed_evidence}})
        for k in success_orders:
            orders_and_results.update({k: {"log_evidence": 10}})

        stripped = getter._strip_convergence_errors_from(orders_and_results)

        self.assertEqual(set(stripped.keys()), set(success_orders))
        for result in stripped.values():
            self.assertNotEqual(result['log_evidence'], failed_evidence)

    def test_result_recapitulates_data(self):
        np.random.seed(1458)
        x = np.random.randn(100)
        y = 2 * x + 0.3
        noise = 1e-4 * np.random.randn(x.size)
        getter = regressionresult.RegressionResultsGetter(
            {'x': x}, {'y': y + noise}, max_order=2)
        rr = getter.make_regression_result()
        prediction = rr.predict_for_map_model(x.reshape(-1, 1))

        for a, b in zip(prediction, y):
            self.assertAlmostEqual(a, b, places=3)

    def test_make_predictor(self):
        nvars = 2
        x_names = tuple('abc'[:nvars])
        getter = make_regression_results_getter(x_names=x_names)

        order = (1,) * nvars
        predictor = getter._make_predictor(order)

        self.assertIsInstance(
            predictor,
            regress.NoninteractingMultivariatePredictor)
        self.assertEqual(predictor.order, order)

    def test_find_prediction_gives_correct_keys(self):
        np.random.seed(1222)
        nvars = 2
        x_names = tuple('abc'[:nvars])
        getter = make_regression_results_getter(x_names=x_names)

        order = (1,) * nvars
        prediction = getter.find_prediction(order)
        keys = {'log_evidence', 'result', 'posterior_covariance'}
        self.assertEqual(set(prediction.keys()), keys)

    def test_find_prediction_gives_correct_answer(self):
        # moderately correlated x-data
        x_dict = {
            'a': np.array([
                0.17491131, 2.71542135, 1.15516316, 0.13247331, -0.60474454,
                0.02829037, 2.29199515, -0.28914556, -1.13247949, -0.71725676,
                0.8041879, 1.0750634, 1.05186973, 0.8297467, 0.96021319,
                -1.56828591, 1.31930697, -0.86527756, 1.55604893,
                -0.83317568]),
            'b': np.array([
                -1.7010468, 0.76340537, -1.13147175, -0.88684698, -0.64565894,
                0.3861254, -0.59320056, 0.06797931, 0.54257044, -0.54346153,
                0.51441891, 0.06184495, -0.11528416, -1.90672781, -0.13664491,
                -0.08195446, 0.32005902, 0.57971185, -0.04987975, 0.07618379])
            }
        # and moderately correlated y
        y_dict = {'y': np.array([
            -3.40253289, 2.149867, 1.13975773, 1.10042304, -2.08643635,
            2.09155433, 0.06771938, -0.5573923, -2.74462204, 0.68713746,
            1.75997636, -1.42007755, 0.28118766, 1.09699657, 0.60650905,
            -3.62795094, -3.38429385, 1.36664075, -0.22670035, 1.81833603])}
        getter = regressionresult.RegressionResultsGetter(
                x_dict=x_dict,
                y_dict=y_dict,
                regression_type='gaussian')
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                              Helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_regression_results_getter(
        x_names=('a', 'b'), y_name='y', npts=31, x_dict=None, y_dict=None,
        regression_type='gaussian'):
    if x_dict is None:
        x_dict = {k: np.random.randn(npts) for k in x_names}
    if y_dict is None:
        y_dict = {y_name: np.random.randn(npts)}
    getter = regressionresult.RegressionResultsGetter(
            x_dict=x_dict,
            y_dict=y_dict,
            regression_type=regression_type)
    return getter


def make_minimal_regressionresult(
        orders_and_results=None,
        predictor=None,
        **kwargs):
    if orders_and_results is None:
        orders_and_results = make_orders_and_results()
    if predictor is None:
        predictor = regress.NoninteractingMultivariatePredictor
    rr = regressionresult.RegressionResult(
        orders_and_results=orders_and_results,
        predictor=predictor,
        **kwargs)
    return rr


def make_univariate_gaussian_regressionresult():
    # uses random numbers
    x_offset_scale = np.exp(5 * np.random.rand(1, 2))
    y_offset_scale = np.exp(5 * np.random.rand(2))
    rr = regressionresult.GaussianRegressionResult(
        x_offset_scale=x_offset_scale,
        x_names=('x',),
        y_name='y',
        orders_and_results=make_orders_and_results(gaussian=True),
        y_offset_scale=y_offset_scale,
        predictor=regress.NoninteractingMultivariatePredictor,
        )
    return rr


def make_orders_and_results(gaussian=False):
    add_params = 2 if gaussian else 1
    out = {
        (o,): {
            'result': MockScipyOptimizeResult(o + add_params),
            'posterior_covariance': np.eye(o + add_params),
            'log_evidence': np.random.randn(),
            }
        for o in range(10)
        }
    return out


class MockScipyOptimizeResult(object):
    def __init__(self, nparams):
        self.x = np.random.randn(nparams)  # this is all we need


class MockEvidenceCaller(object):
    def __init__(self, function):
        self.function = function
        self.counter = 0

    def __call__(self, *args):
        self.counter += 1
        return {'log_evidence': self.function(*args)}


def _raise_convergence_error(x):
    raise regress.ConvergenceError


if __name__ == '__main__':
    unittest.main()

