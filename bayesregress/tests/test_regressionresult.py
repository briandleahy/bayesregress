import unittest
import warnings

import numpy as np

from bayesregress import regress
from bayesregress import regressionresult
from bayesregress.prior import GaussianLogPrior
from bayesregress.predictors import NoninteractingMultivariatePredictor
from bayesregress.tests.common import raise_convergence_error


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

    def test_repr_uses_names_when_provided(self):
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

    def test_repr_omits_names_when_not_provided(self):
        for n in range(1, 4):
            x_offset_scale = [(0, 1) for _ in range(n)]
            rr = regressionresult.RegressionResult(
                x_offset_scale=x_offset_scale,
                orders_and_results=make_orders_and_results())

            should_be_in = ['1 variable', f'{n} variable', 'RegressionResult']
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
        predictor = NoninteractingMultivariatePredictor

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
        predictor = NoninteractingMultivariatePredictor

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

    def test_predict_model_scaled_calls_correct_predictor(self):
        np.random.seed(1346)
        rr = make_minimal_regressionresult()

        x = np.random.standard_normal((10, 1))
        for order in rr.orders_and_results:
            from_rr = rr._predict_model_scaled(x, order)

            coeffs = rr.orders_and_results[order]['result'].x
            correct = rr.predictor(order)(x, coeffs)
            for y1, y2 in zip(from_rr, correct):
                self.assertEqual(y1, y2)

    def test_normalize_x_correct_scaling(self):
        scale = 13.50
        offset = 50.13

        x_offset_scale = np.reshape([offset, scale], (1, 2))
        rr = make_minimal_regressionresult(x_offset_scale=x_offset_scale)

        correct = np.random.standard_normal((100, 1))
        descaled = correct * scale + offset
        scaled = rr._normalize_x(descaled)

        for v1, v2 in zip(correct.ravel(), scaled.ravel()):
            self.assertAlmostEqual(v1, v2, places=11)

    def test_normalize_x_casts_1d_to_nd_like(self):
        x_offset_scale = np.reshape([0, 10], (1, 2))
        rr = make_minimal_regressionresult(x_offset_scale=x_offset_scale)
        npts = 161

        x = np.ones(npts)
        z = rr._normalize_x(x)

        self.assertEqual(z.shape, (npts, 1))

    def test_predict_raises_error_if_x_wrong_shape(self):
        n_variables = 4
        n_data_points = 100

        x_offset_scale = [(0, 1) for _ in range(n_variables)]
        add_params = 2
        orders_and_results = {
            (0,) * n_variables: {
                'result': MockScipyOptimizeResult(2),
                'posterior_covariance': np.eye(2),
                'log_evidence': np.random.randn(),
                }
            }
        rr = regressionresult.GaussianRegressionResult(
            x_offset_scale=x_offset_scale,
            y_offset_scale=(0, 1),
            orders_and_results=orders_and_results,
            predictor=NoninteractingMultivariatePredictor,
            )
        wrong_shape = (n_data_points, n_variables - 1)
        self.assertRaisesRegex(
            ValueError,
            'shape',
            rr.predict_for_model,
            np.random.standard_normal(wrong_shape),
            (0,) * n_variables)

    def test_model_probabilities_returns_keys_of_orders(self):
        orders_and_results = make_orders_and_results(nparams=22)
        rr = regressionresult.GaussianRegressionResult(
            orders_and_results=orders_and_results)

        probs = rr.model_probabilities
        self.assertEqual(set(probs), set(orders_and_results))

    def test_model_probabilities_returns_valid_probabilities(self):
        orders_and_results = make_orders_and_results(nparams=22)
        rr = regressionresult.GaussianRegressionResult(
            orders_and_results=orders_and_results)

        probs = rr.model_probabilities
        for p in probs.values():
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 1)

    def test_model_probabilities_keeps_model_ranking(self):
        orders_and_results = make_orders_and_results(nparams=22)
        rr = regressionresult.GaussianRegressionResult(
            orders_and_results=orders_and_results)

        probs = rr.model_probabilities

        correct = sorted(
            orders_and_results,
            key=lambda k: orders_and_results[k]['log_evidence'])
        check = sorted(probs, key=lambda k: probs[k])

        self.assertEqual(check, correct)


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
            predictor=NoninteractingMultivariatePredictor,
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                              Helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_minimal_regressionresult(
        orders_and_results=None,
        predictor=None,
        **kwargs):
    if orders_and_results is None:
        orders_and_results = make_orders_and_results()
    if predictor is None:
        predictor = NoninteractingMultivariatePredictor
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
        predictor=NoninteractingMultivariatePredictor,
        )
    return rr


def make_orders_and_results(gaussian=False, nparams=10):
    add_params = 2 if gaussian else 1
    out = {
        (o,): {
            'result': MockScipyOptimizeResult(o + add_params),
            'posterior_covariance': np.eye(o + add_params),
            'log_evidence': np.random.randn(),
            }
        for o in range(nparams)
        }
    return out


class MockScipyOptimizeResult(object):
    def __init__(self, nparams):
        self.x = np.random.randn(nparams)  # this is all we need


if __name__ == '__main__':
    unittest.main()

