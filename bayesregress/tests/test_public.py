import unittest
import warnings
import itertools

import numpy as np

from bayesregress import public
from bayesregress.tests.common import *


class TestPreprocessInputs(unittest.TestCase):
    def test_reshapes_x(self):
        npts = 100
        x = np.zeros(npts)
        y = np.zeros(npts)

        x_y, *_ = public.preprocess_inputs(x, y)
        x_reshaped = x_y[0]
        self.assertEqual(x_reshaped.shape, (npts, 1))

    def test_autochooses_bernoulli_when_boolean_data(self):
        npts = 100
        x = np.zeros(npts)
        y = np.zeros(npts, dtype='bool')

        xy, kwargs, names = public.preprocess_inputs(x, y)

        self.assertEqual(kwargs['regression_type'], 'bernoulli')

    def test_autochooses_gaussian_when_float_data(self):
        npts = 100
        x = np.zeros(npts)
        y = np.zeros(npts, dtype='float64')

        xy, kwargs, names = public.preprocess_inputs(x, y)

        self.assertEqual(kwargs['regression_type'], 'gaussian')

    def test_raises_error_if_x_weird_datatype(self):
        x1 = np.zeros(100)
        y = 0 * x1
        x = ({'x1': x1},)
        self.assertRaisesRegex(
            ValueError,
            'unrecognized',
            public.preprocess_inputs,
            x, y)


class TestFitData(unittest.TestCase):
    def test_stores_x_names(self):
        np.random.seed(1347)
        npts = 31

        x_names = ('a', 'b', 'c', 'd')
        rr = perform_simple_regression(x_names=x_names)

        self.assertIsInstance(rr.x_names, (list, tuple))
        self.assertEqual(set(rr.x_names), set(x_names))

    def test_stores_y_name(self):
        np.random.seed(1347)
        npts = 31
        y_name = 'the-y-name'
        rr = perform_simple_regression(y_name=y_name)
        self.assertEqual(y_name, rr.y_name)

    def test_sets_x_offset_scale_if_not_set(self):
        rr = perform_simple_regression()
        self.assertIsInstance(rr.x_offset_scale, np.ndarray)
        self.assertEqual(rr.x_offset_scale.ndim, 2)

    def test_default_is_gaussian(self):
        rr = perform_simple_regression(x_names=('a', 'b'))
        self.assertIsInstance(rr, regress.GaussianRegressionResult)

    def test_logistic_returns_logistic(self):
        npts = 10
        rr = perform_simple_regression(
            x_names=('a', 'b'),
            y_dict={'y': np.random.choice([True, False], size=npts)},
            npts=npts,
            regression_type='bernoulli')
        self.assertIsInstance(rr, regress.LogisticRegressionResult)

    def test_result_recapitulates_data(self):
        np.random.seed(1458)
        x = np.random.randn(100)
        y = 2 * x + 0.3
        noise = 1e-4 * np.random.randn(x.size)
        rr = public.fit_data(
            {'x': x}, {'y': y + noise}, max_order=2)
        prediction = rr.predict_for_map_model(x.reshape(-1, 1))

        for a, b in zip(prediction, y):
            self.assertAlmostEqual(a, b, places=3)

    def test_init_stores_x_offset_scale_if_set_as_dict(self):
        np.random.seed(1622)
        x = np.random.randn(100) * 10
        y = np.random.randn(x.size)

        x_dict = {'x': x}
        y_dict = {'y': y}
        offset = 0
        scale = 1
        x_offset_scale = {k: (offset, scale) for k in x_dict}

        rr = public.fit_data(
            x_dict,
            y_dict,
            x_offset_scale=x_offset_scale.copy(),
            max_order=1)

        self.assertEqual(rr.x_offset_scale.shape[0], len(x_offset_scale))
        for check in rr.x_offset_scale:
            self.assertEqual(check[0], offset)
            self.assertEqual(check[1], scale)


def perform_simple_regression(
        x_names=('a', 'b'), y_name='y', npts=31, x_dict=None, y_dict=None,
        regression_type='gaussian'):
    if x_dict is None:
        x_dict = {k: np.random.randn(npts) for k in x_names}
    if y_dict is None:
        y_dict = {y_name: np.random.randn(npts)}
    return public.fit_data(
        x_dict, y_dict, max_order=0, regression_type=regression_type)


if __name__ == '__main__':
    unittest.main()
