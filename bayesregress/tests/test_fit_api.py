import unittest
import warnings

import numpy as np

from bayesregress.regress import fit_data


TOLS = {'atol': 1e-13, 'rtol': 1e-13}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}


class TestDocs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # scipy optimize throws a lot of warnings
        warnings.simplefilter('ignore')

    @classmethod
    def tearDownClass(cls):
        warnings.simplefilter('default')

    def test_least_squares_regression(self):
        rng = np.random.default_rng(seed=1516)

        x = np.linspace(-10, 10, 901)
        noise = rng.standard_normal(x.shape)
        y = 0.5 * x * np.cos(0.5 * x) + noise

        result = fit_data(x, y)

        prediction = result.predict_for_map_model(x)
        residuals = y - prediction
        self.assertLessEqual(residuals.std(), 1.1 * noise.std())

    def test_logistic_regression(self):
        rng = np.random.default_rng(seed=1621)

        x = np.linspace(-1, 1, 4000)
        p = 1 - x**2
        y = rng.random(size=x.size) < p

        result = fit_data(x, y, regression_type='bernoulli')

        prediction = result.predict_for_map_model(x)
        delta = prediction - p
        self.assertLess(delta.std(), 0.03)

    def test_multivariate_regression(self):
        rng = np.random.default_rng(seed=1676)
        x1, x2, x3 = rng.standard_normal((3, 1300))
        model = (
            100 +
            3 * x1 +
            2 * x2**2 + 0.3 * x2 +
            np.sin(x3))
        noise = rng.standard_normal(x1.shape)
        y = model + noise
        x = {'x1': x1, 'x2': x2, 'x3': x3}

        result = fit_data(x, y)

        prediction = result.predict_for_map_model(x)
        residuals = y - prediction
        self.assertLessEqual(residuals.std(), 1.1 * noise.std())


if __name__ == '__main__':
    unittest.main()
