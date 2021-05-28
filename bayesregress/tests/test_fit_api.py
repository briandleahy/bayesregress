import unittest

import numpy as np

from bayesregress.regress import fit_data


TOLS = {'atol': 1e-13, 'rtol': 1e-13}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}


class TestDocs(unittest.TestCase):
    def test_least_squares_regression(self):
        rng = np.random.default_rng(seed=1516)

        x = np.linspace(-10, 10, 901)
        noise = rng.standard_normal(x.shape)
        y = 0.5 * x * np.cos(0.5 * x) + noise

        result = fit_data(x, y)

        prediction = result.predict_for_map_model(x)
        residuals = y - prediction
        self.assertLessEqual(residuals.std(), 1.1 * noise.std())


if __name__ == '__main__':
    unittest.main()
