import unittest

import numpy as np
from scipy.stats import binom
import pgmpy

from bayesregress import util


TOLS = {"atol": 1e-12, "rtol": 1e-12}


class TestDifferentiation(unittest.TestCase):
    def test_gradient_returns_correct_shape_on_scalar_function(self):
        x = np.zeros(1)
        f = lambda x: np.sin(x[0])
        grad = util._calculate_gradient(f, x)
        self.assertEqual(grad.shape, x.shape)

    def test_gradient_returns_correct_shape_on_vector_function(self):
        x = np.zeros(2)
        f = lambda x: np.sin(x[0] + x[1])
        grad = util._calculate_gradient(f, x)
        self.assertEqual(grad.shape, x.shape)

    def test_hessian_returns_correct_shape_on_scalar_function(self):
        x = np.zeros(1)
        f = lambda x: np.sin(x[0])
        hess = util.calculate_hessian(f, x)
        self.assertEqual(hess.shape, 2 * x.shape)

    def test_hessian_returns_correct_shape_on_vector_function(self):
        x = np.zeros(2)
        f = lambda x: np.sin(x[0] + x[1])
        hess = util.calculate_hessian(f, x)
        self.assertEqual(hess.shape, 2 * x.shape)

    def test_gradient_returns_correct_answer_on_linear_1d(self):
        grad_true = 0.952  # random
        f = lambda x: grad_true * x[0]
        grad = util._calculate_gradient(f, np.zeros(1))
        self.assertAlmostEqual(grad[0], grad_true, 6)

    def test_hessian_returns_correct_answer_on_quadratic_1d(self):
        f = lambda x: 0.5 * x[0]**2
        hess = util.calculate_hessian(f, np.zeros(1))
        self.assertAlmostEqual(hess[0, 0], 1, 6)

    def test_hessian_returns_correct_answer_on_quadratic_2d(self):
        x = np.zeros(2)
        np.random.seed(1426)
        hess_true = np.random.randn(2, 2); hess_true += hess_true.T

        f = lambda xy: 0.5 * hess_true.dot(xy).dot(xy)
        hess = util.calculate_hessian(f, x)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(hess_true[i, j], hess[i, j], 6)

    def test_gradient_vector_dl_gives_correct_answer(self):
        np.random.seed(957)
        grad_true = np.random.randn(9)
        f = lambda x: grad_true.dot(x)
        dl = np.random.randn(grad_true.size)
        grad_out = util._calculate_gradient(f, np.zeros(grad_true.size), dl=dl)
        for t, o in zip(grad_true, grad_out):
            self.assertAlmostEqual(t, o, 6)

    def test_gradient_vector_dl_raises_error_if_dl_wrong_size(self):
        n = 3
        f = lambda x: np.ones(x.size).dot(x)
        self.assertRaisesRegex(
            ValueError, 'dl wrong shape', util._calculate_gradient, f,
            np.ones(n), dl=np.ones(n - 1))

    def test_hessian_vector_dl_gives_correct_answer(self):
        np.random.seed(953)
        x = np.zeros(4)
        hess_true = np.random.randn(x.size, x.size); hess_true += hess_true.T

        f = lambda x: 0.5 * hess_true.dot(x).dot(x)
        dl = np.random.randn(x.size)
        hess = util.calculate_hessian(f, x, dl=dl)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(hess_true[i, j], hess[i, j], 6)

    def test_hessian_vector_dl_raises_error_if_dl_wrong_size(self):
        np.random.seed(953)
        x = np.zeros(4)
        hess_true = np.random.randn(x.size, x.size); hess_true += hess_true.T

        f = lambda x: 0.5 * hess_true.dot(x).dot(x)
        dl = np.random.randn(x.size - 1)
        self.assertRaisesRegex(
            ValueError, 'dl wrong shape', util.calculate_hessian, f, x, dl=dl)

    def test_estimate_best_hessian_dl(self):
        np.random.seed(1533)
        scale = np.exp(10 * np.random.randn(4))

        f = lambda x: 0.5 * np.sum((x / scale)**2)
        scale_estimated = util._estimate_function_scale_from_hessian(
            f, np.zeros(scale.size))

        ratio = scale_estimated / scale
        for r in ratio:
            self.assertLess(r, 5)
            self.assertGreater(r, 0.2)

    def test_hessian_correct_on_asymmetric_function_when_dl_auto(self):
        np.random.seed(1549)
        scale = np.exp(10 * np.random.randn(4))

        f = lambda x: 1 - np.cos(x / scale).sum()
        hess = util.calculate_hessian(f, np.zeros(scale.size), dl='auto')

        hess_true = np.eye(scale.size)
        for i in range(scale.size):
            hess_true[i, i] = scale[i]**-2

        for a, b in zip(hess.ravel(), hess_true.ravel()):
            if 0 == b:
                self.assertAlmostEqual(a, b, places=4)
            else:
                self.assertAlmostEqual(a / b, 1, places=4)


class TestCalculatePoissonBinomialPmf(unittest.TestCase):
    def test_outputs_correct_size(self):
        np.random.seed(940)
        for number_of_attempts in [1, 2, 5, 13]:
            trial_probs = np.random.random((number_of_attempts,))
            success_pmf = util.calculate_poisson_binomial_pmf(trial_probs)
            self.assertEqual(success_pmf.size, number_of_attempts + 1)

    def test_correct_on_2_trials_unequal_probabilities(self):
        np.random.seed(942)
        trial_probs = np.random.random((2,))
        p1, p2 = trial_probs
        success_pmf_calculated = util.calculate_poisson_binomial_pmf(
            trial_probs)
        success_pmf_true = np.array([
            (1 - p1) * (1 - p2),  # both fail
            p1 * (1 - p2) + p2 * (1 - p1),  # one succeeds
            p1 * p2,  # both succeed
            ])
        correct = np.allclose(success_pmf_calculated, success_pmf_true, **TOLS)
        self.assertTrue(correct)

    def test_correct_on_equal_probabilities_returns_binomial(self):
        np.random.seed(946)
        trial_prob = np.random.random()
        n_trials = 10
        trial_probs = np.full(n_trials, trial_prob)

        success_pmf_calculated = util.calculate_poisson_binomial_pmf(
            trial_probs)
        success_pmf_true = binom.pmf(
            np.arange(0, n_trials + 1),  # # of successes
            n_trials,  # number of total trials
            trial_prob,
            )
        correct = np.allclose(success_pmf_calculated, success_pmf_true, **TOLS)
        self.assertTrue(correct)


if __name__ == '__main__':
    unittest.main()

