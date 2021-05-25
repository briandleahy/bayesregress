import unittest

import numpy as np

from bayesregress import predictors
from bayesregress.tests.common import *


TOLS = {"atol": 1e-12, "rtol": 1e-12}


class TestPredictor(unittest.TestCase):
    def test_call_raises_valueerror_if_coeffs_wrong_size(self):
        order = (3, 4)
        predictor = predictors.NoninteractingMultivariatePredictor(order)
        x = np.zeros((100, 2))
        wrongsize = list(range(predictor.ncoeffs * 2))
        wrongsize.remove(predictor.ncoeffs)
        for ncoeffs in wrongsize:
            with self.subTest(ncoeffs=ncoeffs):
                self.assertRaises(ValueError, predictor, x, np.zeros(ncoeffs))

    def test_call_raises_valueerror_if_x_wrong_shape(self):
        order = (3,)
        ncoeffs = sum(order) + 1
        predictor = predictors.NoninteractingMultivariatePredictor(order)
        x = np.zeros(100)
        self.assertRaises(ValueError, predictor, x.ravel(), np.zeros(ncoeffs))

    def test_call_correct_values(self):
        np.random.seed(1605)
        xpolycoeffs = np.random.randn(5)
        ypolycoeffs = np.random.randn(3)
        zpolycoeffs = np.random.randn(2)
        order = (xpolycoeffs.size - 1, ypolycoeffs.size, zpolycoeffs.size)
        coeffs = np.concatenate((xpolycoeffs, ypolycoeffs, zpolycoeffs))

        predictor = predictors.NoninteractingMultivariatePredictor(order)
        x = np.random.standard_normal((100, 3))

        f = np.polynomial.chebyshev.chebval
        correct = (
            f(x[:, 0], xpolycoeffs) +
            f(x[:, 1], np.append([0], ypolycoeffs)) +
            f(x[:, 2], np.append([0], zpolycoeffs))
            )
        out = predictor(x, coeffs)

        self.assertTrue(np.allclose(out, correct, **TOLS))


class TestNoninteractingMultivariatePredictor(unittest.TestCase):
    def test_stores_order(self):
        order = (2, 3)
        predictor = predictors.NoninteractingMultivariatePredictor(order)
        self.assertEqual(predictor.order, order)

    def test_stores_include_constant(self):
        for include_constant in [True, False]:
            predictor = predictors.NoninteractingMultivariatePredictor(
                (2, 3),
                include_constant=include_constant)
            self.assertEqual(predictor.include_constant, include_constant)

    def test_calculates_ncoeffs_include_constant_true(self):
        np.random.seed(227)
        for _ in range(10):
            dimensionality = np.random.randint(2, 5, 1)
            order = np.random.randint(0, 10, dimensionality)
            predictor = predictors.NoninteractingMultivariatePredictor(order)
            correct_ncoeffs = order.sum() + 1
            self.assertEqual(predictor.ncoeffs, correct_ncoeffs)

    def test_group_coeffs_gives_correct_shape(self):
        order = (3, 4)
        for ic in [True, False]:
            predictor = predictors.NoninteractingMultivariatePredictor(
                order, include_constant=ic)
            coeffs = np.ones(predictor.ncoeffs)
            grouped = predictor._group_coefficients(coeffs)
            for i in range(len(order)):
                with self.subTest(include_constant=ic, i=i):
                    self.assertEqual(grouped[i].shape, (order[i] + 1,))

    def test_group_coeffs_zero_pads_latter_coeffs_include_const_true(self):
        order = (3, 4)
        predictor = predictors.NoninteractingMultivariatePredictor(order)
        coeffs = np.ones(predictor.ncoeffs)
        grouped = predictor._group_coefficients(coeffs)
        for group in grouped[1:]:
            self.assertEqual(group[0], 0)

    def test_call_raises_valueerror_if_x_wrong_shape(self):
        order = (3, 4)
        predictor = predictors.NoninteractingMultivariatePredictor(order)
        coeffs = np.ones(predictor.ncoeffs)
        npts = 100
        for ndim in [1, 3, 4]:
            x = np.zeros((npts, ndim))
            with self.subTest(ndim=ndim):
                self.assertRaises(ValueError, predictor, x, coeffs)

    def test_reduces_to_chebval_in_1d_include_constant_true(self):
        np.random.seed(1605)
        x = np.random.standard_normal((100, 1))
        coeffs = np.random.standard_normal((5,))
        order = (coeffs.size - 1,)

        predictor = predictors.NoninteractingMultivariatePredictor(order)
        f = np.polynomial.chebyshev.chebval
        correct =  f(x[:, 0], coeffs)
        out = predictor(x, coeffs)
        self.assertTrue(np.allclose(out, correct, **TOLS))

    def test_calculates_ncoeffs_include_constant_false(self):
        np.random.seed(227)
        for _ in range(10):
            dimensionality = np.random.randint(2, 5, 1)
            order = np.random.randint(0, 10, dimensionality)
            predictor = predictors.NoninteractingMultivariatePredictor(
                order, include_constant=False)
            correct_ncoeffs = order.sum()
            self.assertEqual(predictor.ncoeffs, correct_ncoeffs)

    def test_group_coeffs_treats_first_constant_term_correctly(self):
        order = (3, 4)
        for ic in [True, False]:
            predictor = predictors.NoninteractingMultivariatePredictor(
                order, include_constant=ic)
            coeffs = np.ones(predictor.ncoeffs)
            grouped = predictor._group_coefficients(coeffs)
            correct = (1 if ic else 0)
            with self.subTest(include_constant=ic):
                self.assertEqual(grouped[0][0], correct)


class TestBinnedPredictor(unittest.TestCase):
    def test_nvariables_is_1(self):
        binned = predictors.BinnedPredictor(np.arange(10))
        self.assertEqual(binned._nvariables, 1)

    def test_ncoeffs(self):
        ncoeffs = 10
        nedges = ncoeffs - 1
        binned = predictors.BinnedPredictor(np.arange(nedges))
        self.assertEqual(binned.ncoeffs, ncoeffs)

    def test_init_raises_error_on_2d_bins(self):
        self.assertRaises(
            ValueError,
            predictors.BinnedPredictor,
            np.arange(10).reshape(5, 2))

    def test_init_raises_error_on_unordered_bins(self):
        np.random.seed(1103)
        self.assertRaises(
            ValueError,
            predictors.BinnedPredictor,
            np.random.randn(11))

    def test_call_raises_error_on_multivariate_x(self):
        binned = predictors.BinnedPredictor(np.arange(10))
        np.random.seed(1106)
        x = np.random.randn(10, 2)
        coeffs = np.random.randn(binned.ncoeffs)
        self.assertRaises(ValueError, binned, x, coeffs)

    def test_call_raises_error_on_coeffs_wrong_shape(self):
        binned = predictors.BinnedPredictor(np.arange(10))
        np.random.seed(1106)
        x = np.random.randn(10, 2)
        coeffs_wrong_size = np.random.randn(binned.ncoeffs - 1)
        self.assertRaises(ValueError, binned, x, coeffs_wrong_size)

    def test_call_gives_correct_answer(self):
        np.random.seed(1109)
        bin_centers = np.arange(10)
        bin_edges = bin_centers[:-1] + 0.5
        bin_values = np.random.randn(bin_centers.size)

        predictor = predictors.BinnedPredictor(bin_edges)

        out = predictor(bin_centers.reshape(-1, 1), bin_values)
        for i in range(out.size):
            with self.subTest(i=i):
                self.assertEqual(out[i], bin_values[i])

    def test_call_gives_correct_answer_shuffling(self):
        np.random.seed(1125)
        bin_centers = np.arange(10)
        bin_edges = bin_centers[:-1] + 0.5
        bin_values = np.random.randn(bin_centers.size)

        predictor = predictors.BinnedPredictor(bin_edges)
        indices = np.arange(bin_centers.size)
        for _ in range(10):
             np.random.shuffle(indices)
             x = bin_centers[indices].reshape(-1, 1)
             out = predictor(x, bin_values)
             self.assertTrue(np.all(out == bin_values[indices]))


class TestCompositePredictor(unittest.TestCase):
    def test_init_sets_nvariables(self):
        order1 = (1, 2, 3)
        order2 = (4, 5)
        p1 = predictors.NoninteractingMultivariatePredictor(order1)
        p2 = predictors.NoninteractingMultivariatePredictor(order2)
        composite = predictors.CompositePredictor(p1, p2)

        self.assertEqual(composite._nvariables, len(order1) + len(order2))

    def test_init_sets_ncoeffs(self):
        order1 = (1, 2, 3)
        order2 = (4, 5)
        p1 = predictors.NoninteractingMultivariatePredictor(order1)
        p2 = predictors.NoninteractingMultivariatePredictor(order2)
        composite = predictors.CompositePredictor(p1, p2)

        self.assertEqual(composite.ncoeffs, 2 + sum(order1) + sum(order2))

    def test_call_gives_correct_values(self):
        np.random.seed(1423)

        x1 = np.random.randn(100, 2)
        order1 = (1,) * x1.shape[1]
        c1 = np.random.randn(sum(order1) + 1)
        p1 = predictors.NoninteractingMultivariatePredictor(order1)

        x2 = np.random.randn(100, 3)
        order2 = (1,) * x2.shape[1]
        c2 = np.random.randn(sum(order2) + 1)
        p2 = predictors.NoninteractingMultivariatePredictor(order2)

        composite = predictors.CompositePredictor(p1, p2)
        xc = np.concatenate([x1, x2], axis=1)
        cc = np.concatenate([c1, c2], axis=0)

        correct = p1(x1, c1) + p2(x2, c2)
        out = composite(xc, cc)
        for c, o in zip(correct, out):
            self.assertAlmostEqual(c, o, places=11)

    def test_call_raise_error_if_x_wrong_shape(self):
        composite = predictors.CompositePredictor(
            predictors.NoninteractingMultivariatePredictor((1,)),
            predictors.NoninteractingMultivariatePredictor((1,)))
        x_wrong_shape = np.zeros((100, composite._nvariables - 1))
        coeffs = np.zeros(composite.ncoeffs)
        self.assertRaises(ValueError, composite, x_wrong_shape, coeffs)

    def test_call_raise_error_if_coeffs_wrong_shape(self):
        composite = predictors.CompositePredictor(
            predictors.NoninteractingMultivariatePredictor((1,)),
            predictors.NoninteractingMultivariatePredictor((1,)))
        x = np.zeros((100, composite._nvariables))
        coeffs_wrong_shape = np.zeros(composite.ncoeffs - 1)
        self.assertRaises(ValueError, composite, x, coeffs_wrong_shape)

    def test_group_coefficients(self):
        np.random.seed(1439)
        p1 = predictors.NoninteractingMultivariatePredictor((1, 2))
        p2 = predictors.NoninteractingMultivariatePredictor((5,))
        composite = predictors.CompositePredictor(p1, p2)

        coeffs = np.random.randn(composite.ncoeffs)
        grouped = composite._group_coefficients(coeffs)

        for predictor, these_coeffs in zip([p1, p2], grouped):
            self.assertEqual(these_coeffs.size, predictor.ncoeffs)

    def test_repr(self):
        p1 = predictors.NoninteractingMultivariatePredictor((1, 2))
        p2 = predictors.NoninteractingMultivariatePredictor((5,))
        composite = predictors.CompositePredictor(p1, p2)

        the_repr = repr(composite)
        self.assertIn('{}('.format(composite.__class__.__name__), the_repr)


class TestCompositePredictorFactory(unittest.TestCase):
    def test_make_predictor_returns_composite(self):
        p1 = predictors.NoninteractingMultivariatePredictor((1, 2))
        factory = predictors.CompositePredictorFactory(p1)
        composite = factory.make_predictor((2,))
        self.assertIsInstance(composite, predictors.CompositePredictor)

    def test_make_predictor_gives_correct_composite(self):
        p1 = predictors.NoninteractingMultivariatePredictor((1, 2))
        factory = predictors.CompositePredictorFactory(p1)

        new_order = (3,)
        composite = factory.make_predictor(new_order)

        self.assertIs(composite.predictor1, p1)
        self.assertIsInstance(
            composite.predictor2,
            predictors.NoninteractingMultivariatePredictor)
        self.assertEqual(composite.predictor2.order, new_order)

    def test_call_calls_make_predictor(self):
        p1 = predictors.NoninteractingMultivariatePredictor((1, 2))
        factory = predictors.CompositePredictorFactory(p1)
        new_order = (3,)
        composite = factory(new_order)
        self.assertIs(composite.predictor1, p1)
        self.assertEqual(composite.predictor2.order, new_order)

    def test_call_accepts_kwargs(self):
        p1 = predictors.NoninteractingMultivariatePredictor((1, 2))
        factory = predictors.CompositePredictorFactory(p1)
        new_order = (3,)
        composite = factory(new_order, include_constant=False)
        self.assertFalse(composite.predictor2.include_constant)


class TestMisc(unittest.TestCase):
    def test_prepad_with_0(self):
        np.random.seed(1521)
        x = np.random.randn(5)
        y = predictors.prepad_with_0(x)
        self.assertEqual(y[0], 0)
        for a, b in zip(x, y[1:]):
            self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
