import unittest
import warnings

import numpy as np

from bayesregress import regress


TOLS = {"atol": 1e-12, "rtol": 1e-12}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}


class TestLogLikelihood(unittest.TestCase):
    def test_additional_params_is_0(self):
        self.assertEqual(regress.LogLikelihood.additional_params, 0)

    def test_default_predictor_is_chebval(self):
        ll = regress.LogLikelihood(predictor=None)
        self.assertEqual(ll.predictor, np.polynomial.chebyshev.chebval)

    def test_log_likelihood_not_implemented(self):
        ll = regress.LogLikelihood(predictor=None)
        self.assertRaises(NotImplementedError, ll.log_likelihood, 0)

    def test_call_calls_log_likelihood(self):
        ll = MockLL(predictor=None)
        self.assertFalse(ll._log_likelihood_called)
        ll(0)
        self.assertTrue(ll._log_likelihood_called)


class TestGaussianLogLikelihood(unittest.TestCase):
    def test_additional_params_is_1(self):
        self.assertEqual(regress.GaussianLogLikelihood.additional_params, 1)

    def test_init_stores_xy(self):
        x, y = generate_correlated_pair_of_data()
        ll = regress.GaussianLogLikelihood(x, y)
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
        ll = regress.GaussianLogLikelihood(x, y)
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
        ll = regress.GaussianLogLikelihood(x, y)
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
        ll = regress.BinomialLogLikelihood(x, trials, successes)
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
        calc = regress.BinomialLogLikelihood(x, tried, success)

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
        ll = regress.BernoulliLogLikelihood(x, successes)
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
        ll = regress.BernoulliLogLikelihood(x, successes)

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


class TestGaussianLogPrior(unittest.TestCase):
    def test_log_prior_scalar(self):
        prior = regress.GaussianLogPrior()
        p = prior(np.ones(5))
        self.assertEqual(np.size(p), 1)

    def test_log_prior_decreases_as_paramvalues_increase(self):
        scales = np.arange(10)
        np.random.seed(202)
        params = np.random.randn(3)
        log_prior = regress.GaussianLogPrior()
        log_priors = [log_prior(s * params) for s in scales]
        self.assertTrue(np.all(np.diff(log_priors) < 0))

    def test_log_prior_quadratic(self):
        log_prior = regress.GaussianLogPrior()
        prior0 = log_prior(np.zeros(1))
        prior3 = log_prior(np.zeros(1) + 3)
        self.assertEqual(prior0 - prior3, 0.5 * 3**2)

    def test_log_prior_includes_gaussian_normalization(self):
        n_terms = np.arange(1, 10, dtype='int')
        for std in [1.0, 1.2]:
            log_prior = regress.GaussianLogPrior(std=std)
            priors = [log_prior(np.zeros(n)) for n in n_terms]
            log_normalization = np.log(1.0 / np.sqrt(2 * np.pi * std**2))
            for dp in np.diff(priors):
                with self.subTest(std=std):
                    self.assertAlmostEqual(dp, log_normalization, places=10)


class TestUniformDistLogitLogPrior(unittest.TestCase):
    def test_log_prior_scalar(self):
        prior = regress.UniformDistLogitLogPrior()
        p = prior(np.ones(5))
        self.assertEqual(np.size(p), 1)

    def test_log_prior_decreases_as_paramvalues_increase(self):
        np.random.seed(1715)
        scales = np.arange(10)
        params = np.random.randn(3)
        log_prior = regress.UniformDistLogitLogPrior()
        log_priors = [log_prior(s * params) for s in scales]
        self.assertTrue(np.all(np.diff(log_priors) < 0))

    def test_dist_is_normalized(self):
        t = np.linspace(-3e2, 3e2, int(1e3))
        log_prior = regress.UniformDistLogitLogPrior()
        rho = np.exp([log_prior(ti) for ti in t])

        normalization = np.trapz(rho, t)
        self.assertAlmostEqual(normalization, 1.0, places=10)

    def test_dist_mean_prob_from_logit_is_correct(self):
        # <p> = 0.5, where p = np.exp(z) / (1 + np.exp(z))
        t = np.linspace(-3e2, 3e2, int(1e3))
        log_prior = regress.UniformDistLogitLogPrior()
        rho = np.exp([log_prior(ti) for ti in t])
        p = 1.0 / (1 + np.exp(t))

        mean = np.trapz(p * rho, t)
        self.assertAlmostEqual(mean, 0.5, places=10)

    def test_dist_var_prob_from_logit_is_correct(self):
        # really test <p^2> = 1/3, where p = np.exp(z) / (1 + np.exp(z))
        t = np.linspace(-3e2, 3e2, int(1e3))
        log_prior = regress.UniformDistLogitLogPrior()
        rho = np.exp([log_prior(ti) for ti in t])
        p = 1.0 / (1 + np.exp(t))

        var = np.trapz(p**2 * rho, t)
        self.assertAlmostEqual(var, 1.0 / 3, places=10)


class TestLogPosterior(unittest.TestCase):
    def test_log_posterior(self):
        np.random.seed(218)
        params = np.random.randn(4)
        log_prior = regress.GaussianLogPrior()
        log_likelihood = make_gaussian_log_likelihood()
        log_posterior = regress.LogPosterior(log_prior, log_likelihood)

        tocheck = log_posterior(params)
        correct = (log_prior(params) +
                   log_likelihood(params))
        self.assertAlmostEqual(tocheck, correct, places=12)


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


class TestPredictor(unittest.TestCase):
    def test_call_raises_valueerror_if_coeffs_wrong_size(self):
        order = (3, 4)
        predictor = regress.NoninteractingMultivariatePredictor(order)
        x = np.zeros((100, 2))
        wrongsize = list(range(predictor.ncoeffs * 2))
        wrongsize.remove(predictor.ncoeffs)
        for ncoeffs in wrongsize:
            with self.subTest(ncoeffs=ncoeffs):
                self.assertRaises(ValueError, predictor, x, np.zeros(ncoeffs))

    def test_call_raises_valueerror_if_x_wrong_shape(self):
        order = (3,)
        ncoeffs = sum(order) + 1
        predictor = regress.NoninteractingMultivariatePredictor(order)
        x = np.zeros(100)
        self.assertRaises(ValueError, predictor, x.ravel(), np.zeros(ncoeffs))

    def test_call_correct_values(self):
        np.random.seed(1605)
        xpolycoeffs = np.random.randn(5)
        ypolycoeffs = np.random.randn(3)
        zpolycoeffs = np.random.randn(2)
        order = (xpolycoeffs.size - 1, ypolycoeffs.size, zpolycoeffs.size)
        coeffs = np.concatenate((xpolycoeffs, ypolycoeffs, zpolycoeffs))

        predictor = regress.NoninteractingMultivariatePredictor(order)
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
        predictor = regress.NoninteractingMultivariatePredictor(order)
        self.assertEqual(predictor.order, order)

    def test_stores_include_constant(self):
        for include_constant in [True, False]:
            predictor = regress.NoninteractingMultivariatePredictor(
                (2, 3),
                include_constant=include_constant)
            self.assertEqual(predictor.include_constant, include_constant)

    def test_calculates_ncoeffs_include_constant_true(self):
        np.random.seed(227)
        for _ in range(10):
            dimensionality = np.random.randint(2, 5, 1)
            order = np.random.randint(0, 10, dimensionality)
            predictor = regress.NoninteractingMultivariatePredictor(order)
            correct_ncoeffs = order.sum() + 1
            self.assertEqual(predictor.ncoeffs, correct_ncoeffs)

    def test_group_coeffs_gives_correct_shape(self):
        order = (3, 4)
        for ic in [True, False]:
            predictor = regress.NoninteractingMultivariatePredictor(
                order, include_constant=ic)
            coeffs = np.ones(predictor.ncoeffs)
            grouped = predictor._group_coefficients(coeffs)
            for i in range(len(order)):
                with self.subTest(include_constant=ic, i=i):
                    self.assertEqual(grouped[i].shape, (order[i] + 1,))

    def test_group_coeffs_zero_pads_latter_coeffs_include_const_true(self):
        order = (3, 4)
        predictor = regress.NoninteractingMultivariatePredictor(order)
        coeffs = np.ones(predictor.ncoeffs)
        grouped = predictor._group_coefficients(coeffs)
        for group in grouped[1:]:
            self.assertEqual(group[0], 0)

    def test_call_raises_valueerror_if_x_wrong_shape(self):
        order = (3, 4)
        predictor = regress.NoninteractingMultivariatePredictor(order)
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

        predictor = regress.NoninteractingMultivariatePredictor(order)
        f = np.polynomial.chebyshev.chebval
        correct =  f(x[:, 0], coeffs)
        out = predictor(x, coeffs)
        self.assertTrue(np.allclose(out, correct, **TOLS))

    def test_calculates_ncoeffs_include_constant_false(self):
        np.random.seed(227)
        for _ in range(10):
            dimensionality = np.random.randint(2, 5, 1)
            order = np.random.randint(0, 10, dimensionality)
            predictor = regress.NoninteractingMultivariatePredictor(
                order, include_constant=False)
            correct_ncoeffs = order.sum()
            self.assertEqual(predictor.ncoeffs, correct_ncoeffs)

    def test_group_coeffs_treats_first_constant_term_correctly(self):
        order = (3, 4)
        for ic in [True, False]:
            predictor = regress.NoninteractingMultivariatePredictor(
                order, include_constant=ic)
            coeffs = np.ones(predictor.ncoeffs)
            grouped = predictor._group_coefficients(coeffs)
            correct = (1 if ic else 0)
            with self.subTest(include_constant=ic):
                self.assertEqual(grouped[0][0], correct)



class TestBinnedPredictor(unittest.TestCase):
    def test_nvariables_is_1(self):
        binned = regress.BinnedPredictor(np.arange(10))
        self.assertEqual(binned._nvariables, 1)

    def test_ncoeffs(self):
        ncoeffs = 10
        nedges = ncoeffs - 1
        binned = regress.BinnedPredictor(np.arange(nedges))
        self.assertEqual(binned.ncoeffs, ncoeffs)

    def test_init_raises_error_on_2d_bins(self):
        self.assertRaises(
            ValueError,
            regress.BinnedPredictor,
            np.arange(10).reshape(5, 2))

    def test_init_raises_error_on_unordered_bins(self):
        np.random.seed(1103)
        self.assertRaises(
            ValueError,
            regress.BinnedPredictor,
            np.random.randn(11))

    def test_call_raises_error_on_multivariate_x(self):
        binned = regress.BinnedPredictor(np.arange(10))
        np.random.seed(1106)
        x = np.random.randn(10, 2)
        coeffs = np.random.randn(binned.ncoeffs)
        self.assertRaises(ValueError, binned, x, coeffs)

    def test_call_raises_error_on_coeffs_wrong_shape(self):
        binned = regress.BinnedPredictor(np.arange(10))
        np.random.seed(1106)
        x = np.random.randn(10, 2)
        coeffs_wrong_size = np.random.randn(binned.ncoeffs - 1)
        self.assertRaises(ValueError, binned, x, coeffs_wrong_size)

    def test_call_gives_correct_answer(self):
        np.random.seed(1109)
        bin_centers = np.arange(10)
        bin_edges = bin_centers[:-1] + 0.5
        bin_values = np.random.randn(bin_centers.size)

        predictor = regress.BinnedPredictor(bin_edges)

        out = predictor(bin_centers.reshape(-1, 1), bin_values)
        for i in range(out.size):
            with self.subTest(i=i):
                self.assertEqual(out[i], bin_values[i])

    def test_call_gives_correct_answer_shuffling(self):
        np.random.seed(1125)
        bin_centers = np.arange(10)
        bin_edges = bin_centers[:-1] + 0.5
        bin_values = np.random.randn(bin_centers.size)

        predictor = regress.BinnedPredictor(bin_edges)
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
        p1 = regress.NoninteractingMultivariatePredictor(order1)
        p2 = regress.NoninteractingMultivariatePredictor(order2)
        composite = regress.CompositePredictor(p1, p2)

        self.assertEqual(composite._nvariables, len(order1) + len(order2))

    def test_init_sets_ncoeffs(self):
        order1 = (1, 2, 3)
        order2 = (4, 5)
        p1 = regress.NoninteractingMultivariatePredictor(order1)
        p2 = regress.NoninteractingMultivariatePredictor(order2)
        composite = regress.CompositePredictor(p1, p2)

        self.assertEqual(composite.ncoeffs, 2 + sum(order1) + sum(order2))

    def test_call_gives_correct_values(self):
        np.random.seed(1423)

        x1 = np.random.randn(100, 2)
        order1 = (1,) * x1.shape[1]
        c1 = np.random.randn(sum(order1) + 1)
        p1 = regress.NoninteractingMultivariatePredictor(order1)

        x2 = np.random.randn(100, 3)
        order2 = (1,) * x2.shape[1]
        c2 = np.random.randn(sum(order2) + 1)
        p2 = regress.NoninteractingMultivariatePredictor(order2)

        composite = regress.CompositePredictor(p1, p2)
        xc = np.concatenate([x1, x2], axis=1)
        cc = np.concatenate([c1, c2], axis=0)

        correct = p1(x1, c1) + p2(x2, c2)
        out = composite(xc, cc)
        for c, o in zip(correct, out):
            self.assertAlmostEqual(c, o, places=11)

    def test_call_raise_error_if_x_wrong_shape(self):
        composite = regress.CompositePredictor(
            regress.NoninteractingMultivariatePredictor((1,)),
            regress.NoninteractingMultivariatePredictor((1,)))
        x_wrong_shape = np.zeros((100, composite._nvariables - 1))
        coeffs = np.zeros(composite.ncoeffs)
        self.assertRaises(ValueError, composite, x_wrong_shape, coeffs)

    def test_call_raise_error_if_coeffs_wrong_shape(self):
        composite = regress.CompositePredictor(
            regress.NoninteractingMultivariatePredictor((1,)),
            regress.NoninteractingMultivariatePredictor((1,)))
        x = np.zeros((100, composite._nvariables))
        coeffs_wrong_shape = np.zeros(composite.ncoeffs - 1)
        self.assertRaises(ValueError, composite, x, coeffs_wrong_shape)

    def test_group_coefficients(self):
        np.random.seed(1439)
        p1 = regress.NoninteractingMultivariatePredictor((1, 2))
        p2 = regress.NoninteractingMultivariatePredictor((5,))
        composite = regress.CompositePredictor(p1, p2)

        coeffs = np.random.randn(composite.ncoeffs)
        grouped = composite._group_coefficients(coeffs)

        for predictor, these_coeffs in zip([p1, p2], grouped):
            self.assertEqual(these_coeffs.size, predictor.ncoeffs)

    def test_repr(self):
        p1 = regress.NoninteractingMultivariatePredictor((1, 2))
        p2 = regress.NoninteractingMultivariatePredictor((5,))
        composite = regress.CompositePredictor(p1, p2)

        the_repr = repr(composite)
        self.assertIn('{}('.format(composite.__class__.__name__), the_repr)


class TestCompositePredictorFactory(unittest.TestCase):
    def test_make_predictor_returns_composite(self):
        p1 = regress.NoninteractingMultivariatePredictor((1, 2))
        factory = regress.CompositePredictorFactory(p1)
        composite = factory.make_predictor((2,))
        self.assertIsInstance(composite, regress.CompositePredictor)

    def test_make_predictor_gives_correct_composite(self):
        p1 = regress.NoninteractingMultivariatePredictor((1, 2))
        factory = regress.CompositePredictorFactory(p1)

        new_order = (3,)
        composite = factory.make_predictor(new_order)

        self.assertIs(composite.predictor1, p1)
        self.assertIsInstance(
            composite.predictor2,
            regress.NoninteractingMultivariatePredictor)
        self.assertEqual(composite.predictor2.order, new_order)

    def test_call_calls_make_predictor(self):
        p1 = regress.NoninteractingMultivariatePredictor((1, 2))
        factory = regress.CompositePredictorFactory(p1)
        new_order = (3,)
        composite = factory(new_order)
        self.assertIs(composite.predictor1, p1)
        self.assertEqual(composite.predictor2.order, new_order)

    def test_call_accepts_kwargs(self):
        p1 = regress.NoninteractingMultivariatePredictor((1, 2))
        factory = regress.CompositePredictorFactory(p1)
        new_order = (3,)
        composite = factory(new_order, include_constant=False)
        self.assertFalse(composite.predictor2.include_constant)


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
            regress.GaussianLogLikelihood(x, y))

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

        likelihood = regress.GaussianLogLikelihood(x, y)
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
        predictor = regress.NoninteractingMultivariatePredictor(order)
        ll = regress.GaussianLogLikelihood(x, y, predictor)
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
        ll = regress.GaussianLogLikelihood(x, y)
        posterior = regress.LogPosterior(
            regress.GaussianLogPrior(),
            ll)
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


class TestMisc(unittest.TestCase):
    def test_prepad_with_0(self):
        np.random.seed(1521)
        x = np.random.randn(5)
        y = regress.prepad_with_0(x)
        self.assertEqual(y[0], 0)
        for a, b in zip(x, y[1:]):
            self.assertEqual(a, b)


# ~~~~~~~~~~~~~~~~~~~~ Supporting functions for tests ~~~~~~~~~~~~~~~~~~


def generate_correlated_pair_of_data(npts=100, correlation=0.5, seed=72):
    # What is a reasonable empirical correlation?
    # <sum_i x_i y_i> / N = r
    # <(sum_i x_i y_i)^2 > / N^2 = r^2 + (1 + 2r - r^2) / N
    # so the empirical correlation is ~N(r, sqrt((1+2r-r^2)/N) )
    np.random.seed(seed)
    x = np.random.randn(npts)
    y = correlation * x + np.sqrt(1 - correlation**2) * np.random.randn(npts)
    return x, y


def make_regressor():
    posterior = make_gaussian_log_posterior()
    return regress.BayesianRegressor(posterior)


def make_gaussian_log_posterior():
    likelihood = make_gaussian_log_likelihood()
    prior = regress.GaussianLogPrior(mean=0, std=1)
    posterior = regress.LogPosterior(prior, likelihood)
    return posterior


def make_gaussian_log_likelihood():
    x, y = generate_correlated_pair_of_data()
    return regress.GaussianLogLikelihood(x, y)


def make_binomial_log_likelihood():
    npts = 100
    x = np.random.randn(npts)
    trials = np.random.randint(1, 30, npts)
    successes = np.random.randint(0, trials, npts)
    return regress.BinomialLogLikelihood(x, trials, successes)


def make_bernoulli_log_likelihood():
    npts = 100
    x = np.random.randn(npts)
    successes = np.random.choice([True, False], size=npts)
    return regress.BernoulliLogLikelihood(x, successes)


class MockLL(regress.LogLikelihood):
    def __init__(self, *args, **kwargs):
        super(MockLL, self).__init__(*args, **kwargs)
        self._log_likelihood_called = False

    def log_likelihood(self, params):
        self._log_likelihood_called = True


if __name__ == '__main__':
    unittest.main()

