import itertools
import warnings

import numpy as np
from scipy.optimize import minimize, least_squares

from bayesregress.util import calculate_hessian
from bayesregress.likelihood import (
    GaussianLogLikelihood,
    BernoulliLogLikelihood,
    )
from bayesregress.prior import GaussianLogPrior
from bayesregress.posterior import LogPosterior
from bayesregress.predictors import NoninteractingMultivariatePredictor
from bayesregress.regressionresult import (
    GaussianRegressionResult,
    LogisticRegressionResult,
    )


def fit_data(x, y, **kwargs):
    args, kwargs, names = preprocess_inputs(x, y, **kwargs)
    factory = RegressionResultsGetter(*args, **kwargs)
    rr = factory.fit_data()
    set_names(rr, names)
    return rr


def preprocess_inputs(x, y, x_offset_scale=None, y_offset_scale=None, **kwargs):
    if isinstance(x, dict):
        x_names = list(x.keys())
        x = np.transpose([x[k] for k in x_names])
    else:
        x_names = None
        x = np.asarray(x)
    # cast 1D regressions to N-D like:
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if isinstance(y, dict):
        y_name = list(y.keys())[0]
        y = y[y_name]
        y = np.asarray(y)
    else:
        y_name = None
        y = np.asarray(y)

    if isinstance(x_offset_scale, dict):
        x_offset_scale = np.array([x_offset_scale[k] for k in x_names])
    if isinstance(y_offset_scale, dict):
        y_offset_scale = np.array([y_offset_scale[k] for k in y_names])

    args = (x, y)
    new_kwargs = {
        'x_offset_scale': x_offset_scale,
        'y_offset_scale': y_offset_scale,
        }
    kwargs.update(new_kwargs)

    names = {'x_names': x_names, 'y_name': y_name}
    return args, kwargs, names


def set_names(regression_result, names):
    regression_result.x_names = names['x_names']
    regression_result.y_name = names['y_name']


class RegressionResultsGetter(object):
    predictor_factory = NoninteractingMultivariatePredictor

    def __init__(self, x, y, regression_type='gaussian',
                 x_offset_scale=None, y_offset_scale=None, max_order=10):
        self.x = x
        self.y = y
        self.regression_type = regression_type
        self.max_order = max_order

        if x_offset_scale is None:
            x_offset_scale = self._find_x_offset_and_scale()
        self.x_offset_scale = np.asarray(x_offset_scale)
        if y_offset_scale is None:
            y_offset_scale = self._find_y_offset_and_scale()
        self.y_offset_scale = np.asarray(y_offset_scale)

        self.likelihood_class, self.result_class = self._select_classes()
        self._n_variables = len(self.x_offset_scale)

    def fit_data(self):
        orders_and_results = self._get_orders_and_results()

        kwargs = {
            "x_offset_scale": self.x_offset_scale,
            "orders_and_results": orders_and_results,
            "predictor": self.predictor_factory,
            }
        if self.regression_type == 'gaussian':
            kwargs.update({"y_offset_scale": self.y_offset_scale})

        result = self.result_class(**kwargs)
        return result

    def _find_x_offset_and_scale(self):
        return np.array([(xi.mean(), xi.std()) for xi in self.x.T])

    def _find_y_offset_and_scale(self):
        if 'gaussian' == self.regression_type:
            return np.array([self.y.mean(), self.y.std()])

    def _normalize_x(self):
        z = self.x.copy()
        for i, (offset, scale) in enumerate(self.x_offset_scale):
            z[:, i] -= offset
            z[:, i] /= scale
        return z

    def _normalize_y(self):
        y_raw = self.y
        # FIXME the scaling knows about the regression result scaling!
        if self.regression_type == 'gaussian':
            y_mean_std = [y_raw.mean(), y_raw.std()]
            y_normalized = (y_raw - y_mean_std[0]) / y_mean_std[1]
        else:
            y_normalized = y_raw
        return y_normalized

    def _get_orders_and_results(self):
        logger = EvidenceFunctionLogger(self.find_prediction)
        initial_guess = (2,) * self._n_variables
        if 1 == self._n_variables:
            _ = maximize_discrete_exhaustively(
                logger, initial_guess, max_order=self.max_order)
        else:
            _ = maximize_discrete_relevant(
                logger, initial_guess, max_order=self.max_order)
        orders_and_results = logger.orders_and_results
        return self._strip_convergence_errors_from(orders_and_results)

    def find_prediction(self, order):
        x = self._normalize_x()
        y = self._normalize_y()
        predictor = self._make_predictor(order)
        ll = self.likelihood_class(x, y, predictor)
        prior = self._make_prior()
        posterior = LogPosterior(prior, ll)
        regressor = BayesianRegressor(posterior)
        initial_guess = self._get_initial_guess(ll)
        return regressor.find_model_evidence(initial_guess)

    def _make_predictor(self, order):
        return NoninteractingMultivariatePredictor(order)

    def _make_prior(self):
        return GaussianLogPrior()

    # FIXME TDD
    def _get_initial_guess(self, ll):
        if 'gaussian' == self.regression_type:
            coeff_guess = least_squares(
                ll.evaluate_residuals,
                np.zeros(ll.predictor.ncoeffs),
                max_nfev=4,
                ).x
            noise_guess = np.log(ll.evaluate_residuals(coeff_guess).std())
            guess = np.append([noise_guess], coeff_guess)
        else:
            nparams = ll.additional_params + ll.predictor.ncoeffs
            guess = np.zeros(nparams)
        return guess

    def _select_classes(self):
        if self.regression_type.lower() == 'gaussian':
            ll = GaussianLogLikelihood
            rr = GaussianRegressionResult
        elif self.regression_type.lower() == 'bernoulli':
            ll = BernoulliLogLikelihood
            rr = LogisticRegressionResult
        else:
            raise ValueError(
                f"{self.regression_type} is not a valid regression type")
        return ll, rr

    def _strip_convergence_errors_from(self, orders_and_results):
        valid_orders = {
            o: r
            for o, r in orders_and_results.items()
            if r['log_evidence'] != EvidenceFunctionLogger._failed_convergence}
        return valid_orders


class BayesianRegressor(object):
    _number_of_minimization_loops = 4

    def __init__(self, log_posterior):
        """
        Parameters
        ----------
        log_posterior : callable
            A function which, given a set of parameters, returns a log
            posterior.
        """
        self.log_posterior = log_posterior

    def negative_log_posterior(self, params):
        return -self.log_posterior(params)

    def find_max_a_posteriori_params_for_given_order(self, initial_guess):
        return self.minimize(self.negative_log_posterior, initial_guess)

    def find_model_evidence(self, initial_guess):
        # cf. McKay pg 350
        pbest = self.find_max_a_posteriori_params_for_given_order(initial_guess)
        hess = self._calculate_negative_posterior_hessian(pbest.x)

        # I want to check for convergence errors, which are certain if
        # any eigenvalues of the Hessian is negative. However, these
        # hessians are very ill-conditioned, so we need to check if the
        # min eigenvalue is lower than what is reasonable numerically:

        accessible_volume = np.linalg.det(hess / (2 * np.pi))**-0.5
        if np.isnan(accessible_volume):
            raise ConvergenceError("hessian ill-conditioned!")
        log_posterior = self.log_posterior(pbest.x)
        log_evidence = log_posterior + np.log(accessible_volume)

        posterior_covariance = np.linalg.inv(hess)
        out = {
            'result': pbest,
            'log_evidence': log_evidence,
            'posterior_covariance': posterior_covariance,
            }
        return out

    @classmethod
    def minimize(cls, function, initial_guess):
        # fucking scipy.minimize stops prematurely, without converging.
        # If I just do either of these in 1 loop, the likelihood does
        # not increase as the number of parameters increases.
        # Even doing this, the likelihood starts decreasing when the
        # number of parameters is larger than 9. So.... wtf?

        for _ in range(cls._number_of_minimization_loops):
            r1 = minimize(
                function,
                initial_guess,
                method='Nelder-Mead',
                tol=1e-2)
            r2 = minimize(
                function,
                r1.x,
                method='BFGS',
                tol=1e-8)
            initial_guess = r2.x
        return r2

    def _calculate_negative_posterior_hessian(self, x):
        hess = calculate_hessian(self.negative_log_posterior, x)

        # These hessians can be very ill-conditioned, so I want to check
        # that what I'm getting is not nonsense. We do that by
        # estimating a scale from the first hessian, then re-calculating
        # the hessian again and ensuring that the answers are similar.
        # FIXME For now, I'm just ignoring this, but I need to come back
        # to a better way to calculate the hessians.
        """
        scale = np.sqrt(1.0 / np.diag(hess))
        h2 = calculate_hessian(
            self.negative_log_posterior, x, dl=1e-4 * scale)

        # are the matrix determinants greatly different?
        # If so, then the hessians are ill-conditioned
        det1 = np.linalg.det(hess)
        det2 = np.linalg.det(h2)
        determinants_bad = (
            det1 < 0 or det2 < 0 or
            np.abs(np.log(det1) - np.log(det2)) > 2)
        if determinants_bad:
            raise HessianIllConditionedError
        """
        # Are any of the eigenvalues negative?
        # If so, it may not have converged.
        eigvals = np.linalg.eigvalsh(hess)
        if eigvals.min() < -1e-13 * np.abs(eigvals).max():
            raise ConvergenceError("Not at a local minimum!!")

        return hess


class EvidenceFunctionLogger(object):
    _failed_convergence = -np.inf

    def __init__(self, function):
        self.function = function
        self.orders_and_results = dict()

    def __call__(self, order):
        if order in self.orders_and_results:
            this_result = self.orders_and_results[order]
        else:
            try:
                this_result = self.function(order)
                self.orders_and_results[order] = this_result
            except (ConvergenceError, HessianIllConditionedError) as e:
                msg = "{}, order={}".format(
                    e.__class__.__name__,
                    order)
                warnings.warn(msg)
                this_result = {'log_evidence': self._failed_convergence}
                self.orders_and_results[order] = this_result
        return this_result['log_evidence']


def maximize_discrete_exhaustively(function, initial_guess, max_order=10):
    n_variables = len(initial_guess)
    orders = range(max_order)
    results = dict()
    for x in itertools.product(orders, repeat=n_variables):
        y = function(x)
        results.update({y: x})
    best_value = max(results)
    best_x = results[best_value]
    return best_x


def maximize_discrete_relevant(function, initial_guess, max_order=10):
    results = dict()
    best_zn = tuple(initial_guess)
    n_variables = len(initial_guess)

    best_f = function(best_zn)
    results.update({best_zn: best_f})
    best_has_updated = True

    while best_has_updated:
        best_has_updated = False
        for do in [-1, 1]:
            for index in range(n_variables):
                new_order = [i for i in best_zn]
                new_order[index] = max(0, new_order[index] + do)
                new_order = tuple(new_order)

                if new_order not in results and max(new_order) <= max_order:
                    f = function(new_order)
                    results.update({tuple(new_order): f})
                    if f > best_f:
                        best_has_updated = True
                        best_f = f
                        best_zn = new_order

    best_value = max(results.values())
    best_x = [x for x in results if results[x] == best_value][0]
    return best_x



class ConvergenceError(Exception):
    pass


class HessianIllConditionedError(Exception):
    pass
