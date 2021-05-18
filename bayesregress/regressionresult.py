import os
import itertools
import warnings

import numpy as np
from scipy.optimize import least_squares

from bayesregress.regress import (
    NoninteractingMultivariatePredictor,
    BayesianRegressor,
    ConvergenceError,
    HessianIllConditionedError,
    )
from bayesregress.likelihood import (
    LogLikelihood,
    GaussianLogLikelihood,
    BernoulliLogLikelihood,
    )
from bayesregress.prior import GaussianLogPrior
from bayesregress.posterior import LogPosterior

def make_regression_result(x_dict, y_dict, **kwargs):
    factory = RegressionResultsGetter(x_dict, y_dict, **kwargs)
    return factory.make_regression_result()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                            Regression Results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class RegressionResult(object):
    irrelevant_params = LogLikelihood.additional_params

    def __init__(self,
                 x_offset_scale=None,
                 x_names=None,
                 y_name=None,
                 orders_and_results=None,
                 predictor=None,
                 ):
        self.x_offset_scale = x_offset_scale
        self.x_names = x_names
        self.y_name = y_name
        self.orders_and_results = orders_and_results
        self.predictor = predictor

        self._map_model_order = self._find_map_model_order()

    def predict_for_map_model(self, x):
        return self.predict_for_model(x, self._map_model_order)

    def predict_for_model(self, x, model_order):
        z = self._normalize_x(x)
        prediction_scaled = self._predict_model_scaled(z, model_order)
        prediction = self._unnormalize_prediction(prediction_scaled)
        return prediction

    def errors_averaged_over_models(self, x):
        z = self._normalize_x(x)
        errors_scaled = self._errors_all_models_scaled(z)
        errors = self._unnormalize_error(errors_scaled)
        return errors

    def errors_for_map_model(self, x):
        z = self._normalize_x(x)
        errors_scaled = self._errors_map_scaled(z)
        errors = self._unnormalize_error(errors_scaled)
        return errors

    def _normalize_x(self, x):
        z = np.copy(x)
        for i in range(z.shape[1]):
            z[:, i] -= self.x_offset_scale[i][0]
            z[:, i] /= self.x_offset_scale[i][1]
        return z

    def _predict_model_scaled(self, z, model_order):
        predictor = self.predictor(model_order)
        parameters = self.orders_and_results[model_order]['result'].x
        coeffs = parameters[self.irrelevant_params:]
        return predictor(z, coeffs)

    def _errors_map_scaled(self, z):
        return self._errors_predictor_scaled(z, self._map_model_order)

    # need to think if this needs to change for generalization
    def _errors_all_models_scaled(self, z):
        model_orders = list(self.orders_and_results.keys())

        log_evidences = np.array([
            self.orders_and_results[o]['log_evidence'] for o in model_orders])
        log_evidences -= log_evidences.max()
        model_probs = np.exp(log_evidences).reshape(-1, 1)
        model_probs /= model_probs.sum()

        model_means = np.array([
            self._predict_model_scaled(z, o) for o in model_orders])
        model_vars = np.array([
            self._errors_predictor_scaled(z, o) for o in model_orders])**2

        variance = (
            np.sum(model_probs * (model_vars + model_means**2), axis=0) -
            np.sum(model_probs * model_means, axis=0)**2)
        return np.sqrt(variance)

    def _errors_predictor_scaled(self, z, model_order):
        # All the predictors are linear models.
        # For a linear model with Gaussian posteriors in the params, the
        # variance of the uncertainty of the prediction is
        #       Var(y) = \sum_ij Cov(a_i, a_j) f_i(x) f_j(x)
        # where a_i are coefficients and f_i(x) the functions.
        # We've stored the coefficients, so we just need f_i f_j:
        this_result = self.orders_and_results[model_order]
        param_covariance = this_result['posterior_covariance']
        totake = slice(self.irrelevant_params, None)
        coeff_covariance = param_covariance[totake, totake]
        ncoeffs = coeff_covariance.shape[0]

        predictor = self.predictor(model_order)
        fi = np.zeros((len(z), ncoeffs))
        for i in range(ncoeffs):
            coeffs1 = np.zeros(ncoeffs); coeffs1[i] = 1
            fi[:, i] = predictor(z, coeffs1)

        variance = np.array([coeff_covariance.dot(f).dot(f) for f in fi])
        return np.sqrt(variance)

    def _unnormalize_prediction(self, y):
        raise NotImplementedError

    def _unnormalize_error(self, errs):
        raise NotImplementedError

    def _find_map_model_order(self):
        orders = list(self.orders_and_results.keys())
        evidences = [self.orders_and_results[o]['log_evidence'] for o in orders]
        best_index = np.argmax(evidences)
        return orders[best_index]

    def __repr__(self):
        out = "<{} for {} vs {} at {}>".format(
            self.__class__.__name__,
            self.y_name,
            self.x_names,
            hex(id(self)))
        return out


class GaussianRegressionResult(RegressionResult):
    irrelevant_params = GaussianLogLikelihood.additional_params

    def __init__(self,
                 x_offset_scale=None,
                 y_offset_scale=None,
                 x_names=None,
                 y_name=None,
                 orders_and_results=None,
                 predictor=None,
                 ):
        """
        Parameters
        ----------
        x_offset_scale : (N, 2)-shaped list-like,
            mean, std of each x. Used for rescaling.
        y_offset_scale : (2,) element list
            mean, std of y. Used for rescaling.
        x_names : (N,) element list of strings.
            Names of the x-variables. Just for storing.
        y_name : string
            Name of the y-variables. Just for storing.
        orders_and_results : dict
            (key: value) pairs of the order of the regression (the key)
            and the associated fit result (the value).
        predictor : callable
            predictor(order).__call__(coefficients, x) should return
            the mean...
        """
        self.y_offset_scale = y_offset_scale
        super(GaussianRegressionResult, self).__init__(
            x_offset_scale=x_offset_scale,
            x_names=x_names,
            y_name=y_name,
            orders_and_results=orders_and_results,
            predictor=predictor,
            )

    def _unnormalize_prediction(self, y):
        return y * self.y_offset_scale[1] + self.y_offset_scale[0]

    def _unnormalize_error(self, errs):
        return errs * self.y_offset_scale[1]


class LogisticRegressionResult(RegressionResult):
    def _unnormalize_prediction(self, logits):
        probabilities = 1.0 / (1 + np.exp(-logits))
        return probabilities

    def errors_for_map_model(self, x):
        z = self._normalize_x(x)
        logit_errors = self._errors_map_scaled(z)
        logits = self._predict_model_scaled(z, self._map_model_order)

        map_prob = self._unnormalize_prediction(logits)
        max_prob = self._unnormalize_prediction(logits + logit_errors)
        min_prob = self._unnormalize_prediction(logits - logit_errors)

        errors = np.array([map_prob - min_prob, max_prob - map_prob])
        return errors

    def errors_averaged_over_models(self, x):
        raise NotImplementedError


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                         regression results maker
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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


def maximize_discrete_relevant(function, initial_guess):
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

                if new_order not in results:
                    f = function(new_order)
                    results.update({tuple(new_order): f})
                    if f > best_f:
                        best_has_updated = True
                        best_f = f
                        best_zn = new_order

    best_value = max(results.values())
    best_x = [x for x in results if results[x] == best_value][0]
    return best_x


class RegressionResultsGetter(object):
    predictor_factory = NoninteractingMultivariatePredictor

    def __init__(self, x_dict, y_dict, regression_type='gaussian',
                 x_offset_scale=None, max_order=10):
        self.x_dict = x_dict
        self.y_dict = y_dict
        self.regression_type = regression_type
        self.max_order = max_order

        self.x_names = list(self.x_dict.keys())
        self.y_name = list(self.y_dict.keys())[0]

        if x_offset_scale is None:
            x_offset_scale = self._find_x_offset_and_scale()
        self.x_offset_scale = x_offset_scale
        self.y_offset_scale = self._find_y_offset_and_scale()
        self.likelihood_class, self.result_class = self._select_classes()
        self._n_variables = len(self.x_names)

    def make_regression_result(self):
        orders_and_results = self._get_orders_and_results()

        kwargs = {
            "x_offset_scale": self._cast_x_offset_scale_to_array(),
            "x_names": self.x_names,
            "y_name": self.y_name,
            "orders_and_results": orders_and_results,
            "predictor": self.predictor_factory,
            }
        if self.regression_type == 'gaussian':
            kwargs.update({"y_offset_scale": self.y_offset_scale})

        result = self.result_class(**kwargs)
        return result

    def _find_x_offset_and_scale(self):
        return {k: (v.mean(), v.std()) for k, v in self.x_dict.items()}

    def _find_y_offset_and_scale(self):
        if 'gaussian' == self.regression_type:
            y = self.y_dict[self.y_name]
            return np.array([y.mean(), y.std()])

    def _cast_x_offset_scale_to_array(self):
        return np.array([self.x_offset_scale[k] for k in self.x_names])

    def _normalize_x(self):
        x = np.transpose([self.x_dict[k] for k in self.x_names])
        for i, k in enumerate(self.x_names):
            x[:, i] -= self.x_offset_scale[k][0]
            x[:, i] /= self.x_offset_scale[k][1]
        return x

    def _normalize_y(self):
        y_raw = list(self.y_dict.values())[0]
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
                logger, initial_guess)
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

