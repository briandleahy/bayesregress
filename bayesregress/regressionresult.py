import numpy as np

from bayesregress.likelihood import (
    LogLikelihood,
    GaussianLogLikelihood,
    BernoulliLogLikelihood,
    )


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
        # For a linear model with Gaussian posteriors in the parameters,
        # the variance of the uncertainty of the prediction is
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
        y_name = (self.y_name if self.y_name is not None else "1 variable")
        if self.x_names is None:
            n_x = len(self.x_offset_scale)
            suffix = 's' if n_x > 1 else ''
            x_names = f"{len(self.x_offset_scale)} variable" + suffix
        else:
            x_names = self.x_names
        out = "<{} for {} vs {} at {}>".format(
            self.__class__.__name__,
            y_name,
            x_names,
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
