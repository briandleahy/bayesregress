import numpy as np
from scipy.special import comb


class LogLikelihood(object):
    additional_params = 0

    def __init__(self, predictor=None):
        if predictor is None:
            predictor = np.polynomial.chebyshev.chebval
        self.predictor = predictor

    def __call__(self, params):
        return self.log_likelihood(params)

    def log_likelihood(self, params):
        raise NotImplementedError


class GaussianLogLikelihood(LogLikelihood):
    additional_params = 1

    def __init__(self, x, y, predictor=None):
        self.x = x
        self.y = y
        super(GaussianLogLikelihood, self).__init__(predictor)

    def evaluate_conditional_mean_of_y(self, coeffs):
        return self.predictor(self.x, coeffs)

    def evaluate_residuals(self, coeffs):
        mean_given_x = self.evaluate_conditional_mean_of_y(coeffs)
        return self.y - mean_given_x

    def log_likelihood(self, params):
        log_noise = params[0]
        coeffs = params[1:]
        noise_std = np.exp(log_noise)

        residuals = self.evaluate_residuals(coeffs)
        residuals_zscore = residuals / noise_std
        n_data_points = self.y.size

        data_term = -0.5 * np.sum(residuals_zscore**2)
        noise_term = -n_data_points / 2 * np.log(2 * np.pi * noise_std**2)

        return data_term + noise_term


class BernoulliLogLikelihood(LogLikelihood):
    def __init__(self, x, successes, predictor=None):
        """
        Parameters
        ----------
        x : numpy.ndarray, float
        successes : numpy.ndarray, bool
        predictor : callable, optional
        """
        self.x = x
        self.successes = successes
        self.failures = ~successes
        super(BernoulliLogLikelihood, self).__init__(predictor)

    def log_likelihood(self, params):
        probs = self.calculate_trial_probabilities(params)
        log_likelihood = (np.log(probs[self.successes]).sum() +
                          np.log(1 - probs[self.failures]).sum())
        return log_likelihood

    def calculate_trial_probabilities(self, params):
        logits = self.predictor(self.x, params)
        probs_raw = 1.0 / (1 + np.exp(-logits))
        return np.clip(probs_raw, 1e-10, 1 - 1e-10)


class BinomialLogLikelihood(LogLikelihood):
    def __init__(self, x, trials, successes, predictor=None):
        self.trials = trials
        self.successes = successes
        self.x = x
        super(BinomialLogLikelihood, self).__init__(predictor)

    def log_likelihood(self, params):
        probs = self.calculate_trial_probabilities(params)
        log_likelihood = 0
        for t, s, p in zip(self.trials, self.successes, probs):
            log_likelihood += self._per_event_log_likelihood(t, s, p)
        return log_likelihood

    @classmethod
    def _per_event_log_likelihood(cls, tried, success, prob):
        log_likelihood = (
            np.log(comb(tried, success)) +
            success * np.log(prob) +
            (tried - success) * np.log(1 - prob)
        )
        return log_likelihood

    def calculate_trial_probabilities(self, params):
        logits = self.predictor(self.x, params)
        probs_raw = 1.0 / (1 + np.exp(-logits))
        return np.clip(probs_raw, 1e-10, 1 - 1e-10)

