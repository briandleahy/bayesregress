import numpy as np
from scipy.optimize import minimize

from bayesregress.util import calculate_hessian


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


class LogPosterior(object):
    def __init__(self, log_prior, log_likelihood):
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood

    def __call__(self, params):
        return self.log_prior(params) + self.log_likelihood(params)


class Predictor(object):
    """
    Subclasses must implement

    Attributes
    ----------
    self.ncoeffs
    self._nvariables

    Methods
    -------
    self._call(x, coeffs)

    """

    def __call__(self, x, coeffs):
        """
        Parameters
        ----------
        x : (N, d) element numpy array
            where d is the number of variables (i.e. the len of self.order)
        coeffs : (self.ncoeffs,) element numpy array
        """
        if coeffs.shape != (self.ncoeffs,):
            msg = (f"coeffs incorrect shape! is {coeffs.shape}," +
                   f"should be {(self.ncoeffs,)}")
            raise ValueError(msg)
        if x.ndim != 2 or x.shape[1] != self._nvariables:
            msg = (f"x incorrect shape! is {x.shape}, " +
                   f"should be (?, {self._nvariables})")
            raise ValueError(msg)
        return self._call(x, coeffs)


class NoninteractingMultivariatePredictor(Predictor):
    """
    Prediction portion of a generalized nonlinear model for regression.

    This gives non-interacting predictions for mulitple variables,
    i.e. of the form f(x) + h(y) + g(z).
    """
    def __init__(self, order, include_constant=True):
        """
        Parameters
        ----------
        order : tuple of ints
            The degree of the Chebyshev polynomials for each variable.
            Note this is the degree, not number of parameters.
            The total number of parameters is sum(order) + 1, counting
            the constant shift.
        include_constant : bool
            Whether to include a constant term of the polynomial. Default
            is True.
        """
        self.order = order
        self.include_constant = include_constant

        additional_coefficients = (1 if include_constant else 0)
        self.ncoeffs = sum(order) + additional_coefficients
        self._nvariables = len(self.order)

    def _call(self, x, coeffs):
        groups = self._group_coefficients(coeffs)
        separate = [np.polynomial.chebyshev.chebval(x[:, i], groups[i])
                    for i in range(x.shape[1])]
        return np.sum(separate, axis=0)

    def _group_coefficients(self, coeffs):
        group_start = 0
        groups = list()
        for i, o in enumerate(self.order):
            if (i == 0) and self.include_constant:
                blocksize = o + 1
                g = coeffs[group_start: group_start + blocksize]
            else:
                blocksize = o
                g = prepad_with_0(coeffs[group_start: group_start + blocksize])
            group_start += blocksize
            groups.append(g)
        return groups


class BinnedPredictor(Predictor):
    def __init__(self, bin_edges):
        """
        Parameters
        ----------
        bin_edges : numpy.ndarray
            The regions that split the bins. The bins are taken to be
            items in the range
            (-inf, bin_edges[0]]
            (bin_edges[0], bin_edges[1]]
            ...
            (bin_edges[-1], inf)

        """
        self.bin_edges = np.asarray(bin_edges)
        self.ncoeffs = self.bin_edges.size + 1
        self._nvariables = 1
        if self.bin_edges.shape != (self.bin_edges.size,):
            msg = "Only univariate bins currently supported, Brian."
            raise ValueError(msg)
        if not np.all(np.diff(bin_edges) > 0):
            raise ValueError("`bin_edges` must be sorted in increasing order")

    def _call(self, x, coeffs):
        if x.shape[1] != 1:
            msg = "Only univariate bins currently supported, Brian."
            raise ValueError(msg)
        x = x.squeeze()
        bin_lowers = np.hstack([[-np.inf], self.bin_edges])
        bin_uppers = np.hstack([self.bin_edges, [np.inf]])
        out = np.full(x.size, np.nan, dtype='float')
        for i, value in enumerate(coeffs):
            lower_limit = bin_lowers[i]
            upper_limit = bin_uppers[i]
            m = (x > lower_limit) & (x <= upper_limit)
            out[m] = value
        # Checking, should be tested away:
        if np.isnan(out).any():
            raise RuntimeError
        return out


class CompositePredictor(Predictor):
    def __init__(self, predictor1, predictor2):
        self.predictor1 = predictor1
        self.predictor2 = predictor2
        self.ncoeffs = (
            self.predictor1.ncoeffs +
            self.predictor2.ncoeffs)
        self._nvariables = (
            self.predictor1._nvariables +
            self.predictor2._nvariables)

    def _group_coefficients(self, coeffs):
        cutoff = self.predictor1.ncoeffs
        return [coeffs[:cutoff], coeffs[cutoff:]]

    def _call(self, x, coeffs):
        c1, c2 = self._group_coefficients(coeffs)

        x1 = x[:, :self.predictor1._nvariables]
        x2 = x[:, self.predictor1._nvariables:]
        return self.predictor1(x1, c1) + self.predictor2(x2, c2)

    def __repr__(self):
        out = "{}({}, {})".format(
            self.__class__.__name__,
            self.predictor1,
            self.predictor2)
        return out


class CompositePredictorFactory(object):
    def __init__(self, predictor1):
        self.predictor1 = predictor1

    def make_predictor(self, order, **kwargs):
        predictor2 = NoninteractingMultivariatePredictor(order, **kwargs)
        return CompositePredictor(self.predictor1, predictor2)

    def __call__(self, order, **kwargs):
        # allows this to act as a predictor
        return self.make_predictor(order, **kwargs)


class ConvergenceError(Exception):
    pass


class HessianIllConditionedError(Exception):
    pass


def prepad_with_0(x):
    return np.append([0], x)
