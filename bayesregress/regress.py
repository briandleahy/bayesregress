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


class ConvergenceError(Exception):
    pass


class HessianIllConditionedError(Exception):
    pass
