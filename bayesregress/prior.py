import numpy as np


class LogPrior(object):
    def __call__(self, params):
        raise NotImplementedError


class GaussianLogPrior(LogPrior):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = 0
        if std is None:
            std = 1
        self.mean = mean
        self.std = std

    def __call__(self, params):
        params_z = (params - self.mean) / self.std
        data_term = -0.5 * np.sum(params_z**2)
        # While this is sufficient for finding the maximum a posteriori
        # parameters given the model, it is *not* sufficient for model
        # comparison. For model comparison we need to include the
        # sqrt(1/2pi sigma^2) prefactor
        std = 0 * params + self.std  # cast as array
        prefactor_term = -0.5 * np.log(2 * np.pi * std**2).sum()
        return data_term + prefactor_term


class UniformDistLogitLogPrior(LogPrior):
    def __call__(self, params):
        return np.log(0.25 / np.cosh(params / 2)**2).sum()

