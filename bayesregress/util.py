import numpy as np


def calculate_poisson_binomial_pmf(trial_probabilities):
    n = trial_probabilities.size
    l = np.arange(0, n + 1, 1).reshape(-1, 1)
    p_m = trial_probabilities.reshape(1, -1)

    product_matrix = 1 + (np.exp(2 * np.pi * 1j * l / (n + 1)) - 1) * p_m
    # 9.8 us, above
    product_term = np.prod(product_matrix, axis=1)

    # A fft is slightly faster than doing the matrix evaluation directly,
    # for the small sizes that we care about (n=4):
    pmf = np.fft.fft(product_term).real / (n + 1)
    return pmf
    xc = x.copy()
    np.random.shuffle(xc)
    yc = y.copy()
    np.random.shuffle(yc)
    return np.corrcoef(xc, yc)[0, 1]


def calculate_hessian(function, x, dl=1e-4):
    if dl is 'auto':
        dl = 1e-4 * _estimate_function_scale_from_hessian(function, x)
    gradient = lambda x: _calculate_gradient(function, x, dl=dl)
    # hessian = grad(grad(f)), or
    hessian = _calculate_gradient(gradient, x, dl=dl)
    return hessian


def _calculate_gradient(function, x, dl=1e-4):
    f_x = function(x)
    gradient = np.zeros(x.shape + f_x.shape, dtype=f_x.dtype)
    dx = np.zeros_like(x)
    if np.size(dl) == 1:
        dl = np.full(x.size, dl)
    if dl.shape != x.shape:
        raise ValueError("dl wrong shape; should be ({},)".format(x.size))
    for i in range(x.size):
        dx *= 0
        dx[i] = dl[i]
        plus_2dx = function(x + 2 * dx)
        plus_1dx = function(x + dx)
        minus_1dx = function(x - dx)
        minus_2dx = function(x - 2 * dx)
        numerator = -plus_2dx + 8 * plus_1dx - 8 * minus_1dx + minus_2dx
        gradient[i] = numerator / (12 * dl[i])
    return gradient


def _estimate_function_scale_from_hessian(f, x):
    hess_raw = calculate_hessian(f, x, dl=1e-4)
    return 1.0 / np.sqrt(np.diag(hess_raw))
