import numpy as np

from bayesregress.regress import RegressionResultsGetter


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
    if x.dtype.name == 'object':
        raise ValueError(f'x unrecognized data-type: {x}')

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
    if 'regression_type' not in kwargs:
        if y.dtype.name == 'bool':
            kwargs['regression_type'] = 'bernoulli'
        elif 'float' in y.dtype.name:
            kwargs['regression_type'] = 'gaussian'

    names = {'x_names': x_names, 'y_name': y_name}
    return args, kwargs, names


def set_names(regression_result, names):
    regression_result.x_names = names['x_names']
    regression_result.y_name = names['y_name']
