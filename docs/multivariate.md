```python
import numpy as np
from bayesregress import fit_data
import matplotlib.pyplot as plt

# We set a seed so that this demo is reproducible:
rng = np.random.default_rng(seed=1676)
x1, x2, x3 = rng.standard_normal((3, 1300))
model = (
    100 +
    3 * x1 +
    2 * x2**2 + 0.3 * x2 +
    np.sin(x3))
noise = rng.standard_normal(x1.shape)
y = model + noise
x = {'x1': x1, 'x2': x2, 'x3': x3}

result = fit_data(x, y)
result
>>> <GaussianRegressionResult for 1 variable vs ['x1', 'x2', 'x3'] at 0x7fbd882facf8>

prediction = result.predict_for_map_model(x)
rsquared = np.corrcoef(model, prediction)[0, 1]**2

plt.plot(model, prediction, '.', ms=1)
plt.xlabel("Correct Model")
plt.ylabel("Prediction")
plt.title("Model-vs-Prediction: R^2 = {:0.3f}".format(rsquared))
```
