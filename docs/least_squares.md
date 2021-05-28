```python
import numpy as np
from bayesregress import fit_data
import matplotlib.pyplot as plt

# We set a seed so that this demo is reproducible:
rng = np.random.default_rng(seed=1516)

x = np.linspace(-10, 10, 901)
noise = rng.standard_normal(x.shape)
correct = 0.5 * x * np.cos(0.5 * x)
y = correct + noise

result = fit_data(x, y)

prediction = result.predict_for_map_model(x)

plt.plot(x, y, '.', label="Raw Data", ms=2)
plt.plot(x, correct, label='Correct Model', lw=3)
plt.plot(x, prediction, label='Inferred Model', lw=3)
plt.legend(loc='lower left')
```
