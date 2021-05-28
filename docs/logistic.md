```python
import numpy as np
from bayesregress import fit_data
import matplotlib.pyplot as plt

# We set a seed so that this demo is reproducible:
rng = np.random.default_rng(seed=1621)
x = np.linspace(-1, 1, 4000)
p = 1 - x**2
y = rng.random(size=x.size) < p

result = fit_data(x, y)

prediction = result.predict_for_map_model(x)
delta = prediction - p

plt.plot(x, y.astype('float'), '.', label="Raw Data", ms=2, zorder=2)
plt.plot(x, p, label='Correct Model', lw=3, zorder=1)
plt.plot(x, prediction, label='Inferred Model', lw=3, zorder=1)
plt.legend(loc='center')
```
