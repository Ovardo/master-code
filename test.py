import lap
import numpy as np

IC = np.random.rand(4, 5)
print(IC)
print(lap.lapjv(IC, extend_cost=True))

fo