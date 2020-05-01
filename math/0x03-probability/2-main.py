#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(1., 1000).tolist()
p1 = Poisson(data)
print('F(9):', p1.cdf(1))

p2 = Poisson(lambtha=1)
print('F(9):', p2.cdf(10))
