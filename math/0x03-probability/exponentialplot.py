#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
Exponential = __import__('exponential').Exponential


np.random.seed(0)
data = np.random.exponential(0.5, 9999).tolist()
e = Exponential(data)
print('f(1):', e.pdf(1))
print('F(1):', e.cdf(1))

x = np.arange(0, 15, 0.001)
y = [e.pdf(x) for x in x]
z = [e.cdf(x) for x in x]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

plt.title('Exponential Distribution')
ax1.hist(data, 60, density=True)
pdf = ax1.plot(x, y, color='red', label='pdf')
cdf = ax2.plot(x, z, color='green', label='cdf')
plt.xticks(np.arange(0, 3.5, step=0.5))
plt.xlim(0, 3)
plt.ylim(0, 1)

ax1.set_xlabel('x')
ax1.set_ylabel('pdf')
ax2.set_ylabel('cdf')
plt.show()
