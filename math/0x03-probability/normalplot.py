#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
Normal = __import__('normal').Normal


np.random.seed(0)
data = np.random.normal(70, 10, 9999).tolist()
n = Normal(data)
print('PSI(90):', n.pdf(90))
print('PHI(90):', n.cdf(90))


x = np.arange(0, 120, 0.001)
y = [n.pdf(x) for x in x]
z = [n.cdf(x) for x in x]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

plt.title('Normal Distribution')
ax1.hist(data, 60, density=True)
pdf = ax1.plot(x, y, color='red', label='pdf')
cdf = ax2.plot(x, z, color='green', label='cdf')
plt.xticks(np.arange(40, 100, step=10))
plt.xlim(40, 100)
plt.ylim(0, 1)

ax1.set_xlabel('x')
ax1.set_ylabel('pdf')
ax2.set_ylabel('cdf')
plt.show()
