#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(4., 999).tolist()
print(data.count(10))
print(data)

p = Poisson(data)
print('p--> P(9):', p.pmf(9))
print('F(9):', p.cdf(1))

x = np.arange(0, 12, 1)
y = [p.pmf(x) for x in x]
z = [p.cdf(x) for x in x]

for i in range(len(y)):
    print("x:{:.2f} y:{:.3f}".format(x[i], y[i]))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

plt.title('Poisson Distribution')
ax1.hist(data, x, density=True)
pdf = ax1.plot(x, y, color='red', label='pmf')
cdf = ax2.plot(x, z, color='green', label='cdf')
plt.xticks(np.arange(0, 13, step=1))
plt.xlim(0, 11)
plt.ylim(0, 1)

ax1.set_xlabel('x')
ax1.set_ylabel('pmf')
ax2.set_ylabel('cdf')
plt.show()
