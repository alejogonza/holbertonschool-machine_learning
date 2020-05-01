#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 9999).tolist()
b = Binomial(data)
print('P(30):', b.pmf(30))
print('F(30):', b.cdf(30))

x = np.arange(0, 50, 1)
y = [b.pmf(x) for x in x]
z = [b.cdf(x) for x in x]

for i in range(len(y)):
    print("x:{:.2f} y:{:.3f}".format(x[i], y[i]))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

plt.title('Binomial Distribution')
ax1.hist(data, x, density=True)
pdf = ax1.plot(x, y, color='red', label='pmf')
cdf = ax2.plot(x, z, color='green', label='cdf')
plt.xticks(np.arange(5, 55, step=5))
plt.xlim(15, 45)
plt.ylim(0, 1)

ax1.set_xlabel('x')
ax1.set_ylabel('pmf')
ax2.set_ylabel('cdf')
plt.show()
