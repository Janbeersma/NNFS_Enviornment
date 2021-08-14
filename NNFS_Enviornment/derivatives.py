import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

p2_delta = 0.0001

x1 = 1
x2 = x1 + p2_delta

y1 = f(x1)
y2 = f(x2)

de_geschatte_differentiatie = (y2-y1) / (x2-x1)

print(de_geschatte_differentiatie)

#plt.plot(x, y)
#plt.show()