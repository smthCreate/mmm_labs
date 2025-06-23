import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return x**2 + 100 * y**2 - 3*x + 4*y + 1

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('График плохо обусловленной функции\n$f(x, y) = x^2 + 100y^2 - 3x + 4y + 1$')
ax.view_init(elev=30, azim=-60)
ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis')
plt.show()