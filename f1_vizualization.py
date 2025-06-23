import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Определение функции
def f(x, y):
    return x**2 + 0.5*x*y + y**2 - 3*x + 4*y + 1

# Диапазон для x и y
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# Создание сетки
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Построение графика
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Настройка графика
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('График функции $f(x, y) = x^2 + 0.5xy + y^2 - 3x + 4y + 1$')
ax.view_init(elev=30, azim=-60)  # Углы обзора

# Добавление контурного графика сверху
ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis')

plt.show()