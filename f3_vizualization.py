import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Определение функции Розенброка
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Диапазон для x и y
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)

# Создание сетки
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Построение графика
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Настройка графика
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('График функции Розенброка\n$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$')
ax.view_init(elev=30, azim=-60)  # Углы обзора

# Добавление контурного графика сверху
ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='viridis')

plt.show()