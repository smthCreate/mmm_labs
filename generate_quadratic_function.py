import numpy as np


def generate_quadratic_function(n, condition_number):
    """
    Генерирует квадратичную функцию f(x) = 0.5*x^T*A*x - b^T*x
    с заданной размерностью и числом обусловленности

    Параметры:
    n - размерность пространства
    condition_number - желаемое число обусловленности матрицы A

    Возвращает:
    A - положительно определенная матрица n×n
    b - вектор n×1
    f - callable функция f(x)
    """
    # Создаем случайную ортогональную матрицу
    H = np.random.randn(n, n)
    Q, _ = np.linalg.qr(H)

    # Генерируем спектр с заданным числом обусловленности
    lambda_max = np.random.uniform(1, 10)
    lambda_min = lambda_max / condition_number
    lambdas = np.linspace(lambda_min, lambda_max, n)

    # Строим матрицу A
    A = Q @ np.diag(lambdas) @ Q.T

    # Генерируем вектор b
    b = np.random.randn(n)

    # Определяем саму функцию
    f = lambda x: 0.5 * x.T @ A @ x - b.T @ x

    return A, b, f