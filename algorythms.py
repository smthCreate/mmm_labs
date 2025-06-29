import numpy as np

def momentum_method(f, x0, step_size=0.1, max_iter=1000, tol=1e-6,
                   divergence_threshold=1e10, momentum=0.9, h=1e-5):
    """
    Градиентный спуск с моментом (Momentum)

    Параметры:
    ----------
    f : callable
        Функция, которую минимизируем.
    x0 : list или np.array
        Начальная точка.
    step_size : float, optional
        Размер шага (по умолчанию 0.1).
    max_iter : int, optional
        Максимальное число итераций (по умолчанию 1000).
    tol : float, optional
        Точность остановки (по умолчанию 1e-6).
    divergence_threshold : float, optional
        Порог для определения расходимости (по умолчанию 1e10).
    momentum : float, optional
        Параметр момента (по умолчанию 0.9).
    h : float, optional
        Шаг для численного градиента (по умолчанию 1e-5).

    Возвращает:
    -----------
    x : np.array
        Найденная точка минимума.
    n_iter : int
        Количество выполненных итераций.
    n_fev : int
        Количество вычислений функции.
    n_gev : int
        Количество вычислений градиента.
    converged : bool
        Флаг сходимости алгоритма.
    divergence : bool
        Флаг расходимости алгоритма.
    message : str
        Сообщение о результате работы.
    history : list
        История всех точек x на каждой итерации.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    velocity = np.zeros_like(x)  # Инициализация "скорости"
    n_iter = 0
    n_fev = 0
    n_gev = 0
    converged = False
    divergence = False
    message = "Good"

    # Начальное значение функции
    prev_f_value = f(x)
    n_fev += 1

    for _ in range(max_iter):
        # Численный градиент
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            n_fev += 2
        n_gev += 1

        # Обновление скорости (с моментом)
        velocity = momentum * velocity - step_size * grad

        # Обновление параметров
        x_new = x + velocity

        # Текущее значение функции
        current_f_value = f(x_new)
        n_fev += 1

        # Проверка на расходимость
        if (abs(current_f_value) > divergence_threshold or
            np.any(np.isnan(x_new)) or
            np.any(np.isinf(x_new))):
            divergence = True
            message = f"Предупреждение: Алгоритм расходится на итерации {n_iter}!"
            break

        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < tol:
            converged = True
            break

        x = x_new
        history.append(x.copy())
        n_iter += 1
        prev_f_value = current_f_value

    if not converged and not divergence and n_iter == max_iter:
        message = f"Предупреждение: Достигнуто максимальное число итераций {max_iter}"

    return x, n_iter, n_fev, n_gev, converged, divergence, message, history

def nesterov_method(f, x0, step_size=0.1, max_iter=1000, tol=1e-6,
                   divergence_threshold=1e10, momentum=0.9, h=1e-5):
    """
    Градиентный спуск с ускорением Нестерова (Nesterov Accelerated Gradient)

    Параметры:
    ----------
    f : callable
        Функция, которую минимизируем.
    x0 : list или np.array
        Начальная точка.
    step_size : float, optional
        Размер шага (по умолчанию 0.1).
    max_iter : int, optional
        Максимальное число итераций (по умолчанию 1000).
    tol : float, optional
        Точность остановки (по умолчанию 1e-6).
    divergence_threshold : float, optional
        Порог для определения расходимости (по умолчанию 1e10).
    momentum : float, optional
        Параметр момента (по умолчанию 0.9).
    h : float, optional
        Шаг для численного градиента (по умолчанию 1e-5).

    Возвращает:
    -----------
    x : np.array
        Найденная точка минимума.
    n_iter : int
        Количество выполненных итераций.
    n_fev : int
        Количество вычислений функции.
    n_gev : int
        Количество вычислений градиента.
    converged : bool
        Флаг сходимости алгоритма.
    divergence : bool
        Флаг расходимости алгоритма.
    message : str
        Сообщение о результате работы.
    history : list
        История всех точек x на каждой итерации.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    velocity = np.zeros_like(x)  # Инициализация "скорости"
    n_iter = 0
    n_fev = 0
    n_gev = 0
    converged = False
    divergence = False
    message = "Good"

    # Начальное значение функции
    prev_f_value = f(x)
    n_fev += 1

    for _ in range(max_iter):
        # Промежуточная точка (lookahead)
        x_ahead = x + momentum * velocity

        # Численный градиент в промежуточной точке
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x_ahead.copy()
            x_plus[i] += h
            x_minus = x_ahead.copy()
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            n_fev += 2
        n_gev += 1

        # Обновление скорости
        velocity = momentum * velocity - step_size * grad

        # Обновление параметров
        x_new = x + velocity

        # Текущее значение функции
        current_f_value = f(x_new)
        n_fev += 1

        # Проверка на расходимость
        if (abs(current_f_value) > divergence_threshold or
            np.any(np.isnan(x_new)) or
            np.any(np.isinf(x_new))):
            divergence = True
            message = f"Предупреждение: Алгоритм расходится на итерации {n_iter}!"
            break

        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < tol:
            converged = True
            break

        x = x_new
        history.append(x.copy())
        n_iter += 1
        prev_f_value = current_f_value

    if not converged and not divergence and n_iter == max_iter:
        message = f"Предупреждение: Достигнуто максимальное число итераций {max_iter}"

    return x, n_iter, n_fev, n_gev, converged, divergence, message, history

def adagrad_method(f, x0, step_size=0.1, max_iter=1000, tol=1e-6,
                  divergence_threshold=1e10, epsilon=1e-8, h=1e-5):
    """
    AdaGrad (Adaptive Gradient Algorithm) с численным градиентом

    Параметры:
    ----------
    f : callable
        Функция, которую минимизируем.
    x0 : list или np.array
        Начальная точка.
    step_size : float, optional
        Начальный размер шага (по умолчанию 0.1).
    max_iter : int, optional
        Максимальное число итераций (по умолчанию 1000).
    tol : float, optional
        Точность остановки (по умолчанию 1e-6).
    divergence_threshold : float, optional
        Порог для определения расходимости (по умолчанию 1e10).
    epsilon : float, optional
        Малая константа для численной стабильности (по умолчанию 1e-8).
    h : float, optional
        Шаг для численного градиента (по умолчанию 1e-5).

    Возвращает:
    -----------
    x : np.array
        Найденная точка минимума.
    n_iter : int
        Количество выполненных итераций.
    n_fev : int
        Количество вычислений функции.
    n_gev : int
        Количество вычислений градиента.
    converged : bool
        Флаг сходимости алгоритма.
    divergence : bool
        Флаг расходимости алгоритма.
    message : str
        Сообщение о результате работы.
    history : list
        История всех точек x на каждой итерации.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    G = np.zeros_like(x)  # Сумма квадратов градиентов
    n_iter = 0
    n_fev = 0
    n_gev = 0
    converged = False
    divergence = False
    message = "Good"

    # Начальное значение функции
    prev_f_value = f(x)
    n_fev += 1

    for _ in range(max_iter):
        # Численный градиент
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            n_fev += 2
        n_gev += 1

        # Накопление суммы квадратов градиентов
        G += grad**2

        # Адаптивное обновление параметров
        adaptive_step = step_size / (np.sqrt(G) + epsilon)
        x_new = x - adaptive_step * grad

        # Текущее значение функции
        current_f_value = f(x_new)
        n_fev += 1

        # Проверка на расходимость
        if (abs(current_f_value) > divergence_threshold or
            np.any(np.isnan(x_new)) or
            np.any(np.isinf(x_new))):
            divergence = True
            message = f"Предупреждение: Алгоритм расходится на итерации {n_iter}!"
            break

        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < tol:
            converged = True
            break

        x = x_new
        history.append(x.copy())
        n_iter += 1
        prev_f_value = current_f_value

    if not converged and not divergence and n_iter == max_iter:
        message = f"Предупреждение: Достигнуто максимальное число итераций {max_iter}"

    return x, n_iter, n_fev, n_gev, converged, divergence, message, history

def adadelta_method(f, x0, step_size=1.0, max_iter=1000, tol=1e-6,
                   divergence_threshold=1e10, rho=0.95, epsilon=1e-6, h=1e-5):
    """
    AdaDelta (Adaptive Delta) метод с численным градиентом
    Параметры:
    ----------
    f : callable
        Функция, которую минимизируем.
    x0 : list или np.array
        Начальная точка.
    step_size : float, optional
        Начальный размер шага (по умолчанию 1.0, в AdaDelta обычно не используется)
    max_iter : int, optional
        Максимальное число итераций (по умолчанию 1000).
    tol : float, optional
        Точность остановки (по умолчанию 1e-6).
    divergence_threshold : float, optional
        Порог для определения расходимости (по умолчанию 1e10).
    rho : float, optional
        Коэффициент забывания (по умолчанию 0.95).
    epsilon : float, optional
        Малая константа для численной стабильности (по умолчанию 1e-6).
    h : float, optional
        Шаг для численного градиента (по умолчанию 1e-5).
    Возвращает:
    -----------
    x : np.array
        Найденная точка минимума.
    n_iter : int
        Количество выполненных итераций.
    n_fev : int
        Количество вычислений функции.
    n_gev : int
        Количество вычислений градиента.
    converged : bool
        Флаг сходимости алгоритма.
    divergence : bool
        Флаг расходимости алгоритма.
    message : str
        Сообщение о результате работы.
    history : list
        История всех точек x на каждой итерации.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    E_g2 = np.zeros_like(x)  # Бегущее среднее квадратов градиентов
    E_dx2 = np.zeros_like(x)  # Бегущее среднее квадратов изменений параметров
    n_iter = 0
    n_fev = 0
    n_gev = 0
    converged = False
    divergence = False
    message = "Good"
    # Начальное значение функции
    prev_f_value = f(x)
    n_fev += 1

    for _ in range(max_iter):
        # Численный градиент
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            n_fev += 2
        n_gev += 1

        # Обновление бегущего среднего квадратов градиентов
        E_g2 = rho * E_g2 + (1 - rho) * grad**2

        # Вычисление RMS (root mean square) для градиентов и изменений параметров
        RMS_g = np.sqrt(E_g2 + epsilon)
        RMS_dx = np.sqrt(E_dx2 + epsilon)

        # Вычисление адаптивного шага
        delta_x = - (RMS_dx / RMS_g) * grad

        # Обновление параметров
        x_new = x + delta_x

        # Обновление бегущего среднего квадратов изменений параметров
        E_dx2 = rho * E_dx2 + (1 - rho) * delta_x**2

        # Текущее значение функции
        current_f_value = f(x_new)
        n_fev += 1

        # Проверка на расходимость
        if abs(current_f_value) > divergence_threshold or np.any(np.isnan(x_new)) or np.any(np.isinf(x_new)):
            divergence = True
            message = f"Предупреждение: Алгоритм расходится на итерации {n_iter}!"
            break

        # Проверка на сходимость по изменению x
        if np.linalg.norm(x_new - x) < tol:
            converged = True
            break

        x = x_new
        history.append(x.copy())
        n_iter += 1
        prev_f_value = current_f_value

    if not converged and not divergence and n_iter == max_iter - 1:
        message = f"Предупреждение: Достигнуто максимальное число итераций {max_iter}"

    return x, n_iter, n_fev, n_gev, converged, divergence, message, history

def rmsprop_method(f, x0, step_size=0.01, max_iter=1000, tol=1e-6,
                  divergence_threshold=1e10, rho=0.9, epsilon=1e-6, h=1e-5):
    """
    RMSProp (Root Mean Square Propagation) с численным градиентом

    Параметры:
    ----------
    f : callable
        Функция, которую минимизируем.
    x0 : list или np.array
        Начальная точка.
    step_size : float, optional
        Начальный размер шага (по умолчанию 0.01).
    max_iter : int, optional
        Максимальное число итераций (по умолчанию 1000).
    tol : float, optional
        Точность остановки (по умолчанию 1e-6).
    divergence_threshold : float, optional
        Порог для определения расходимости (по умолчанию 1e10).
    rho : float, optional
        Коэффициент забывания (по умолчанию 0.9).
    epsilon : float, optional
        Малая константа для численной стабильности (по умолчанию 1e-6).
    h : float, optional
        Шаг для численного градиента (по умолчанию 1e-5).

    Возвращает:
    -----------
    x : np.array
        Найденная точка минимума.
    n_iter : int
        Количество выполненных итераций.
    n_fev : int
        Количество вычислений функции.
    n_gev : int
        Количество вычислений градиента.
    converged : bool
        Флаг сходимости алгоритма.
    divergence : bool
        Флаг расходимости алгоритма.
    message : str
        Сообщение о результате работы.
    history : list
        История всех точек x на каждой итерации.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    E_g2 = np.zeros_like(x)  # Скользящее среднее квадратов градиентов
    n_iter = 0
    n_fev = 0
    n_gev = 0
    converged = False
    divergence = False
    message = "Good"

    # Начальное значение функции
    prev_f_value = f(x)
    n_fev += 1

    for _ in range(max_iter):
        # Численный градиент
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            n_fev += 2
        n_gev += 1

        # Обновление скользящего среднего квадратов градиентов
        E_g2 = rho * E_g2 + (1 - rho) * grad**2

        # Вычисление адаптивного шага
        adaptive_step = step_size / (np.sqrt(E_g2) + epsilon)
        x_new = x - adaptive_step * grad

        # Текущее значение функции
        current_f_value = f(x_new)
        n_fev += 1

        # Проверка на расходимость
        if (abs(current_f_value) > divergence_threshold or
            np.any(np.isnan(x_new)) or
            np.any(np.isinf(x_new))):
            divergence = True
            message = f"Предупреждение: Алгоритм расходится на итерации {n_iter}!"
            break

        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < tol:
            converged = True
            break

        x = x_new
        history.append(x.copy())
        n_iter += 1
        prev_f_value = current_f_value

    if not converged and not divergence and n_iter == max_iter:
        message = f"Предупреждение: Достигнуто максимальное число итераций {max_iter}"

    return x, n_iter, n_fev, n_gev, converged, divergence, message, history

def adam_method(f, x0, step_size=0.001, max_iter=1000, tol=1e-6,
               divergence_threshold=1e10, beta1=0.9, beta2=0.999,
               epsilon=1e-8, h=1e-5):
    """
    Adam (Adaptive Moment Estimation) с численным градиентом

    Параметры:
    ----------
    f : callable
        Функция, которую минимизируем.
    x0 : list или np.array
        Начальная точка.
    step_size : float, optional
        Размер шага (по умолчанию 0.001).
    max_iter : int, optional
        Максимальное число итераций (по умолчанию 1000).
    tol : float, optional
        Точность остановки (по умолчанию 1e-6).
    divergence_threshold : float, optional
        Порог для определения расходимости (по умолчанию 1e10).
    beta1 : float, optional
        Коэффициент для оценки первого момента (по умолчанию 0.9).
    beta2 : float, optional
        Коэффициент для оценки второго момента (по умолчанию 0.999).
    epsilon : float, optional
        Малая константа для численной стабильности (по умолчанию 1e-8).
    h : float, optional
        Шаг для численного градиента (по умолчанию 1e-5).

    Возвращает:
    -----------
    x : np.array
        Найденная точка минимума.
    n_iter : int
        Количество выполненных итераций.
    n_fev : int
        Количество вычислений функции.
    n_gev : int
        Количество вычислений градиента.
    converged : bool
        Флаг сходимости алгоритма.
    divergence : bool
        Флаг расходимости алгоритма.
    message : str
        Сообщение о результате работы.
    history : list
        История всех точек x на каждой итерации.
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    m = np.zeros_like(x)  # Первый момент (среднее)
    v = np.zeros_like(x)  # Второй момент (нецентрированная дисперсия)
    n_iter = 0
    n_fev = 0
    n_gev = 0
    converged = False
    divergence = False
    message = "Good"

    # Начальное значение функции
    prev_f_value = f(x)
    n_fev += 1

    for t in range(1, max_iter + 1):
        # Численный градиент
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            n_fev += 2
        n_gev += 1

        # Обновление оценок первых и вторых моментов
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Коррекция смещения (bias correction)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Обновление параметров
        x_new = x - step_size * m_hat / (np.sqrt(v_hat) + epsilon)

        # Текущее значение функции
        current_f_value = f(x_new)
        n_fev += 1

        # Проверка на расходимость
        if (abs(current_f_value) > divergence_threshold or
            np.any(np.isnan(x_new)) or
            np.any(np.isinf(x_new))):
            divergence = True
            message = f"Предупреждение: Алгоритм расходится на итерации {n_iter}!"
            break

        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < tol:
            converged = True
            break

        x = x_new
        history.append(x.copy())
        n_iter = t
        prev_f_value = current_f_value

    if not converged and not divergence and n_iter == max_iter:
        message = f"Предупреждение: Достигнуто максимальное число итераций {max_iter}"

    return x, n_iter, n_fev, n_gev, converged, divergence, message, history
def conjugate_gradient_quadratic(A, b, x0, tol=1e-6, max_iter=None):
    """
    Метод сопряженных градиентов для квадратичной функции

    Параметры:
    A, b - параметры квадратичной функции
    x0 - начальная точка
    tol - точность
    max_iter - максимальное число итераций

    Возвращает:
    x - решение
    iterations - число итераций
    """
    if max_iter is None:
        max_iter = len(b)

    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    iterations = 0

    for _ in range(max_iter):
        iterations += 1
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap

        if np.linalg.norm(r_new) < tol:
            break

        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new

    return x, iterations
