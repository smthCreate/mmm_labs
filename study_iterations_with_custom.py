import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def study_iterations_with_custom_sd(dim_range, cond_range, trials=5, tol=1e-6):
    """
    Исследует зависимость числа итераций с использованием вашего steepest_descent

    Параметры:
    dim_range - диапазон размерностей
    cond_range - диапазон чисел обусловленности
    trials - количество испытаний для усреднения
    tol - точность

    Возвращает:
    sd_results - результаты для метода наискорейшего спуска
    cg_results - результаты для метода сопряженных градиентов
    """
    sd_results = np.zeros((len(dim_range), len(cond_range)))
    cg_results = np.zeros((len(dim_range), len(cond_range)))

    for i, n in enumerate(dim_range):
        for j, cond in enumerate(cond_range):
            sd_total = 0
            cg_total = 0

            for _ in range(trials):
                A, b, f = generate_quadratic_function(n, cond)
                x0 = np.random.randn(n)

                # Используем ваш steepest_descent
                _, iter_sd, _, _, _, _, _, _ = steepest_descent(f, x0, tol=tol)

                # Используем метод сопряженных градиентов
                _, iter_cg = conjugate_gradient_quadratic(A, b, x0, tol)

                sd_total += iter_sd
                cg_total += iter_cg

            sd_results[i, j] = sd_total / trials
            cg_results[i, j] = cg_total / trials

    return sd_results, cg_results

def visualize_results(dim_range, cond_range, sd_results, cg_results):
    """
    Визуализирует результаты исследования

    Параметры:
    dim_range - диапазон размерностей
    cond_range - диапазон чисел обусловленности
    sd_results - результаты для SD
    cg_results - результаты для CG
    """
    # Создаем сетку для 3D графиков
    D, C = np.meshgrid(dim_range, np.log10(cond_range))

    fig = plt.figure(figsize=(14, 6))

    # График для метода наискорейшего спуска
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(D, C, sd_results.T, cmap='viridis', edgecolor='none')
    ax1.set_title('Метод наискорейшего спуска (ваша реализация)')
    ax1.set_xlabel('Размерность')
    ax1.set_ylabel('log10(Число обусловленности)')
    ax1.set_zlabel('Число итераций')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # График для метода сопряженных градиентов
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(D, C, cg_results.T, cmap='plasma', edgecolor='none')
    ax2.set_title('Метод сопряженных градиентов')
    ax2.set_xlabel('Размерность')
    ax2.set_ylabel('log10(Число обусловленности)')
    ax2.set_zlabel('Число итераций')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()

# Параметры исследования
dimensions = [10, 20, 50, 100]  # Размерности
condition_numbers = [1,4,7,10,15]  # Числа обусловленности 10^1..10^5

# Проводим исследование с использованием вашего steepest_descent
sd_res, cg_res = study_iterations_with_custom_sd(dimensions, condition_numbers)

# Визуализируем результаты
visualize_results(dimensions, condition_numbers, sd_res, cg_res)