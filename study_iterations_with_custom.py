import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from generate_quadratic_function import generate_quadratic_function
from algorythms import adam_method, conjugate_gradient_quadratic
def study_iterations_with_custom_sd(dim_range, cond_range, trials=5, tol=1e-6):
    """
    Исследует зависимость числа итераций с использованием вашего steepest_descent

    Параметры:
    dim_range - диапазон размерностей
    cond_range - диапазон чисел обусловленности
    trials - количество испытаний для усреднения
    tol - точность

    Возвращает:
    adam_results - результаты для метода Адама
    cg_results - результаты для метода сопряженных градиентов
    """

    adam_results = np.zeros((len(dim_range), len(cond_range)))
    cg_results = np.zeros((len(dim_range), len(cond_range)))

    for i, n in enumerate(dim_range):
        for j, cond in enumerate(cond_range):
            adam_total = 0
            cg_total = 0

            for _ in range(trials):
                A, b, f = generate_quadratic_function(n, cond)
                x0 = np.random.randn(n)

                # Используем ваш steepest_descent
                _, iter_adam, _, _, _, _, _, _ = adam_method(f, x0, tol=tol)

                # Используем метод сопряженных градиентов
                _, iter_cg = conjugate_gradient_quadratic(A, b, x0, tol)

                adam_total += iter_adam
                cg_total += iter_cg

            adam_results[i, j] = adam_total / trials
            cg_results[i, j] = cg_total / trials

    return adam_results, cg_results


def visualize_results(dim_range, cond_range, adam_results, cg_results, filename='results/iteration_comparison.png'):
    """
    Визуализирует результаты исследования и сохраняет их в файл.

    Параметры:
    ----------
    dim_range - диапазон размерностей
    cond_range - диапазон чисел обусловленности
    adam_results - результаты для метода Адама
    cg_results - результаты для метода сопряженных градиентов
    filename - путь и имя файла для сохранения графика
    """
    # Создаем сетку для 3D графиков
    D, C = np.meshgrid(dim_range, np.log10(cond_range))

    fig = plt.figure(figsize=(14, 6))

    # График для метода Адама
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(D, C, adam_results.T, cmap='viridis', edgecolor='none')
    ax1.set_title('Метод Адама')
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

    # Автоматически создаём папку results, если её нет
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Сохраняем график в файл
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"График успешно сохранён в файл: {filename}")

