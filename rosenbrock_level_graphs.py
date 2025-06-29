import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_optimization_trajectories(functions, x0, algorithms, step_size=0.01, save_to_file=False):
    """
    Визуализирует траектории оптимизации для всех переданных алгоритмов

    Параметры:
        functions (list): Список тестовых функций (name, func)
        x0 (list): Начальная точка
        algorithms (list): Список алгоритмов оптимизации (name, method)
        step_size (float): Размер шага для методов
        save_to_file (bool): Сохранять ли графики в файл
    """
    # Создаем папку для результатов если нужно сохранять
    if save_to_file:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

    for func_name, func in functions:
        # Пропускаем не функцию Розенброка (f3)
        if func_name != "rosenbrock":
            continue

        try:
            # Запускаем все методы оптимизации
            results = []
            for algo_name, algorithm in algorithms:
                x_opt, n_iter, n_fev, n_gev, converged, _, _, history = algorithm(
                    func, x0, step_size
                )
                results.append({
                    'name': algo_name,
                    'x_opt': x_opt,
                    'n_iter': n_iter,
                    'history': history,
                    'converged': converged
                })

            # Создаем сетку для графика
            x = np.linspace(-1, 1.5, 400)
            y = np.linspace(-1, 2, 400)
            X, Y = np.meshgrid(x, y)
            Z = func([X, Y])

            # Определяем количество строк и столбцов для subplots
            n_methods = len(algorithms)
            n_cols = min(3, n_methods)
            n_rows = (n_methods + n_cols - 1) // n_cols

            # Создаем фигуру с subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
            if n_methods == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            # Цвета для графиков
            colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))

            # Строим графики для каждого метода
            for i, (ax, result) in enumerate(zip(axes, results)):
                traj = np.array(result['history'])
                ax.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis', alpha=0.5)
                ax.scatter(x0[0], x0[1], c='black', marker='o', s=100, label='Начальная точка')
                ax.plot(traj[:, 0], traj[:, 1], '.-', color=colors[i], linewidth=1,
                        markersize=3, label='Траектория')
                ax.scatter(result['x_opt'][0], result['x_opt'][1],
                           color=colors[i], marker='*', s=200, label='Конечная точка')

                status = "Успешно" if result['converged'] else "Не сошелся"
                ax.set_title(f"{result['name']}\nИтераций: {result['n_iter']} ({status})")
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.legend()
                ax.grid(True)

            # Скрываем пустые subplots если есть
            for j in range(len(results), len(axes)):
                axes[j].axis('off')

            # Общий заголовок
            plt.suptitle(f'Траектории оптимизации для функции {func_name}', y=1.02)
            plt.tight_layout()

            if save_to_file:
                filename = results_dir / f"optimization_trajectories_{func_name}.png"
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                print(f"График сохранен в {filename}")
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"Ошибка при построении графиков для {func_name}: {str(e)}")