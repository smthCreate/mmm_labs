import time
import csv
import threading
import re
import numpy as np


def to_csv_without_stepsize(functions, precision_levels, x0, optimizer_func, method_name, timeout=60):
    """
    Сохраняет результаты оптимизации в CSV файл без вывода таблиц

    Параметры:
    ----------
    functions : list of tuples
        Список пар (название функции, функция)
    precision_levels : list
        Список уровней точности/шагов для анализа
    x0 : list
        Начальная точка
    optimizer_func : callable
        Функция оптимизации (например, steepest_descent)
    method_name : str
        Название метода оптимизации
    timeout : int
        Максимальное время выполнения (в секундах)
    """
    # Заголовки таблицы
    headers = [
        'precision',
        'min_point',
        'iterations',
        'func_evals',
        'grad_evals',
        'converged',
        'time_sec'
    ]

    # Получаем имя функции градиентного спуска
    func_name = optimizer_func.__name__
    # Очищаем имя от специальных символов
    clean_func_name = re.sub(r'[<>:]', '', func_name)
    # Формируем имя файла
    csv_filename = f"{clean_func_name}_results.csv"

    # Запись заголовков в CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['function_name', 'method_name'] + headers)

    def save_to_csv(func_name, method, row_data):
        """Вспомогательная функция для записи в CSV"""
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([func_name, method] + row_data)

    for func_name, func in functions:
        for precision in precision_levels:
            start_time = time.time()

            try:
                # Запуск оптимизации с таймаутом
                result = None

                def optimize():
                    nonlocal result
                    result = optimizer_func(
                        func, x0,
                        tol=precision,
                        line_search_tol=precision * 0.1
                    )

                thread = threading.Thread(target=optimize)
                thread.start()
                thread.join(timeout=timeout)

                if thread.is_alive():
                    raise TimeoutError()

                # Обработка результатов
                x_opt, n_iter, n_fev, n_gev, converged, divergence, message, history = result
                elapsed = time.time() - start_time

                # Данные для CSV
                csv_row = [
                    f"{precision:.1e}",
                    f"[{x_opt[0]},{x_opt[1]}]" if isinstance(x_opt, (list, np.ndarray)) else str(x_opt),
                    n_iter,
                    n_fev,
                    n_gev,
                    converged,
                    elapsed
                ]

                # Сохранение результатов
                save_to_csv(func_name, method_name, csv_row)

            except TimeoutError:
                elapsed = time.time() - start_time
                save_to_csv(func_name, method_name,
                            [f"{precision:.1e}", "-", "-", "-", "-", False, elapsed])

            except Exception as e:
                elapsed = time.time() - start_time
                save_to_csv(func_name, method_name,
                            [f"{precision:.1e}", "-", "-", "-", "-", False, elapsed])