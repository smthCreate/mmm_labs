import time
import csv
import threading
import re
import os
from pathlib import Path

def to_csv_stepsize(functions, step_sizes, x0, gradient_descent_func, timeout=60):
    """
    Сохраняет результаты градиентного спуска в CSV файл в папке results.
    Создает папку, если она не существует.

    Параметры:
    ----------
    functions : list of tuples
        Список пар (название функции, функция)
    step_sizes : list
        Список размеров шагов
    x0 : list
        Начальная точка
    gradient_descent_func : callable
        Функция градиентного спуска
    timeout : int
        Максимальное время выполнения для одного шага (в секундах)
    """
    headers = [
        'step',
        'min',
        'n_iter',
        'n_fev',
        'n_gev',
        'converged',
        'time_sec'
    ]

    # Создаем папку results, если ее нет
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Получаем имя функции градиентного спуска
    method_name = gradient_descent_func.__name__
    # Очищаем имя от специальных символов
    clean_func_name = re.sub(r'[<>:]', '', method_name)
    # Формируем путь к файлу
    csv_filename = results_dir / f"{clean_func_name}_results.csv"

    # Создаем CSV файл и записываем заголовки
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['function_name', 'method_name'] + headers)

    # Функция для записи строки в CSV
    def write_to_csv(func_name, method_name, row_data):
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([func_name, method_name] + row_data)

    for func_name, func in functions:
        for step in step_sizes:
            start_time = time.time()

            try:
                # Запускаем с таймаутом
                result = None

                def worker():
                    nonlocal result
                    result = gradient_descent_func(func, x0, step)

                t = threading.Thread(target=worker)
                t.start()
                t.join(timeout=timeout)

                elapsed_time = time.time() - start_time

                if t.is_alive():
                    raise TimeoutError()

                x_opt, n_iter, n_fev, n_gev, converged, divergence, message, history = result

                # Формируем данные для CSV
                csv_row = [
                    f"{step:.3f}",
                    f"{x_opt[0]},{x_opt[1]}",
                    n_iter,
                    n_fev,
                    n_gev,
                    converged,
                    elapsed_time
                ]

                # Записываем в CSV
                write_to_csv(func_name, method_name, csv_row)

            except TimeoutError:
                elapsed_time = time.time() - start_time
                write_to_csv(func_name, method_name,
                    [f"{step:.3f}", "-", "-", "-", "-", "-", elapsed_time])
            except Exception as e:
                elapsed_time = time.time() - start_time
                write_to_csv(func_name, method_name,
                    [f"{step:.3f}", "-", "-", "-", "-", "-", elapsed_time])