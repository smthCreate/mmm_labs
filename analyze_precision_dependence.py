import csv
import matplotlib.pyplot as plt
from tabulate import tabulate
import csv
import numpy as np
from pathlib import Path

def load_specific_csv_files():
    """
    Загружает данные из CSV файлов методов оптимизации из папки results.
    Фильтрует только те строки, где converged=True.
    Обрабатывает файлы: adagrad, adam, momentum, nesterov, rmsprop, adadelta.

    Возвращает:
        dict: Словарь с данными, где ключи - названия методов,
              значения - списки словарей с параметрами оптимизации (только сошедшиеся)
    """
    data = {}
    results_dir = Path("results")

    if not results_dir.exists():
        print(f"Warning: Directory '{results_dir}' not found")
        return data

    # Список ожидаемых файлов методов оптимизации
    optimization_methods = [
        "adagrad", "adam", "momentum",
        "nesterov", "rmsprop", "adadelta"
    ]

    # Обрабатываем каждый файл метода
    for method in optimization_methods:
        filename = f"{method}_method_results.csv"
        filepath = results_dir / filename

        if not filepath.exists():
            print(f"Warning: File '{filename}' not found")
            continue

        data[method] = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Пропускаем строки, где нет сходимости
                    if row['converged'] != 'True':
                        continue

                    # Разбираем координаты минимума
                    min_coords = tuple(map(float, row['min'].split(',')))

                    processed_row = {
                        'function': row['function_name'],
                        'method': row['method_name'],
                        'step': float(row['step']),
                        'min_x': min_coords[0],
                        'min_y': min_coords[1],
                        'n_iter': safe_int(row['n_iter']),
                        'n_fev': safe_int(row['n_fev']),
                        'n_gev': safe_int(row['n_gev']),
                        'converged': True,
                        'time': safe_float(row['time_sec'])
                    }
                    data[method].append(processed_row)
                except Exception as e:
                    print(f"Error processing row in {filename}: {e}")

    return data


def safe_int(value):
    """Безопасное преобразование в int с обработкой нечисловых значений"""
    if isinstance(value, str) and not value.strip('-').isdigit():
        return np.nan
    try:
        return int(value)
    except (ValueError, TypeError):
        return np.nan


def safe_float(value):
    """Безопасное преобразование в float с обработкой нечисловых значений"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def analyze_precision_dependence(data, precisions=None):
    """Анализирует зависимость от точности и выводит таблицы с графиками"""
    if precisions is None:
        precisions = sorted({d['step'] for method_data in data.values() for d in method_data})

    # Группируем данные по функциям и методам
    results = {}
    for func_name in {d['function'] for method_data in data.values() for d in method_data}:
        results[func_name] = {}

        for method_name in data.keys():
            # Проверяем, есть ли данные для данного метода и функции
            method_data = [d for d in data[method_name] if d['function'] == func_name]

            if not method_data:
                print(f"Предупреждение: Нет данных для метода '{method_name}' и функции '{func_name}'")
                continue

            results[func_name][method_name] = {
                'precision': [],
                'n_iter': [],
                'n_fev': [],
                'n_gev': []
            }

            # Собираем данные для текущей функции и метода
            for precision in precisions:
                # Находим запись с наиболее близким шагом к требуемой точности
                closest = min(method_data, key=lambda x: abs(x['step'] - precision))

                results[func_name][method_name]['precision'].append(precision)
                results[func_name][method_name]['n_iter'].append(closest['n_iter'])
                results[func_name][method_name]['n_fev'].append(closest['n_fev'])
                results[func_name][method_name]['n_gev'].append(closest['n_gev'])

    # Выводим таблицы и графики для каждой функции
    for func_name, func_data in results.items():
        print(f"\n\nАнализ зависимости от точности для функции: {func_name}")
        print("="*80)

        # print_tables(func_data, precisions)

        # Строим графики
        plot_precision_dependence(func_data, func_name)

def print_tables(func_data, precisions):
    """Печатает таблицы с результатами"""
    metrics = {
        'n_iter': 'Количество итераций',
        'n_fev': 'Количество вычислений функции',
        'n_gev': 'Количество вычислений градиента'
    }

    for metric, title in metrics.items():
        print(f"\n{title} в зависимости от точности")

        # Подготавливаем данные для таблицы
        table_data = []
        headers = ['Точность'] + list(func_data.keys())

        for i, precision in enumerate(precisions):
            row = [f"{precision:.1e}"]
            for method in func_data.keys():
                value = func_data[method][metric][i]
                row.append(f"{value:.0f}" if not np.isnan(value) else "-")
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt='grid'))

def plot_precision_dependence(func_data, func_name):
    """Строит графики зависимости от точности"""
    metrics = {
        'n_iter': 'Количество итераций',
        'n_fev': 'Количество вычислений функции',
        'n_gev': 'Количество вычислений градиента'
    }

    plt.figure(figsize=(15, 10))

    for i, (metric, title) in enumerate(metrics.items(), 1):
        plt.subplot(3, 1, i)

        for method in func_data.keys():
            x = func_data[method]['precision']
            y = func_data[method][metric]

            # Фильтруем NaN значения
            x_clean = []
            y_clean = []
            for x_val, y_val in zip(x, y):
                if not np.isnan(y_val):
                    x_clean.append(x_val)
                    y_clean.append(y_val)

            if x_clean and y_clean:
                plt.plot(x_clean, y_clean, 'o-', label=method)

        plt.xscale('log')
        plt.gca().invert_xaxis()  # Чтобы более высокие точности (меньшие значения) были справа
        plt.xlabel('Точность (log scale)')
        plt.ylabel(title)
        plt.title(f'{title} для функции {func_name}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    # Создаём папку results если её нет
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Сохраняем графики в папку results
    filename = results_dir / f"precision_dependence_{func_name}.png"
    plt.savefig(filename)
    print(f"\nГрафики сохранены в файл: {filename}")
    plt.close()

if __name__ == "__main__":
    # 1. Загружаем данные из конкретных CSV файлов
    data = load_specific_csv_files()

    # 2. Определяем интересующие нас точности
    precisions = sorted({d['step'] for method_data in data.values() for d in method_data})

    # 3. Анализируем зависимость от точности
    analyze_precision_dependence(data, precisions)