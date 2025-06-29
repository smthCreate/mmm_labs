from data_forming import data_forming
from study_iterations_with_custom import visualize_results, study_iterations_with_custom_sd
from to_csv_stepsize import to_csv_stepsize
from analyze_precision_dependence import load_specific_csv_files, analyze_precision_dependence

def main():
    data_forming(to_csv_stepsize)

    data = load_specific_csv_files()

    # 2. Определяем интересующие нас точности
    precisions = sorted({d['step'] for method_data in data.values() for d in method_data})

    # 3. Анализируем зависимость от точности
    analyze_precision_dependence(data, precisions)

    dimensions = [10, 20, 50, 100]  # Размерности
    condition_numbers = [1, 10, 100, 1000, 10000, 100000]
    # Проводим исследование с использованием steepest_descent
    adam_res, cg_res = study_iterations_with_custom_sd(dimensions, condition_numbers)
    # Визуализируем результаты
    visualize_results(dimensions, condition_numbers, adam_res, cg_res)



if __name__ == "__main__":
    main()