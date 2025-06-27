from data_forming import data_forming
from study_iterations_with_custom import visualize_results, study_iterations_with_custom_sd
from to_csv_stepsize import to_csv_stepsize

def main():
    data_forming(to_csv_stepsize)
    dimensions = [10, 20, 50, 100]  # Размерности
    condition_numbers = [1, 10, 100, 1000, 10000, 100000]
    # Проводим исследование с использованием steepest_descent
    adam_res, cg_res = study_iterations_with_custom_sd(dimensions, condition_numbers)
    # Визуализируем результаты
    visualize_results(dimensions, condition_numbers, adam_res, cg_res)



if __name__ == "__main__":
    main()