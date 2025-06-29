from algorythms import (
    momentum_method,nesterov_method,adagrad_method,adadelta_method,rmsprop_method,adam_method)
from to_csv_stepsize import to_csv_stepsize
from analyze_precision_dependence import analyze_precision_dependence
from rosenbrock_level_graphs import plot_optimization_trajectories

def data_forming(to_csv_stepsize):
    # x**2 + 0.5*x*y + y**2 - 3*x + 4*y + 1 u=1
    def f1(x):
        return x[0] ** 2 + 0.5 * x[0] * x[1] + x[1] ** 2 - 3 * x[0] + 4 * x[1] + 1

    # x**2 + 100 * y**2 - 3*x + 4*y + 1 u>10
    def f2(x):
        return x[0] ** 2 + 100 * x[1] ** 2 - 3 * x[0] + 4 * x[1] + 1

    # (1 - x)**2 + 100 * (y - x**2)**2
    def f3(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    x0 = [0.0, 0.0]
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    step_size = 0.01
    functions = [
        ("quadratic", f1),
        ("strong_convex", f2),
        ("rosenbrock", f3)
    ]

    algorithms = [
        ("Momentum", momentum_method),
        ("Nesterov", nesterov_method),
        ("Adagrad", adagrad_method),
        ("Adadelta", adadelta_method),
        ("RMSprop", rmsprop_method),
        ("Adam", adam_method)
    ]

    for algo_name, algorithm in algorithms:
        try:
            to_csv_stepsize(
                functions=functions,
                step_sizes=step_sizes,
                x0=x0,
                gradient_descent_func=algorithm
            )
        except Exception as e:
            print(f"Ошибка при выполнении {algo_name}: {str(e)}")



    plot_optimization_trajectories(
        functions=functions,
        x0=x0,
        algorithms=algorithms,
        save_to_file=True
    )
if __name__ == "__main__":
    data_forming(to_csv_stepsize)