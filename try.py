import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/adam_method_results.csv")

# Фильтруем только те случаи, где алгоритм сошёлся
df_converged = df[df["converged"]]

# Построим график зависимости количества итераций от шага
plt.figure(figsize=(10, 6))
for func in df_converged['function_name'].unique():
    df_plot = df_converged[df_converged['function_name'] == func]
    plt.plot(df_plot['step'], df_plot['n_iter'], 'o-', label=func)

plt.xscale('log')
plt.xlabel("Step size")
plt.ylabel("Number of iterations")
plt.title("Adam Method: Iterations vs Step Size")
plt.legend()
plt.grid(True)
plt.show()