import numpy as np
import matplotlib.pyplot as plt
import timeit

# Define the sinc function
def sinc(x):
    return np.sin(x) / x if x != 0 else 1

# Plain Python implementation of the function
def func_py(x: list[float], N: int) -> list[float]:
    return [sinc(2 * t / N - 1) for t in x]

# NumPy implementation of the function
def func_numpy(x: np.ndarray, N: int) -> np.ndarray:
    x_transformed = 2 * x / N - 1
    return np.sinc(x_transformed / np.pi)  # np.sinc(x) is normalized sinc (sin(πx)/(πx))

# Plain Python tabulation
def tabulate_py(a: float, b: float, n: int, N: int) -> tuple[list[float], list[float]]:
    x = [a + i * (b - a) / n for i in range(n)]
    y = func_py(x, N)
    return x, y

# NumPy tabulation
def tabulate_np(a: float, b: float, n: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(a, b, n)
    y = func_numpy(x, N)
    return x, y

# Plotting function
def plot_comparison(n_values, t_py, t_np):
    plt.style.use('dark_background')

    # Plot execution times
    plt.figure(figsize=(12, 6))
    plt.plot(n_values, t_py, label="Plain Python", color='red', marker='o')
    plt.plot(n_values, t_np, label="NumPy", color='blue', marker='o')
    plt.xlabel("Number of Points")
    plt.ylabel("Execution Time (µs)")
    plt.title("Execution Time Comparison: Plain Python vs NumPy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot time ratio
    plt.figure(figsize=(12, 6))
    plt.plot(n_values, np.array(t_py) / np.array(t_np), label="Python/NumPy Execution Time Ratio", color='green', marker='o')
    plt.xlabel("Number of Points")
    plt.ylabel("Time Ratio")
    plt.title("Execution Time Ratio: Plain Python vs NumPy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    N = 5
    a, b = 0, 13

    n_values = np.array([1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000], dtype="uint32")
    t_py = []
    t_np = []

    for n in n_values:
        # Measure execution time for plain Python
        time_py = timeit.timeit(f"tabulate_py({a}, {b}, {n}, {N})", number=100, globals=globals())
        t_py.append(time_py * 1_000_000 / 100)  # Convert to microseconds

        # Measure execution time for NumPy
        time_np = timeit.timeit(f"tabulate_np({a}, {b}, {n}, {N})", number=100, globals=globals())
        t_np.append(time_np * 1_000_000 / 100)  # Convert to microseconds

    # Plot the comparison
    plot_comparison(n_values, t_py, t_np)

if __name__ == "__main__":
    main()
