import numpy as np
import matplotlib.pyplot as plt
import timeit

def sinc(x):
    """Calculate sinc function."""
    return np.sin(x) / x if x != 0 else 1

def func_py(x: list[float], N: int) -> list[float]:
    """
    Calculate function values for passed array of arguments using plain Python.
    """
    return [sinc(2 * t / N - 1) for t in x]

def tabulate_py(a: float, b: float, n: int, N: int) -> tuple[list[float], list[float]]:
    x = [a + i * (b - a) / n for i in range(n)]
    y = func_py(x, N)
    return x, y

def func_numpy(x: np.ndarray, N: int) -> np.ndarray:
    """
    Calculate function values for passed array of arguments using NumPy.
    """
    x_transformed = 2 * x / N - 1
    return np.sinc(x_transformed / np.pi)  # np.sinc(x) is normalized sinc (sin(πx)/(πx))

def tabulate_np(a: float, b: float, n: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(a, b, n)
    y = func_numpy(x, N)
    return x, y

def main():
    N = 5
    a, b = 0, 13

    sizes = np.array([1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000], dtype="uint32")
    t_py = np.zeros_like(sizes, dtype=float)
    t_np = np.zeros_like(sizes, dtype=float)

    for i, n in enumerate(sizes):
        t_py[i] = 1_000_000 * timeit.timeit(f"tabulate_py({a}, {b}, {n}, {N})", number=100, globals=globals()) / 100
        t_np[i] = 1_000_000 * timeit.timeit(f"tabulate_np({a}, {b}, {n}, {N})", number=100, globals=globals()) / 100

    # Plot the ratio of execution times
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, t_py, label="Plain Python")
    plt.plot(sizes, t_np, label="NumPy")
    plt.xlabel("Number of Points")
    plt.ylabel("Execution Time (µs)")
    plt.title("Execution Time Comparison: Plain Python vs NumPy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the ratio
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, t_py / t_np, label="Python/NumPy Execution Time Ratio")
    plt.xlabel("Number of Points")
    plt.ylabel("Time Ratio")
    plt.title("Execution Time Ratio: Plain Python vs NumPy")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
