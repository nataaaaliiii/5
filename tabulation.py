import numpy as np
from matplotlib import pyplot as plt

# Define the sinc function
def sinc(x):
    return np.sin(x) / x if x != 0 else 1

# Plain Python implementation of the function
def func_py(x: list[float], N: int) -> list[float]:
    return [sinc(2 * t * N - 1) for t in x]

# NumPy implementation of the function
def func_np(x: np.ndarray, N: int) -> np.ndarray:
    return np.sinc(2 * x * N - 1)

# Plain Python tabulation
def tabulate_py(a: float, b: float, n: int, N: int) -> tuple[list[float], list[float]]:
    x = [a + i * (b - a) / n for i in range(n)]
    y = func_py(x, N)
    return x, y

# NumPy tabulation
def tabulate_np(a: float, b: float, n: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(a, b, n)
    y = func_np(x, N)
    return x, y

def test_tabulation(f, a, b, n, N, axis):
    res = f(a, b, n, N)
    axis.plot(res[0], res[1])
    axis.grid()

def main():
    a, b, n = 0, 1, 1000
    N = 5

    fig, (ax1, ax2) = plt.subplots(2, 1)
    test_tabulation(tabulate_py, a, b, n, N, ax1)
    test_tabulation(tabulate_np, a, b, n, N, ax2)
    plt.show()

if __name__ == '__main__':
    main()
