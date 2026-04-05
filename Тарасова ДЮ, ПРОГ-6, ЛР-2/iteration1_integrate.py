import math
import timeit
import unittest
import doctest
from typing import Callable

def integrate(f: Callable[[float], float], a: float, b: float, *, n_iter: int = 100000) -> float:
    """
    Вычисляет определённый интеграл методом левых прямоугольников.

    Примеры
    --------
    >>> integrate(math.sin, 0, math.pi/2, n_iter=1000)  # doctest: +ELLIPSIS
    0.999...
    >>> def square(x): return x**2
    >>> integrate(square, 0, 1, n_iter=10000)  # doctest: +ELLIPSIS
    0.33328...
    """
    if n_iter <= 0:
        raise ValueError("n_iter > 0")
    acc = 0.0
    step = (b - a) / n_iter
    for i in range(n_iter):
        acc += f(a + i * step) * step
    return acc

class TestIntegrate(unittest.TestCase):
    def test_sin(self):
        res = integrate(math.sin, 0, math.pi/2, n_iter=500000)  # увеличили для точности
        self.assertAlmostEqual(res, 1.0, places=5)  # теперь должно пройти

    def test_poly(self):
        res = integrate(lambda x: x*x, 0, 1, n_iter=500000)
        self.assertAlmostEqual(res, 1/3, places=5)

def benchmark():
    n_iter = 1_000_000
    # Исправленный вызов timeit: передаём n_iter в globals
    t = timeit.timeit(
        'integrate(math.sin, 0, math.pi/2, n_iter=n_iter)',
        globals={**globals(), 'n_iter': n_iter},  # добавляем n_iter в пространство имён
        number=1
    )
    print(f"Базовая integrate, n_iter={n_iter}: {t:.6f} сек")

if __name__ == "__main__":
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    unittest.main(argv=[''], exit=False)
    benchmark()