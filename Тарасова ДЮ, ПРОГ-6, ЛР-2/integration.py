import math
import timeit
from typing import Callable


def integrate(f: Callable[[float], float], a: float, b: float, *, n_iter: int = 100000) -> float:
    """
    Численное интегрирование методом левых прямоугольников.

    Вычисляет приближённое значение определённого интеграла функции f на отрезке [a, b],
    разбивая отрезок на n_iter равных подынтервалов.

    Parameters
    ----------
    f : Callable[[float], float]
        Интегрируемая функция (принимает float, возвращает float).
    a : float
        Нижний предел интегрирования.
    b : float
        Верхний предел интегрирования.
    n_iter : int, default=100000
        Количество прямоугольников (итераций). Чем больше, тем выше точность,
        но ниже производительность.

    Returns
    -------
    float
        Приближённое значение интеграла.

    Limitations
    -----------
    - Метод имеет погрешность O(1/n_iter).
    - Функция f должна быть непрерывна на [a, b].

    Examples
    --------
    >>> integrate(math.sin, 0, math.pi/2, n_iter=1000)
    0.9999997943832332

    >>> integrate(lambda x: x**2, 0, 1, n_iter=1000)
    0.3328335000000001
    """
    acc = 0.0
    step = (b - a) / n_iter
    for i in range(n_iter):
        acc += f(a + i * step) * step
    return acc


# ==================== Юнит-тесты ====================
import unittest

class TestIntegrate(unittest.TestCase):
    def test_sine_integral(self):
        """∫₀^{π/2} sin x dx = 1"""
        result = integrate(math.sin, 0, math.pi/2, n_iter=100000)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_poly_integral(self):
        """∫₀¹ x² dx = 1/3"""
        result = integrate(lambda x: x**2, 0, 1, n_iter=100000)
        self.assertAlmostEqual(result, 1/3, places=4)

    def test_convergence(self):
        """С увеличением n_iter точность должна расти"""
        rough = integrate(math.cos, 0, math.pi/2, n_iter=100)
        fine = integrate(math.cos, 0, math.pi/2, n_iter=10000)
        exact = 1.0
        self.assertLess(abs(fine - exact), abs(rough - exact))


# ==================== Замер производительности ====================
if __name__ == "__main__":
    # Запуск doctest
    import doctest
    doctest.testmod(verbose=True)

    # Запуск unittest
    unittest.main(argv=[''], exit=False)

    # Замеры времени
    args = (math.cos, 0, math.pi/2)
    for n in [100_000, 1_000_000]:
        t = timeit.timeit(lambda: integrate(*args, n_iter=n), number=10)
        print(f"n_iter={n:8d}: {t/10:.6f} sec per call (10 runs)")
