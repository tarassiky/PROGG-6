import math
import timeit
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from integration import integrate


def integrate_threads(f: Callable[[float], float], a: float, b: float, *,
                      n_jobs: int = 2, n_iter: int = 100000) -> float:
    """
    Вычисление интеграла с разбиением на n_jobs потоков.

    Каждый поток получает подотрезок [a_i, a_{i+1}] и выполняет integrate()
    с n_iter // n_jobs итерациями. Потом результаты суммируются.

    Parameters
    ----------
    f : Callable
        Интегрируемая функция.
    a, b : float
        Пределы интегрирования.
    n_jobs : int
        Количество потоков.
    n_iter : int
        Общее количество прямоугольников (распределяется между потоками).

    Returns
    -------
    float
        Приближённое значение интеграла.

    Notes
    -----
    Из-за GIL потоки не дают ускорения для CPU-интенсивных задач на чистом Python.
    """
    step_big = (b - a) / n_jobs
    n_per_job = n_iter // n_jobs   # + остаток? упрощённо
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        spawn = partial(executor.submit, integrate, f, n_iter=n_per_job)
        futures = [spawn(a + i*step_big, a + (i+1)*step_big) for i in range(n_jobs)]
        total = sum(f.result() for f in as_completed(futures))
    return total


if __name__ == "__main__":
    args = (math.cos, 0, math.pi/2)
    n_iter = 1_000_000

    print("МНОГОПОТОЧНАЯ ВЕРСИЯ (ожидаем отсутствия ускорения):")
    for jobs in [1, 2, 4, 8]:
        t = timeit.timeit(lambda: integrate_threads(*args, n_jobs=jobs, n_iter=n_iter), number=5)
        print(f"  Потоков: {jobs:2d}, время: {t/5:.4f} сек за вызов")
