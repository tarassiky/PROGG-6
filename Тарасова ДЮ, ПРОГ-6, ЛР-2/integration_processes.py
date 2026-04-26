import math
import timeit
from typing import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from integration import integrate


def integrate_processes(f: Callable[[float], float], a: float, b: float, *,
                        n_jobs: int = 2, n_iter: int = 100000) -> float:
    """
    Вычисление интеграла с разбиением на n_jobs процессов.

    Каждый процесс работает независимо со своей копией данных.
    """
    step_big = (b - a) / n_jobs
    n_per_job = n_iter // n_jobs
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        spawn = partial(executor.submit, integrate, f, n_iter=n_per_job)
        futures = [spawn(a + i*step_big, a + (i+1)*step_big) for i in range(n_jobs)]
        total = sum(f.result() for f in as_completed(futures))
    return total


if __name__ == "__main__":
    args = (math.cos, 0, math.pi/2)
    n_iter = 1_000_000

    print("МНОГОПРОЦЕССНАЯ ВЕРСИЯ (ожидается ускорение до числа ядер):")
    for jobs in [1, 2, 4, 8]:
        t = timeit.timeit(lambda: integrate_processes(*args, n_jobs=jobs, n_iter=n_iter), number=3)
        print(f"  Процессов: {jobs:2d}, время: {t/3:.4f} сек за вызов")
