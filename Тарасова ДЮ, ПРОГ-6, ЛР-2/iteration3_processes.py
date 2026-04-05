import math
import timeit
import concurrent.futures as ftres
from typing import Callable
from iteration1_integrate import integrate

def integrate_processes(f: Callable[[float], float], a: float, b: float, *,
                        n_jobs: int = 2, n_iter: int = 100000) -> float:
    n_iter = (n_iter // n_jobs) * n_jobs
    if n_iter == 0:
        return 0.0
    step = (b - a) / n_jobs
    iter_per_job = n_iter // n_jobs
    with ftres.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(integrate, f, a + i*step, a + (i+1)*step, n_iter=iter_per_job)
                   for i in range(n_jobs)]
        return sum(f.result() for f in ftres.as_completed(futures))

def benchmark():
    n_iter = 1_000_000
    for jobs in (2, 4, 6, 8):
        t = timeit.timeit(
            f'integrate_processes(math.sin, 0, math.pi/2, n_jobs={jobs}, n_iter={n_iter})',
            globals=globals(), number=1
        )
        print(f"Процессы, jobs={jobs}: {t:.6f} сек")

if __name__ == "__main__":
    benchmark()