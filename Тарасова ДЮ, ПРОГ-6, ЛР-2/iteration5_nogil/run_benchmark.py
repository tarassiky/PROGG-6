import math
import timeit
import threading
from integrate_nogil import integrate_nogil

def integrate_threads_nogil(a, b, n_jobs, n_iter):
    step = (b - a) / n_jobs
    iter_per_job = n_iter // n_jobs
    results = [0.0] * n_jobs
    def worker(idx, left, right):
        results[idx] = integrate_nogil(left, right, iter_per_job)
    threads = []
    for i in range(n_jobs):
        left = a + i * step
        right = a + (i + 1) * step
        t = threading.Thread(target=worker, args=(i, left, right))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return sum(results)

def benchmark():
    n_iter = 10_000_000  # большое, чтобы увидеть ускорение
    for jobs in (2, 4, 6, 8):
        t = timeit.timeit(
            f'integrate_threads_nogil(0, math.pi/2, {jobs}, {n_iter})',
            globals=globals(), number=1
        )
        print(f"Cython + nogil + потоки, jobs={jobs}: {t:.6f} сек")

if __name__ == "__main__":
    benchmark()