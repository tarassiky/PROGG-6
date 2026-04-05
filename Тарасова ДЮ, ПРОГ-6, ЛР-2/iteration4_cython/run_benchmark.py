import math
import timeit
from integrate_cy import integrate_cy

def benchmark():
    n_iter = 1_000_000
    t = timeit.timeit(
        'integrate_cy(math.sin, 0, math.pi/2, n_iter=n_iter)',
        globals={**globals(), 'n_iter': n_iter},
        number=1
    )
    print(f"Cython integrate (без nogil), n_iter={n_iter}: {t:.6f} сек")

if __name__ == "__main__":
    benchmark()