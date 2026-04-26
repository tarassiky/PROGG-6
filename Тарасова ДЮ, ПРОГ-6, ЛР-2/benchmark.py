import math
import timeit
from integration import integrate
from integration_threads import integrate_threads
from integration_processes import integrate_processes

# Попробуем импортировать Cython-модули (если скомпилированы)
try:
    from integration_cython import integrate_sin_cython, integrate_sin_parallel
    cython_available = True
except ImportError:
    cython_available = False
    print("Cython модуль не найден. Пропускаем Cython-тесты.")

# Параметры
a, b = 0, math.pi/2
n_iter = 10_000_000   # Большое число для заметной разницы
n_jobs = 4           # Количество потоков/процессов

print(f"Тест: ∫₀^{math.pi/2:.2f} sin(x) dx, n_iter = {n_iter:,}")
print(f"Сравнение при {n_jobs} воркерах (там где применимо).\n")

results = {}

# 1. Чистый Python
t_py = timeit.timeit(lambda: integrate(math.sin, a, b, n_iter=n_iter), number=3) / 3
results["Python (базовый)"] = t_py

# 2. Потоки (без ускорения)
t_threads = timeit.timeit(lambda: integrate_threads(math.sin, a, b, n_jobs=n_jobs, n_iter=n_iter), number=3) / 3
results["Потоки (ThreadPool)"] = t_threads

# 3. Процессы
t_processes = timeit.timeit(lambda: integrate_processes(math.sin, a, b, n_jobs=n_jobs, n_iter=n_iter), number=3) / 3
results["Процессы (ProcessPool)"] = t_processes

# 4. Cython с C-функцией (один поток)
if cython_available:
    t_cy_single = timeit.timeit(lambda: integrate_sin_cython(a, b, n_iter=n_iter), number=10) / 10
    results["Cython (sin, один поток)"] = t_cy_single

    # 5. Cython + nogil + prange (параллельно)
    t_cy_parallel = timeit.timeit(lambda: integrate_sin_parallel(a, b, n_iter=n_iter, n_threads=n_jobs), number=10) / 10
    results["Cython + nogil + prange"] = t_cy_parallel

# Вывод результатов
print(f"{'Метод':<30} {'Время (сек)':<15} {'Ускорение':<10}")
print("-" * 55)
baseline = results["Python (базовый)"]
for name, t in results.items():
    speedup = baseline / t
    print(f"{name:<30} {t:<15.5f} {speedup:<10.2f}")
