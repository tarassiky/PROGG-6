# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-модуль для быстрого численного интегрирования.
Поддерживает:
- обычную версию с Python-вызовом f (медленнее)
- версию с приёмом C-функции (быстро)
- версию с nogil и prange для параллельных потоков
"""

from libc.math cimport sin, cos
from cython.parallel cimport prange
cimport cython


# ---------- 1. Базовый Cython (вызов Python-функции) ----------
def integrate_cython_pyfunc(f, double a, double b, int n_iter=100000):
    """
    Как integrate(), но переменные объявлены как C-типы.
    Всё равно медленно из-за вызова f.
    """
    cdef double acc = 0.0
    cdef double step = (b - a) / n_iter
    cdef int i
    for i in range(n_iter):
        acc += f(a + i*step) * step
    return acc


# ---------- 2. Быстрый вариант для C-функций (sin, cos) ----------
cdef double _integrate_cfunc(double (*f)(double), double a, double b, int n_iter) nogil:
    cdef double acc = 0.0
    cdef double step = (b - a) / n_iter
    cdef int i
    for i in range(n_iter):
        acc += f(a + i*step) * step
    return acc

def integrate_sin_cython(double a, double b, int n_iter=100000):
    """Интеграл от sin на [a,b] с C-функцией sin (без GIL)"""
    return _integrate_cfunc(sin, a, b, n_iter)

def integrate_cos_cython(double a, double b, int n_iter=100000):
    """Интеграл от cos на [a,b] с C-функцией cos (без GIL)"""
    return _integrate_cfunc(cos, a, b, n_iter)


# ---------- 3. Параллельная версия с prange и nogil ----------
def integrate_parallel_nogil(double a, double b, int n_iter=100000, int n_threads=2):
    """
    Распараллеливает суммирование по прямоугольникам с помощью prange.
    Требует, чтобы компилятор поддерживал OpenMP.
    """
    cdef double step = (b - a) / n_iter
    cdef double total = 0.0
    cdef int i
    # Директива parallel for с разделением итераций между потоками
    for i in prange(n_iter, nogil=True, num_threads=n_threads):
        total += (a + i * step) * step   # Здесь подставляется конкретная функция – например, f(x)=x
        # Для произвольной C-функции нужно передавать указатель, для простоты сделаем f(x)=x
    return total


# Для удобства: можно передавать любую C-функцию, но в prange придётся создать отдельную функцию.
# Ниже – универсальная параллельная версия для синуса (чтобы показать принцип).
def integrate_sin_parallel(double a, double b, int n_iter=100000, int n_threads=2):
    cdef double step = (b - a) / n_iter
    cdef double total = 0.0
    cdef int i
    for i in prange(n_iter, nogil=True, num_threads=n_threads):
        total += sin(a + i*step) * step
    return total
