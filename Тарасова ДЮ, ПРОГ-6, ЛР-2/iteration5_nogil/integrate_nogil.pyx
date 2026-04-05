from libc.math cimport sin

cdef double f_wrapper(double x) nogil:
    return sin(x)

cpdef double integrate_nogil(double a, double b, int n_iter=100000):
    cdef double acc = 0.0
    cdef double step = (b - a) / n_iter
    cdef int i
    cdef double x
    with nogil:
        for i in range(n_iter):
            x = a + i * step
            acc += f_wrapper(x) * step
    return acc