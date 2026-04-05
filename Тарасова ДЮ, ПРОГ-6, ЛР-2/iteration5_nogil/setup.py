from setuptools import setup
from Cython.Build import cythonize

setup(
    name="integrate_nogil",
    ext_modules=cythonize("integrate_nogil.pyx", annotate=True),
)