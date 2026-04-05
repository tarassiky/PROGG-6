from setuptools import setup
from Cython.Build import cythonize

setup(
    name="integrate_cy",
    ext_modules=cythonize("integrate_cy.pyx", annotate=True),
)