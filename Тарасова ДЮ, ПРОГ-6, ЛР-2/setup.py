from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="integration_cython",
    ext_modules=cythonize("integration_cython.pyx", compiler_directives={'language_level': 3}),
    include_dirs=[np.get_include()],
    # Для OpenMP (если нужен prange) добавьте:
    # extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']
)
