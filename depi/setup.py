# To build the cython extensions use:
# python setup.py build_ext --inplace
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'depi_cy',
        ['depi_cy.pyx'],
        include_dirs=[np.get_include()],
        # Uncomment the following line for profiling
        #define_macros=[('CYTHON_TRACE', '1')],
    )
]

setup(
    name="DEPI with intra-lifetime diffusion core functions",
    ext_modules=cythonize(
        extensions,
        # Uncomment the following line for profiling
        #compiler_directives={'linetrace': True, 'binding': True},
    ),
)
