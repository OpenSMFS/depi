# To build the cython extensions use:
# python setup.py build_ext --inplace
#

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = "DEPI with intra-lifetime diffusion core functions",
    ext_modules = cythonize('depi_cy.pyx'),
    include_dirs = [np.get_include()],
)
