from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        'depi.depi_cy',
        ['depi/depi_cy.pyx'],
        include_dirs=['.', np.get_include()],
        # Uncomment the following line for profiling
        # define_macros=[('CYTHON_TRACE', '1')],
    )
]

setup(
    name="depi",
    version='0.1',
    author='Antonino Ingargiola',
    author_email='tritemio@gmail.com',
    url='http://opensmfs.github.io/depi/',
    download_url='http://opensmfs.github.io/depi/',
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'matplotlib', 'cython', 'pandas', 'joblib'],
    license='MIT',
    description=("Monte-Carlo analysis of single-molecule FRET data including "
                 "diffusion and photo-physics."),
    platforms=('Windows', 'Linux', 'Mac OS X'),
    ext_modules=cythonize(
        extensions,
        # Uncomment the following line for profiling
        # compiler_directives = {'linetrace': True, 'binding': True},
    ),
    packages=['depi'],
    zip_safe=False,
    keywords='Single-molecule FRET smFRET burst-analysis biophysics photo-physics HSE',
)
