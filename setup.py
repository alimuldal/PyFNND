from distutils.core import setup
from Cython.Distutils import Extension, build_ext


# extra compiler flags
COMMON_CF = ['-O3']

cy_trisolve = Extension(
    name="pyfnnd._cy_trisolve",
    sources=["pyfnnd/_cy_trisolve.pyx"],
    extra_compile_args=COMMON_CF,
)

setup(
    name='pyfnnd',
    author='Alistair Muldal',
    author_email='alistair.muldal@pharm.ox.ac.uk',
    description='A Python implementation of fast non-negative deconvolution',
    py_modules=['_fnndeconv', 'demo', '_tridiag_solvers'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[cy_trisolve],
)
