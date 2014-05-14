from distutils.core import setup

setup(
    name='pyfnnd',
    author='Alistair Muldal',
    author_email='alistair.muldal@pharm.ox.ac.uk',
    description='A Python implementation of fast non-negative deconvolution',
    py_modules=['_fnndeconv', 'demo', '_tridiag_solvers'],
)
