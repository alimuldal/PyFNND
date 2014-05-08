import numpy as np
from numpy.distutils.core import setup, Extension, build_ext
import os

# NB: use dotted relative module names here!
# -----------------------------------------------------------------------------

fortran_sources = "dgtsv.f", "sgtsv.f"
gtsv = Extension(
    name="pyfnnd._gtsv",
    sources=[os.path.join("pyfnnd", "LAPACK", ff) for ff in fortran_sources],
    extra_link_args=['-llapack']
)


# -----------------------------------------------------------------------------

setup(
    name='pyfnnd',
    author='Alistair Muldal',
    author_email='alistair.muldal@pharm.ox.ac.uk',
    description='A Python implementation of fast non-negative deconvolution',
    py_modules=['_fnndeconv', 'demo', '_tridiag_solvers'],
    cmdclass={'build_ext': build_ext.build_ext},
    ext_modules=[gtsv, ],
)
