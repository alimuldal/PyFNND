import numpy as np
from numpy.distutils.core import setup, Extension, build_ext
from numpy.distutils import system_info

# NB: use dotted relative module names here!
# -----------------------------------------------------------------------------

gtsv = Extension(
    name="src.gtsv",
    sources=["src/LAPACK/dgtsv.f", "src/LAPACK/sgtsv.f"],
    extra_link_args=['-llapack']
)


# -----------------------------------------------------------------------------

setup(
    name='pyfnnd',
    author='Alistair Muldal',
    author_email='alistair.muldal@pharm.ox.ac.uk',
    description='A Python implementation of Fast Non-Negative Deconvolution',
    package_dir={'pyfnnd': 'src'},
    cmdclass={'build_ext': build_ext.build_ext},
    ext_modules=[gtsv,],
)
