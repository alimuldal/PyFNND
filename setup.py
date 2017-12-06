from distutils.core import setup

setup(
    name='pyfnnd',
    author='Alistair Muldal',
    version='0.5',
    author_email='alimuldal@gmail.com',
    url='https://github.com/alimuldal/PyFNND',
    description='A Python implementation of fast non-negative deconvolution',
    packages=['pyfnnd'],
    install_requires=[
        'numpy>=1.8.1',
        'scipy>=0.14.0',
    ],
)
