import numpy as np
import ctypes
from ctypes import byref
from ctypes.util import find_library
from numpy.ctypeslib import ndpointer

# try and find a LAPACK shared library
dgtsv, sgtsv = None, None
for name in ('openblas', 'lapack'):
    libname = find_library(name)
    if libname:
        lapack_lib = ctypes.cdll.LoadLibrary(libname)
        try:
            dgtsv = lapack_lib.dgtsv_
            sgtsv = lapack_lib.sgtsv_
            break
        except AttributeError:
            # occurs if the library doesn't define the necessary symbols
            continue
if None in (dgtsv, sgtsv):
    raise EnvironmentError('Could not locate a LAPACK shared library', 2)


# pointer ctypes
_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_double_p = ctypes.POINTER(ctypes.c_double)


def trisolve(dl, d, du, b, inplace=False):
    """
    The tridiagonal matrix (Thomas) algorithm for solving tridiagonal systems
    of equations:

        a_{i}x_{i-1} + b_{i}x_{i} + c_{i}x_{i+1} = y_{i}

    in matrix form:
        Mx = b

    TDMA is O(n), whereas standard Gaussian elimination is O(n^3).

    Arguments:
    -----------
        dl: (n - 1,) vector
            the lower diagonal of M
        d: (n,) vector
            the main diagonal of M
        du: (n - 1,) vector
            the upper diagonal of M
        b: (n,) vector
            the result of Mx
        inplace:
            if True, and if d and b are both float64 vectors, they will be
            modified in place (may be faster)

    Returns:
    -----------
        x: (n,) vector
            the solution to Mx = b

    References:
    -----------
    http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    http://www.netlib.org/lapack/explore-html/d1/db3/dgtsv_8f.html
    """

    if (dl.shape[0] != du.shape[0] or (d.shape[0] != dl.shape[0] + 1)
            or d.shape[0] != b.shape[0]):
        raise ValueError('Invalid diagonal shapes')

    bshape_in = b.shape
    rtype = np.result_type(dl, d, du, b)

    if not inplace:
        # force a copy
        dl = np.array(dl, dtype=rtype, copy=True, order='F')
        d = np.array(d, dtype=rtype, copy=True, order='F')
        du = np.array(du, dtype=rtype, copy=True, order='F')
        b = np.array(b, dtype=rtype, copy=True, order='F')

    # this may also force copies if arrays have inconsistent types / incorrect
    # order
    dl, d, du, b = (np.array(v, dtype=rtype, copy=False, order='F')
                    for v in (dl, d, du, b))

    # use the LAPACK implementation
    _lapack_trisolve(dl, d, du, b, rtype)

    return b.reshape(bshape_in)


def _lapack_trisolve(dl, d, du, b, rtype):

    if b.ndim == 1:
        # needs to be (ldb, nrhs)
        b = b[:, None]

    _n = ctypes.c_int(d.shape[0])
    _nrhs = ctypes.c_int(b.shape[1])
    _ldb = ctypes.c_int(b.shape[0])
    _info = ctypes.c_int(1)

    # b will now be modified in place to give the result
    if rtype == np.float32:
        sgtsv(byref(_n), byref(_nrhs),
              dl.ctypes.data_as(_c_float_p),
              d.ctypes.data_as(_c_float_p),
              du.ctypes.data_as(_c_float_p),
              b.ctypes.data_as(_c_float_p),
              byref(_ldb), byref(_info))

    elif rtype == np.float64:
        dgtsv(byref(_n), byref(_nrhs),
              dl.ctypes.data_as(_c_double_p),
              d.ctypes.data_as(_c_double_p),
              du.ctypes.data_as(_c_double_p),
              b.ctypes.data_as(_c_double_p),
              byref(_ldb), byref(_info))
    else:
        raise ValueError('Unsupported result type: %s' % rtype)


def bench_trisolve():

    import time
    from scipy import sparse
    import scipy.sparse.linalg

    N = np.logspace(2, 6, 5).astype(np.int)
    nreps = 5

    k = 1

    dgtsv_times = []
    LU_times = []

    for n in N:

        d0 = np.random.randn(n)
        d1 = np.random.randn(n - k)

        H = sparse.diags((d1, d0, d1), (-k, 0, k), format='csc')
        x = np.random.randn(n)
        g = H.dot(x)

        start = time.time()
        for _ in xrange(nreps):
            xhat = trisolve(d1, d0, d1, g, inplace=False)
        t1 = (time.time() - start) / nreps
        norm1 = np.linalg.norm(x - xhat)

        start = time.time()
        for _ in xrange(nreps):
            xhat = sparse.linalg.spsolve(H, g)
        t2 = (time.time() - start) / nreps
        norm2 = np.linalg.norm(x - xhat)

        print "\nn = %i" % n
        print "Time (sec):\ttridiag: %g\tLU: %g" % (t1, t2)
        print "||x - xhat||2:\ttridiag: %g\tLU: %g" % (norm1, norm2)

        dgtsv_times.append(t1)
        LU_times.append(t2)

    return N, np.array(dgtsv_times), np.array(LU_times)
