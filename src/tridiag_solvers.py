import numpy as np
import gtsv


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

    if (dl.shape[0] != du.shape[0] or dl.shape[0] >= d.shape[0]
            or d.shape[0] != b.shape[0]):
        raise ValueError('Invalid diagonal shapes')

    bshape_in = b.shape

    if b.ndim == 1:
        # needs to be (ldb, nrhs)
        b = b[:, None]

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

    n = d.shape[0]
    nrhs = b.shape[1]
    ldb = b.shape[0]

    # b will now be modified in place to give the result
    if rtype == np.float32:
        gtsv.sgtsv(n, nrhs, dl, d, du, b, ldb)
    elif rtype == np.float64:
        gtsv.dgtsv(n, nrhs, dl, d, du, b, ldb)
    else:
        raise ValueError('Unsupported result type: %s' % rtype)

    return b.reshape(bshape_in)


def test_trisolve(n=20000):

    from scipy import sparse
    import scipy.sparse.linalg

    d0 = np.random.randn(n)
    d1 = np.random.randn(n - 1)

    H = sparse.diags((d1, d0, d1), (-1, 0, 1), format='csc')
    x = np.random.randn(n)
    g = np.dot(H, x)

    xhat1 = trisolve(d1, d0, d1, g, inplace=False)
    xhat2 = sparse.linalg.spsolve(H, g)

    print "dgtsv: ", np.linalg.norm(x - xhat1)
    print "LU: ", np.linalg.norm(x - xhat2)
