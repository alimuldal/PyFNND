#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython


# Generalized Thomas algorithm for solving Ax = d where A is a tridiagonal
# matrix with an arbitrary diagonal offset, i.e.:
#
# a{i}x{i-k} + b{i}x{i} + c{i}x{i+k} = d{i}
#
# -----------------------------------------------------------------------------
# http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

cdef void _trisolve_offset(const cython.floating[:] a, cython.floating[:] b,
                           const cython.floating[:] c, cython.floating[:] d):

    cdef:
        Py_ssize_t m = b.shape[0]
        Py_ssize_t n = a.shape[0]
        Py_ssize_t k = m - n
        Py_ssize_t ii

    for ii in range(n):
        b[ii + k] -= c[ii] * a[ii] / b[ii]
        d[ii + k] -= d[ii] * a[ii] / b[ii]

    for ii in range(n - 1, -1, -1):
        d[ii] -= d[ii + k] * c[ii] / b[ii + k]

    for ii in range(m):
        d[ii] /= b[ii]

def trisolve_offset(cython.floating[:] a, cython.floating[:] b,
                    cython.floating[:] c, cython.floating[:] d):

    _trisolve_offset(a, b, c, d)
