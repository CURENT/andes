from cvxopt import matrix
from cvxopt import mul, exp


def altb(a, b):
    """Return a matrix of logic comparison of A<B"""
    return matrix(list(map(lambda x, y: x < y, a, b)), a.size)


def agtb(a, b):
    """Return a matrix of logic comparision of A>B"""
    return matrix(list(map(lambda x, y: x > y, a, b)), a.size)


def aorb(a, b):
    """Return a matrix of logic comparison of A or B"""
    return matrix(list(map(lambda x, y: x or y, a, b)), a.size)


def nota(a):
    """Return a matrix of logic negative of A"""
    return matrix(list(map(lambda x: not x, a)), a.size)


def polar(m, a):
    """Return complex number from polar form m*exp(1j*a)"""
    return mul(m, exp(1j*a))


def conj(a):
    """return the conjugate of a"""
    return a.H.T


def neg(u):
    """Return the negative of binary states u"""
    return 1 - u


def zeros(m, n):
    """Return a m-by-n zero-value matrix"""
    return matrix(0.0, (m, n), 'd')


def ones(m, n):
    """Return a m-by-n one-value matrix"""
    return matrix(1.0, (m, n), 'd')


def sort(m, reverse=False):
    """Return sorted m (default: ascending order)"""
    ty = type(m)
    if ty == matrix:
        m = list(m)
    m = sorted(m, reverse=reverse)
    if ty == matrix:
        m = matrix(m)
    return m


def sort_idx(m, reverse=False):
    """Return the indices of m in sorted order (default: ascending order)"""
    return sorted(range(len(m)), key=lambda k: m[k], reverse=reverse)


def findall(m, val):
    """Return the indices of all (val) in m"""
    m = list(m)
    idx = []
    if m.count(val) > 1:
        idx = [i for i, j in enumerate(m) if j == val]
    return idx
