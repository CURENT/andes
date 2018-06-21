from math import floor

from numpy import sign as sgn
from numpy import array

from cvxopt import matrix
from cvxopt import mul, exp


def altb(a, b):
    """Return a matrix of logic comparison of A<B"""
    if type(b) in (int, float):
        b = matrix(b, (len(a), 1), 'd')
    return matrix(list(map(lambda x, y: x < y, a, b)), a.size)


def mmax(a, b):
    """Return a matrix of maximum values in a and b element-wise"""
    if type(b) in (int, float):
        b = matrix(b, (len(a), 1), 'd')
    return matrix(list(map(lambda x, y: x if x > y else y, a, b)), a.size)


def mmin(a, b):
    """Return a matrix of minimum values in a and b element-wise"""
    if type(b) in (int, float):
        b = matrix(b, (len(a), 1), 'd')
    return matrix(list(map(lambda x, y: x if x < y else y, a, b)), a.size)


def agtb(a, b):
    """Return a matrix of logic comparision of A>B"""
    if type(b) in (int, float):
        b = matrix(b, (len(a), 1), 'd')
    return matrix(list(map(lambda x, y: x > y, a, b)), a.size)

def aleb(a, b):
    """Return a matrix of logic comparison of A<=B"""
    if type(b) in (int, float):
        b = matrix(b, (len(a), 1), 'd')
    return matrix(list(map(lambda x, y: x <= y, a, b)), a.size)


def ageb(a, b):
    """Return a matrix of logic comparision of A>=B"""
    if type(b) in (int, float):
        b = matrix(b, (len(a), 1), 'd')
    return matrix(list(map(lambda x, y: x >= y, a, b)), a.size)


def aeb(a, b):
    """Return a matrix of logic comparison of A == B"""
    if type(b) in (int, float):
        return matrix(list(map(lambda x: x == b, a)), a.size)
    else:
        return matrix(list(map(lambda x, y: x == y, a, b)), a.size)


def aneb(a, b):
    """Return a matrix of logic comparison of A != B"""
    if type(b) in (int, float):
        return matrix(list(map(lambda x: x != b, a)), a.size)
    else:
        return matrix(list(map(lambda x, y: x != y, a, b)), a.size)


def aorb(a, b):
    """Return a matrix of logic comparison of A or B"""
    return matrix(list(map(lambda x, y: x or y, a, b)), a.size)

def aandb(a, b):
    """Return a matrix of logic comparison of A or B"""
    return matrix(list(map(lambda x, y: x and y, a, b)), a.size)


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


def mfloor(a):
    """Return the element-wise floor value of a"""
    return matrix(list(map(lambda x: floor(x), a)), a.size)


def mround(a):
    """Return the element-wise round value of a"""
    return matrix(list(map(lambda x: round(x), a)), a.size)


def not0(a):
    """Return u if u!= 0, return 1 if u == 0"""
    return matrix(list(map(lambda x: 1 if x == 0 else x, a)), a.size)


def zeros(m, n):
    """Return a m-by-n zero-value matrix"""
    return matrix(0.0, (m, n), 'd')


def ones(m, n):
    """Return a m-by-n one-value matrix"""
    return matrix(1.0, (m, n), 'd')


def sign(a):
    """Return the sign of a in (1, -1, 0)"""
    return matrix(sgn(array(a)))


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


def findeq(m, val):
    """Return the indices of all (val) in m"""
    m = list(m)
    idx = []
    if m.count(val) > 0:
        idx = [i for i, j in enumerate(m) if j == val]
    return idx


def algeb_limiter(m, upper, lower):
    above = agtb(m, upper)
    idx = findeq(above, 1.0)
    m[idx] = upper[idx]

    below = altb(m, lower)
    idx = findeq(below, 1.0)
    m[idx] = lower[idx]

    return m

def to_number(s):
    """Convert a string to a number. If not successful, return the string without blanks"""
    ret = s
    # try converting to float
    try:
        ret = float(s)
    except ValueError:
        ret = ret.strip('\'').strip()
    # try converting to int
    try:
        ret = int(s)
    except ValueError:
        pass
    # try converting to boolean
    if ret == 'True':
        ret = True
    elif ret == 'False':
        ret = False
    elif ret == 'None':
        ret = None
    return ret


def sdiv(a, b):
    """Safe division: if a == b == 0, sdiv(a, b) == 1"""
    if len(a) != len(b):
        raise ValueError('Argument a and b does not have the same length')
    idx = 0
    ret = matrix(0, (len(a), 1), 'd')
    for m, n in zip(a, b):
        try:
            ret[idx] = m / n
        except ZeroDivisionError:
            ret[idx] = 1
        finally:
            idx += 1
    return ret