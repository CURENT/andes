"""
Scipy sparse linear solver with SuperLU backend.
"""

import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, splu


class SciPySolver:
    """
    Base class for scipy family solvers.
    """

    def __init__(self):
        self.factorize = True

        # when `new_A` is True, rebuild and factorize A
        self.new_A = True

    def solve(self, A, b):
        """
        Solve linear systems.

        Parameters
        ----------
        A : scipy.csc_matrix
            Sparse N-by-N matrix
        b : numpy.ndarray
            Dense 1-dimensional array of size N

        Returns
        -------
        np.ndarray
            Solution x to `Ax = b`

        """
        raise NotImplementedError

    def linsolve(self, A, b):
        """
        Exactly same functionality as `solve`.
        """
        return self.solve(A, b)

    def clear(self):
        pass


class SpSolve(SciPySolver):
    """
    scipy.sparse.linalg.spsolve Solver.
    """

    def solve(self, A, b):

        if self.factorize or self.new_A:
            A_csc = spmatrix_to_csc(A)
            self.lu = splu(A_csc)

            self.factorize = False
            self.new_A = False

        x = self.lu.solve(np.ravel(b))
        return x

    def linsolve(self, A, b):
        """
        Solve using `spsolve`.
        """

        A_csc = spmatrix_to_csc(A)
        b = np.ravel(b)
        return spsolve(A_csc, b)


def spmatrix_to_csc(A):
    """
    Convert A of ``kvxopt.spmatrix`` to ``scipy.sparse.csc_matrix``.

    Parameters
    ----------
    A : kvxopt.spmatrix
        Sparse N-by-N matrix

    Returns
    -------
    scipy.sparse.csc_matrix
        Converted csc_matrix

    """
    ccs = A.CCS
    size = A.size
    data = np.array(ccs[2]).ravel()
    indices = np.array(ccs[1]).ravel()
    indptr = np.array(ccs[0]).ravel()
    return csc_matrix((data, indices, indptr), shape=size)
