"""
SuiteSparse solvers provided by ``kvxopt``.
"""

import logging

import numpy as np
from kvxopt import matrix, umfpack, klu

logger = logging.getLogger(__name__)


class SuiteSparseSolver:
    """
    Base SuiteSparse solver interface.

    Need to be derived by specific solvers such as UMFPACK or KLU.
    """

    def __init__(self):
        self.A = None
        self.b = None
        self.F = None   # symbolic factorization
        self.N = None   # numeric factorization
        self.factorize = True
        self.new_A = False  # does not need to handle new A in suitesparse solvers
        self.use_linsolve = False

    def clear(self):
        """
        Remove all cached PyCapsule of C objects
        """
        self.A = None
        self.b = None
        self.F = None   # symbolic factorization
        self.N = None   # numeric factorization
        self.factorize = True
        self.use_linsolve = False

    def _symbolic(self, A):
        """
        Return the symbolic factorization of sparse matrix ``A``.

        Parameters
        ----------
        A
            Sparse matrix to be factorized.

        Returns
        -------
        A C-object of the symbolic factorization.
        """
        raise NotImplementedError("Method needs to implemented by solver class.")

    def _numeric(self, A, F):
        """
        Return the numeric factorization of sparse matrix ``A`` using symbolic factorization ``F``.

        Parameters
        ----------
        A
            Sparse matrix for the equation set coefficients.
        F
            The symbolic factorization of a matrix with the same non-zero shape as ``A``.

        Returns
        -------
        The numeric factorization of ``A``.
        """
        raise NotImplementedError("Method needs to implemented by solver class.")

    def _solve(self, A, F, N, b):
        """
        Solve linear system ``Ax = b`` using numeric factorization ``N`` and symbolic factorization ``F``.

        Parameters
        ----------
        A
            Sparse matrix.
        F
            Symbolic factorization
        N
            Numeric factorization
        b
            RHS of the equation

        Returns
        -------
        The solution as a ``kvxopt.matrix``.
        """
        raise NotImplementedError("Method needs to implemented by solver class.")

    def solve(self, A, b):
        """
        Solve linear system ``Ax = b`` using numeric factorization ``N`` and symbolic factorization ``F``.
        Store the solution in ``b``.

        This function caches the symbolic factorization in ``self.F`` and is faster in general.
        Will attempt ``Solver.linsolve`` if the cached symbolic factorization is invalid.

        Parameters
        ----------
        A
            Sparse matrix for the equation set coefficients.
        F
            The symbolic factorization of A or a matrix with the same non-zero shape as ``A``.
        N
            Numeric factorization of A.
        b
            RHS of the equation.

        Returns
        -------
        numpy.ndarray
            The solution in a 1-D ndarray
        """
        self.A = A
        self.b = b

        if self.factorize is True:
            self.F = self._symbolic(self.A)
            self.factorize = False

        try:
            self.N = self._numeric(self.A, self.F)
            self._solve(self.A, self.F, self.N, self.b)

            return np.ravel(self.b)
        except ValueError:
            logger.debug('Unexpected symbolic factorization.')
            self.F = self._symbolic(self.A)
            self.solve(self.A, self.b)

            return np.ravel(self.b)
        except ArithmeticError:
            logger.error('Jacobian matrix is singular.')
            # diag = self.A[0:self.A.size[0] ** 2:self.A.size[0]+1]
            # idx = (np.argwhere(np.array(matrix(diag)).ravel() == 0.0)).ravel()
            # logger.error('The xy indices of associated variables:')
            # logger.error(' '.join([str(item) for item in idx]))

            # works around a KVXOPT bug
            suspect_diag = []
            for i in range(self.A.size[0]):
                if self.A[i, i] == 0.0:
                    suspect_diag.append(i)

            logger.error('Suspect diagonal elements: {}'.format(suspect_diag))

            return np.ravel(matrix(np.nan, self.b.size, 'd'))

    def linsolve(self, A, b):
        """
        Solve linear equation set ``Ax = b`` and returns the solutions in a 1-D array.

        This function performs both symbolic and numeric factorizations every time, and can be slower than
        ``Solver.solve``.

        Parameters
        ----------
        A
            Sparse matrix

        b
            RHS of the equation

        Returns
        -------
        The solution in a 1-D np array.
        """
        raise NotImplementedError


class UMFPACKSolver(SuiteSparseSolver):
    """
    UMFPACK solver.

    Utilizes ``kvxopt.umfpack`` for factorization.
    """

    def __init__(self):
        super().__init__()

    def _symbolic(self, A):
        return umfpack.symbolic(A)

    def _numeric(self, A, F):
        return umfpack.numeric(A, F)

    def _solve(self, A, F, N, b):
        umfpack.solve(A, N, b)

    def linsolve(self, A, b):
        try:
            umfpack.linsolve(A, b)
        except ArithmeticError:
            logger.error('Singular matrix. Case is not solvable')
        return np.ravel(b)


class KLUSolver(SuiteSparseSolver):
    """
    KLU solver.
    """

    def __init__(self):
        super().__init__()

    def _symbolic(self, A):
        return klu.symbolic(A)

    def _numeric(self, A, F):
        return klu.numeric(A, F)

    def _solve(self, A, F, N, b):
        klu.solve(A, F, N, b)

    def linsolve(self, A, b):
        try:
            klu.linsolve(A, b)
        except ArithmeticError:
            logger.error('Singular matrix. Case is not solvable')
        return np.ravel(b)
