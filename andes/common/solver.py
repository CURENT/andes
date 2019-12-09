import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from cvxopt import umfpack, matrix  # NOQA

import logging
logger = logging.getLogger(__name__)

try:
    from cvxoptklu import klu
except ImportError:
    klu = None


class Solver(object):
    """
    Sparse matrix solver class. Wraps UMFPACK and KLU with a unified interface.
    """

    def __init__(self, sparselib='umfpack'):
        self.sparselib = sparselib
        self.F = None
        self.A = None
        self.N = None
        self.factorize = True
        self.use_linsolve = False

        # input checking
        if globals()[sparselib] is None:
            self.sparselib = 'umfpack'

    def symbolic(self, A):
        """
        Return the symbolic factorization of sparse matrix ``A``

        Parameters
        ----------
        sparselib
            Library name in ``umfpack`` and ``klu``
        A
            Sparse matrix

        Returns
        symbolic factorization
        -------

        """

        if self.sparselib == 'umfpack':
            return umfpack.symbolic(A)

        elif self.sparselib == 'klu':
            return klu.symbolic(A)

        elif self.sparselib in ('spsolve', 'cupy'):
            raise NotImplementedError

    def numeric(self, A, F):
        """
        Return the numeric factorization of sparse matrix ``A`` using symbolic factorization ``F``

        Parameters
        ----------
        A
            Sparse matrix
        F
            Symbolic factorization

        Returns
        -------
        N
            Numeric factorization of ``A``
        """
        if self.sparselib == 'umfpack':
            return umfpack.numeric(A, F)

        elif self.sparselib == 'klu':
            return klu.numeric(A, F)

        elif self.sparselib in ('spsolve', 'cupy'):
            raise NotImplementedError

    def _solve(self, A, F, N, b):
        """
        Solve linear system ``Ax = b`` using numeric factorization ``N`` and symbolic factorization ``F``.
        Store the solution in ``b``.

        Parameters
        ----------
        A
            Sparse matrix
        F
            Symbolic factorization
        N
            Numeric factorization
        b
            RHS of the equation

        Returns
        -------
        None
        """
        if self.sparselib == 'umfpack':
            umfpack.solve(A, N, b)
            return b

        elif self.sparselib == 'klu':
            klu.solve(A, F, N, b)
            return b

        elif self.sparselib in ('spsolve', 'cupy'):
            raise NotImplementedError

    def solve(self, A, b):
        """
        Solve linear system ``Ax = b`` using numeric factorization ``N`` and symbolic factorization ``F``.
        Store the solution in ``b``.

        Parameters
        ----------
        A
            Sparse matrix
        F
            Symbolic factorization
        N
            Numeric factorization
        b
            RHS of the equation

        Returns
        -------
        None
        """
        self.A = A
        self.b = b

        if self.sparselib in ('umfpack', 'klu'):
            if self.factorize is True:
                self.F = self.symbolic(self.A)
                self.factorize = False

            try:
                self.N = self.numeric(self.A, self.F)
                self._solve(self.A, self.F, self.N, self.b)
                return self.b
            except ValueError:
                logger.debug('Unexpected symbolic factorization.')
                self.factorize = True
                return self.linsolve(self.A, self.b)
            except ArithmeticError:
                logger.error('Jacobian matrix is singular.')
                return np.ravel(matrix(1, self.b.size, 'd'))

        elif self.sparselib in ('spsolve', 'cupy'):
            return self.linsolve(A, b)

    def linsolve(self, A, b):
        """
        Solve linear equation set ``Ax = b`` and store the solutions in ``b``.

        Parameters
        ----------
        A
            Sparse matrix

        b
            RHS of the equation

        Returns
        -------
        None
        """
        if self.sparselib == 'umfpack':
            try:
                umfpack.linsolve(A, b)
            except ArithmeticError:
                logger.error('Singular matrix. Case is not solvable')
            return b

        elif self.sparselib == 'klu':
            try:
                klu.linsolve(A, b)
            except ArithmeticError:
                logger.error('Singular matrix. Case is not solvable')
            return b

        elif self.sparselib in ('spsolve', 'cupy'):
            ccs = A.CCS
            size = A.size
            data = np.array(ccs[2]).reshape((-1,))
            indices = np.array(ccs[1]).reshape((-1,))
            indptr = np.array(ccs[0]).reshape((-1,))

            A = csc_matrix((data, indices, indptr), shape=size)

            if self.sparselib == 'spsolve':
                x = spsolve(A, b)
                return np.ravel(x)

            elif self.sparselib == 'cupy':
                # delayed import for startup speed
                import cupy as cp  # NOQA
                from cupyx.scipy.sparse import csc_matrix as csc_cu  # NOQA
                from cupyx.scipy.sparse.linalg.solve import lsqr as cu_lsqr  # NOQA

                cu_A = csc_cu(A)
                cu_b = cp.array(np.array(b).reshape((-1,)))
                x = cu_lsqr(cu_A, cu_b)

                return np.ravel(cp.asnumpy(x[0]))
