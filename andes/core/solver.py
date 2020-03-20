from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from andes.shared import np, matrix, umfpack, klu

import logging
logger = logging.getLogger(__name__)


class Solver(object):
    """
    A sparse matrix solver class.

    This class wraps ``cvxopt.umfpack`` and ``cvxoptklu.klu`` with a unified interface.
    """

    def __init__(self, sparselib='umfpack'):
        self.sparselib = sparselib
        self.F = None
        self.A = None
        self.N = None
        self.factorize = True
        self.use_linsolve = False

        # check if the sparselib is imported successfully
        if globals()[sparselib] is None:
            self.sparselib = 'umfpack'

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

        if self.sparselib == 'umfpack':
            return umfpack.symbolic(A)

        elif self.sparselib == 'klu':
            return klu.symbolic(A)

        elif self.sparselib in ('spsolve', 'cupy'):
            raise NotImplementedError

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
        if self.sparselib == 'umfpack':
            return umfpack.numeric(A, F)

        elif self.sparselib == 'klu':
            return klu.numeric(A, F)

        elif self.sparselib in ('spsolve', 'cupy'):
            raise NotImplementedError

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
        The solution as a ``cvxopt.matrix``.
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

        if self.sparselib in ('umfpack', 'klu'):
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
                self.N = self._numeric(self.A, self.F)
                self._solve(self.A, self.F, self.N, self.b)
                return np.ravel(self.b)
            except ArithmeticError:
                logger.error('Jacobian matrix is singular.')
                return np.ravel(matrix(np.nan, self.b.size, 'd'))

        elif self.sparselib in ('spsolve', 'cupy'):
            return self.linsolve(A, b)

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
        if self.sparselib == 'umfpack':
            try:
                umfpack.linsolve(A, b)
            except ArithmeticError:
                logger.error('Singular matrix. Case is not solvable')
            return np.ravel(b)

        elif self.sparselib == 'klu':
            try:
                klu.linsolve(A, b)
            except ArithmeticError:
                logger.error('Singular matrix. Case is not solvable')
            return np.ravel(b)

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

    def remove_pycapsule(self):
        self.F = None
        self.A = None
        self.N = None
