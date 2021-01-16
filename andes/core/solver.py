"""
Sparse solvers wrapper.
"""

import logging

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from andes.shared import np, matrix, umfpack, klu, cupy

logger = logging.getLogger(__name__)


class Solver:
    """
    Sparse matrix solver class.

    This class wraps UMFPACK, KLU, SciPy and CuPy solvers to provide an unified
    interface for solving sparse linear equations ``Ax = b``.

    Provides methods ``solve``, ``linsolve`` and ``clear``.
    """

    def __init__(self, sparselib='umfpack'):
        self.sparselib = sparselib

        # check if `sparselib` library has been successfully imported
        if (sparselib not in globals()) or globals()[sparselib] is None:
            self.sparselib = 'umfpack'

        # solvers
        self.umfpack = UMFPACKSolver()
        self.klu = KLUSolver()
        self.spsolve = SpSolve()
        self.cupy = CuPySolver()

        self.worker = self.__dict__[self.sparselib]

    def solve(self, A, b):
        """
        Solve linear equations and cache factorizations if possible.

        Parameters
        ----------
        A : kvxopt.spmatrix
            Sparse N-by-N matrix
        b : kvxopt.matrix or numpy.ndarray
            Dense N-by-1 matrix

        Returns
        -------
        numpy.ndarray
            Dense N-by-1 array
        """
        return self.worker.solve(A, b)

    def linsolve(self, A, b):
        """
        Solve linear equations without caching facorization. Performs full factorization each call.

        Parameters
        ----------
        A : kvxopt.spmatrix
            Sparse N-by-N matrix
        b : kvxopt.matrix or numpy.ndarray
            Dense N-by-1 matrix

        Returns
        -------
        numpy.ndarray
            Dense N-by-1 array
        """
        return self.worker.linsolve(A, b)

    def clear(self):
        """
        Remove all cached objects.
        """
        self.worker.clear()


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
        return None

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
        return None

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
        return None

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
            diag = self.A[0:self.A.size[0] ** 2:self.A.size[0]+1]
            idx = (np.argwhere(np.array(matrix(diag)).ravel() == 0.0)).ravel()
            logger.error('The xy indices of associated variables:')
            logger.error(idx)

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

    Requires package ``kvxoptklu``.
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


class SciPySolver:
    """
    Base class for scipy family solvers.
    """
    def __init__(self):
        pass

    def to_csc(self, A):
        """
        Convert A to scipy.sparse.csc_matrix.

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


class CuPySolver(SciPySolver):
    """
    CuPy lsqr solver (GPU-based).
    """

    def solve(self, A, b):

        # delayed import for startup speed
        from cupyx.scipy.sparse import csc_matrix as csc_cu  # NOQA
        from cupyx.scipy.sparse.linalg import lsqr as cu_lsqr  # NOQA
        A_csc = self.to_csc(A)

        cu_A = csc_cu(A_csc)
        cu_b = cupy.array(np.array(b).ravel())
        x = cu_lsqr(cu_A, cu_b)

        return np.ravel(cupy.asnumpy(x[0]))


class SpSolve(SciPySolver):
    """
    scipy.sparse.linalg.spsolve Solver.
    """

    def solve(self, A, b):
        A_csc = self.to_csc(A)
        x = spsolve(A_csc, b)
        return np.ravel(x)
