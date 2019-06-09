from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


from cvxopt import umfpack, matrix

import numpy as np

try:
    from cvxoptklu import klu
    KLU = True
except ImportError:
    KLU = False


class Solver(object):
    """
    Sparse matrix solver class. Wraps UMFPACK and KLU with a unified interface.
    """
    def __init__(self, sparselib='umfpack'):
        self.sparselib = sparselib

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

    def solve(self, A, F, N, b):
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
            umfpack.linsolve(A, b)
            return b

        elif self.sparselib == 'klu':
            klu.linsolve(A, b)
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
                return matrix(x)

            elif self.sparselib == 'cupy':
                # delayed import for startup speed
                import cupy as cp  # NOQA
                from cupyx.scipy.sparse import csc_matrix as csc_cu  # NOQA
                from cupyx.scipy.sparse.linalg.solve import lsqr as cu_lsqr  # NOQA

                cu_A = csc_cu(A)
                cu_b = cp.array(np.array(b).reshape((-1,)))
                x = cu_lsqr(cu_A, cu_b)

                return matrix(cp.asnumpy(x[0]))
