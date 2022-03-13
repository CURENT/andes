from andes.linsolvers.cupy import CuPySolver
from andes.linsolvers.scipy import SpSolve
from andes.linsolvers.suitesparse import UMFPACKSolver, KLUSolver


class Solver:
    """
    Sparse matrix solver class.

    This class wraps UMFPACK, KLU, SciPy and CuPy solvers to provide an unified
    interface for solving sparse linear equations ``Ax = b``.

    Provides methods ``solve``, ``linsolve`` and ``clear``.
    """

    def __init__(self, sparselib='umfpack'):

        # solvers
        self.umfpack = UMFPACKSolver()
        self.klu = KLUSolver()
        self.spsolve = SpSolve()
        self.cupy = CuPySolver()

        # KLU as failsafe
        if sparselib not in self.__dict__:
            self.sparselib = 'klu'
        else:
            self.sparselib = sparselib

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
