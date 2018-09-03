from cvxopt import umfpack

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

        elif self.sparselib == 'klu':
            klu.solve(A, F, N, b)

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
            return umfpack.linsolve(A, b)

        elif self.sparselib == 'klu':
            return klu.linsolve(A, b)
