"""
CuPy solver that requires the ``cupy`` package.
"""

import numpy as np

from andes.shared import cupy
from andes.linsolvers.scipy import SciPySolver


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
