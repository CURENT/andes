import numpy as np
from andes.programs.base import ProgramBase
from cvxopt import matrix, sparse
from scipy.optimize import newton_krylov

import logging
logger = logging.getLogger(__name__)


class PFlow(ProgramBase):

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.config.add(tol=1e-6, max_iter=20)
        self.models = system.get_models_with_flag('pflow')

        self.converged = False
        self.inc = None
        self.A = None
        self.niter = None
        self.mis = []

    def _initialize(self):
        self.converged = False
        self.inc = None
        self.A = None
        self.niter = None
        self.mis = []
        return self.system.initialize(self.models)

    def nr_step(self):
        """
        Single stepping for Newton Raphson method
        Returns
        -------

        """
        system = self.system
        # evaluate limiters, differential, algebraic, and jacobians
        system.l_update()
        system.f_update()
        system.g_update()
        system.j_update()

        # prepare and solve linear equations
        self.inc = -matrix([matrix(system.dae.f),
                            matrix(system.dae.g)])

        self.A = sparse([[system.dae.fx, system.dae.gx],
                         [system.dae.fy, system.dae.gy]])

        self.inc = self.solver.solve(self.A, self.inc)

        system.dae.x += np.ravel(np.array(self.inc[:system.dae.n]))
        system.dae.y += np.ravel(np.array(self.inc[system.dae.n:]))

        mis = max(abs(matrix([matrix(system.dae.f), matrix(system.dae.g)])))
        self.mis.append(mis)

        system.vars_to_models()
        system.e_clear()

        return mis

    def nr(self):
        """
        Full Newton-Raphson method

        Returns
        -------

        """
        self._initialize()
        self.niter = 0
        while True:
            mis = self.nr_step()
            logger.info(f'{self.niter}:  |F(x)| = {mis:10g}')

            if mis < self.config.tol or self.niter > self.config.max_iter:
                self.converged = True
                break
            if mis > 1e4 * self.mis[0]:
                logger.error(f'Mismatch increase too much. Convergence not likely.')
                break
            self.niter += 1

        return self.converged

    def _g_wrapper(self, y):
        """
        Wrapper for algebraic equations to be used with Newton-Krylov general solver

        Parameters
        ----------
        y

        Returns
        -------

        """
        system = self.system
        system.dae.y = y
        system.vars_to_models()
        system.e_clear()

        system.l_update()
        system.f_update()
        g = system.g_update()

        return g

    def newton_krylov(self, verbose=False):
        """
        Full Newton-Krylov method

        Warnings
        --------
        The result might be wrong if limiters are in use!

        Parameters
        ----------
        verbose

        Returns
        -------

        """
        system = self.system
        system.initialize()
        v0 = system.dae.y
        return newton_krylov(self._g_wrapper, v0, verbose=verbose)
