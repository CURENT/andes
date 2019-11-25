import numpy as np  # NOQA
from collections import OrderedDict
from andes.programs.base import ProgramBase
from cvxopt import matrix, sparse  # NOQA
from scipy.optimize import fsolve, newton_krylov
from scipy.optimize.nonlin import NoConvergence
from scipy.integrate import solve_ivp

import logging
logger = logging.getLogger(__name__)


class TDS(ProgramBase):

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.config.add(OrderedDict((('tol', 1e-6),
                                     ('tf', 20.0),
                                     ('fixt', 1),
                                     ('tstep', 1/30),
                                     ('max_iter', 20))))
        self.tds_models = system.get_models_with_flag('tds')
        self.pflow_tds_models = system.get_models_with_flag(('tds', 'pflow'))
        self.converged = False
        self.niter = 0
        self.mis = []

    def _initialize(self):
        system = self.system

        system.set_address(models=self.tds_models)
        system.set_dae_names(models=self.tds_models)
        system.dae.resize_array()
        system.link_external(models=self.tds_models)
        system.store_sparse_pattern(models=self.tds_models)
        system.store_adder_setter()
        return system.initialize(self.tds_models, tds=True)

    def f_update(self):
        system = self.system
        # evaluate limiters, differential, algebraic, and jacobians
        system.vars_to_models()
        system.e_clear(models=self.pflow_tds_models)
        system.l_update_var(models=self.pflow_tds_models)
        system.f_update(models=self.pflow_tds_models)
        system.l_update_eq(models=self.pflow_tds_models)

    def g_update(self):
        system = self.system
        system.g_update(models=self.pflow_tds_models)

    def _g_wrapper(self, y):
        """
        Wrapper for algebraic equations for general solver

        Parameters
        ----------
        y

        Returns
        -------

        """
        system = self.system
        system.dae.y = y
        system.vars_to_models()
        system.e_clear(models=self.pflow_tds_models)
        system.l_update_var(models=self.pflow_tds_models)
        system.g_update(models=self.pflow_tds_models)
        system.l_update_eq(self.pflow_tds_models)
        return system.dae.g

    def _solve_g(self):
        system = self.system
        self.converged = False
        self.niter = 0
        self.mis = []
        while True:
            inc = -matrix(self._g_wrapper(system.dae.y))
            system.j_update(models=self.pflow_tds_models)
            inc = self.solver.linsolve(system.dae.gy, inc)
            system.dae.y += np.ravel(np.array(inc))
            system.vars_to_models()

            mis = np.max(np.abs(system.dae.g))
            print(f'{system.dae.t}, {inc}, {mis}')
            self.mis.append(mis)
            if mis < self.config.tol:
                self.converged = True
                break
            elif self.niter > self.config.max_iter:
                raise NoConvergence(f'Convergence not reached after {self.config.max_iter} iterations')
            # elif mis > 1e4 * self.mis[0]:
            #     raise NoConvergence('Mismatch increased too fast. Convergence not likely.')
            self.niter += 1

    def _solve_ivp_wrapper(self, t, xy, asolver, verbose):
        system = self.system
        dae = self.system.dae
        dae.t = t
        dae.x = xy[:dae.n]

        # solve for algebraic variables
        if asolver is None:
            self._solve_g()
        elif asolver == 'fsolve':
            sol, _, ier, mesg = fsolve(self._g_wrapper, dae.y, full_output=True)
            if ier != 1:
                raise NoConvergence(f"Cannot solve algebraic equations, error: \n{mesg}")
            dae.y = sol
        elif asolver == 'newton_krylov':
            dae.y = newton_krylov(self._g_wrapper, dae.y, verbose=verbose)
        else:
            raise NotImplementedError(f"Unknown algeb_solver {asolver}")

        system.f_update(models=self.pflow_tds_models)
        system.l_update_eq(models=self.pflow_tds_models)

        return np.hstack((dae.f, np.zeros(dae.g.shape)))

    def simulate(self, tspan, y0=None, asolver=None, verbose=False):
        self._initialize()
        if y0 is None:
            y0 = self.system.dae.xy
        return solve_ivp(lambda t, y: self._solve_ivp_wrapper(t, y, asolver, verbose=verbose),
                         tspan, y0, method='LSODA')

    def nr(self):
        """
        Full Newton-Raphson method

        Returns
        -------

        """
        system = self.system
        self.niter = 0
        self.mis = []
        self.converged = False
        while True:
            # evaluate limiters, differential, algebraic, and jacobians
            system.e_clear(self.pflow_tds_models)
            system.l_update_var(self.pflow_tds_models)
            system.g_update(self.pflow_tds_models)
            system.l_update_eq(self.pflow_tds_models)
            system.j_update(self.pflow_tds_models)

            # prepare and solve linear equations
            inc = -matrix(system.dae.g)
            A = sparse(system.dae.gy)
            inc = self.solver.solve(A, inc)
            system.dae.y += np.ravel(np.array(inc))

            system.c_update(self.pflow_tds_models)
            mis = np.max(np.abs(system.dae.g))
            self.mis.append(mis)

            system.vars_to_models()

            if mis < self.config.tol:
                self.converged = True
                break
            elif self.niter > self.config.max_iter:
                break
            elif mis > 1e4 * self.mis[0]:
                logger.error('Mismatch increased too fast. Convergence not likely.')
                break
            self.niter += 1

        if not self.converged:
            if abs(self.mis[-1] - self.mis[-2]) < self.config.tol:
                max_idx = np.argmax(np.abs(system.dae.xy))
                name = system.dae.xy_name[max_idx]
                logger.error('Mismatch is not correctable possibly due to large load-generation imbalance.')
                logger.info(f'Largest mismatch on equation associated with <{name}>')

        return self.converged
