import numpy as np  # NOQA
from collections import OrderedDict
from andes.programs.base import ProgramBase
from cvxopt import matrix, sparse  # NOQA
from scipy.optimize import fsolve, newton_krylov
from scipy.optimize.nonlin import NoConvergence
from scipy.integrate import solve_ivp, odeint

import logging
logger = logging.getLogger(__name__)


class TDSOut(object):
    def __init__(self, dae):
        self.t = np.zeros(0)


class TDS(ProgramBase):
    # TODO: implement implicit integration routine

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

    def _g_wrapper(self, y=None):
        """
        Wrapper for algebraic equations for general solver

        Parameters
        ----------
        y

        Returns
        -------

        """
        system = self.system
        if y is not None:
            system.dae.y = y
            system.vars_to_models()
        system.e_clear(models=self.pflow_tds_models)
        system.l_update_var(models=self.pflow_tds_models)
        system.g_update(models=self.pflow_tds_models)
        return system.dae.g

    def _solve_g(self, verbose):
        system = self.system
        dae = system.dae
        self.converged = False
        self.niter = 0
        self.mis = []

        while True:
            self._g_wrapper(dae.y)
            inc = -matrix(system.dae.g)
            system.j_update(models=self.pflow_tds_models)
            inc = self.solver.solve(dae.gy, inc)
            dae.y += np.ravel(np.array(inc))
            system.vars_to_models()

            mis = np.max(np.abs(inc))
            self.mis.append(mis)
            if verbose:
                print(f't={dae.t:<.4g}, iter={self.niter:<g}, mis={mis:<.4g}')
            if mis < self.config.tol:
                self.converged = True
                break
            elif self.niter > self.config.max_iter:
                raise NoConvergence(f'Convergence not reached after {self.config.max_iter} iterations')
            elif mis >= 1000 and (mis > 1e4 * self.mis[0]):
                raise NoConvergence('Mismatch increased too fast. Convergence not likely.')

            self.niter += 1

    def _solve_ivp_wrapper(self, t, xy, asolver, verbose):
        system = self.system
        dae = self.system.dae
        dae.set_t(t)
        dae.x = xy[:dae.n]
        dae.y = xy[dae.n:dae.m + dae.n]

        # solve for algebraic variables
        if asolver is None:
            self._solve_g(verbose)
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

        return np.hstack((dae.f, np.zeros_like(dae.g)))

    def simulate(self, tspan, xy0=None, asolver=None, method='RK45', verbose=False):
        self._initialize()
        if xy0 is None:
            xy0 = self.system.dae.xy
        return solve_ivp(lambda t, y: self._solve_ivp_wrapper(t, y, asolver, verbose=verbose),
                         tspan, xy0, method=method)

    def integrate(self, tspan, xy0=None, asolver=None, verbose=False, h=0.05, hmax=0, hmin=0):
        self._initialize()
        if xy0 is None:
            xy0 = self.system.dae.xy
        times = np.arange(tspan[0], tspan[1], h)
        return times, odeint(self._solve_ivp_wrapper,
                             xy0,
                             times,
                             tfirst=True,
                             args=(asolver, verbose),
                             full_output=True,
                             hmax=hmax,
                             hmin=hmin)

    def _residual(self, t, xy, xydot, result):
        """
        Resudual function to be passed to `scikits.odes`
        Parameters
        ----------
        t
        xy
        xydot
        result

        Returns
        -------

        """
        system = self.system
        dae = self.system.dae
        dae.t = t
        dae.x = xy[:dae.n]
        dae.y = xy[dae.n:dae.m + dae.n]

        self._g_wrapper(dae.y)
        system.f_update(models=self.pflow_tds_models)
        system.l_update_eq(models=self.pflow_tds_models)

        result[:dae.n] = dae.f - xydot[:dae.n]
        result[dae.n:dae.n + dae.m] = dae.g

    def _residual_assimulo(self, t, xy, xydot):
        """
        Residual function to be passed to assimulo
        Parameters
        ----------
        t
        xy
        xydot

        Returns
        -------

        """
        result = np.zeros_like(xy)
        self._residual(t, xy, xydot, result)
        return result

    def odes_solve(self, tspan, xy0=None, xyp0=None, method='ida', verbose=False, h=0.05):
        """
        Use scikits.odes to solve the DAE

        Warnings
        --------
        Not working yet

        Parameters
        ----------
        tspan
        xy0
        xyp0
        method
        verbose
        h

        Returns
        -------

        """
        from scikits.odes.dae import dae
        system = self.system

        self._initialize()
        if xy0 is None:
            xy0 = self.system.dae.xy
        xyp0 = np.zeros_like(xy0) if xyp0 is None else xyp0
        algeb_idx = list(range(system.dae.n, system.dae.n + system.dae.m))
        times = np.arange(tspan[0], tspan[1], h)

        solver = dae(method,
                     self._residual,
                     first_step_size=1e-18,
                     atol=1e-6,
                     rtol=1e-6,
                     algebraic_vars_idx=algeb_idx,
                     # compute_initcond='yp0',
                     # compute_initcond_t0=20,
                     old_api=False
                     )
        return solver.solve(times, xy0, xyp0)

    def assimulo_solve(self, tspan, xy0=None, xyp0=None, h=0.05):
        from assimulo.problem import Implicit_Problem
        from assimulo.solvers import ODASSL

        self._initialize()
        if xy0 is None:
            xy0 = self.system.dae.xy

        tfinal = tspan[1]
        ncp = (tspan[1] - tspan[0]) / h

        xyp0 = np.zeros_like(xy0) if xyp0 is None else xyp0
        model = Implicit_Problem(self._residual_assimulo, xy0, xyp0, tspan[0])  # Create an Assimulo problem
        model.name = 'TDS'
        sim = ODASSL(model)

        t, y, yd = sim.simulate(tfinal, ncp)
        # Use the .simulate method to simulate and provide the final time and ncp (optional)
        sim.plot()
        return t, y, yd
