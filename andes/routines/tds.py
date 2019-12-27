import numpy as np  # NOQA
from collections import OrderedDict
from andes.routines.base import BaseRoutine
from andes.common.utils import elapsed
from cvxopt import matrix, sparse, spdiag  # NOQA
from scipy.optimize import fsolve, newton_krylov
from scipy.optimize.nonlin import NoConvergence
from scipy.integrate import solve_ivp, odeint
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


class TDS(BaseRoutine):

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.config.add(OrderedDict((('tol', 1e-6),
                                     ('t0', 0.0),
                                     ('tf', 20.0),
                                     ('fixt', 1),
                                     ('tstep', 1/30),  # suggested step size
                                     ('max_iter', 20),
                                     )))
        # overwrite `tf` from command line
        if system.options.get('tf') is not None:
            self.config.tf = system.options.get('tf')

        self.tds_models = system.get_models_with_flag('tds')
        self.pflow_tds_models = system.get_models_with_flag(('tds', 'pflow'))

        # to be computed
        self.deltat = 0
        self.deltatmin = 0
        self.deltatmax = 0
        self.h = 0
        self.next_pc = 0

        self.converged = False
        self.busted = False
        self.niter = 0
        self._switch_idx = -1  # index into `System.switch_times`
        self._last_switch_t = -999  # the last critical time
        self.mis = []
        self.pbar = None
        self.plotter = None

        self.initialized = False

    def _initialize(self):
        """
        Initialize the status, storage and values for TDS.

        Returns
        -------
        array-like
            The initial values of xy.

        """
        t0, _ = elapsed()
        system = self.system
        self._reset()
        system.set_address(models=self.tds_models)
        system.set_dae_names(models=self.tds_models)

        system.dae.resize_array()
        system.dae.clear_ts()
        system.store_sparse_pattern(models=self.pflow_tds_models)
        system.store_adder_setter(models=self.pflow_tds_models)
        system.vars_to_models()
        system.initialize(self.tds_models)
        system.store_switch_times(self.tds_models)
        self.initialized = True

        _, s1 = elapsed(t0)
        self.pbar.write(f"Initialization completed in {s1}.")
        return system.dae.xy

    def run_implicit(self, verbose=False):
        """
        Run the implicit numerical integration for TDS.

        Parameters
        ----------
        verbose : bool
            verbosity flag for single integration steps
        """
        system = self.system
        dae = self.system.dae
        config = self.config

        logger.info('Time Domain Simulation:')
        self._initialize()

        t0, _ = elapsed()
        while system.dae.t < self.config.tf and (not self.busted):
            if self.calc_h() == 0:
                logger.error("Time step calculated to zero. Simulation terminated.")
                break

            if self._implicit_step(verbose):
                # store values
                dae.store_yt_single()
                dae.store_x_single()

                dae.t += self.h

                # show progress in percentage
                perc = max(min((dae.t - config.t0) / (config.tf - config.t0) * 100, 100), 0)
                if perc >= self.next_pc:
                    self.pbar.update(1)
                    self.next_pc += 1

            # check if the next step is critical time
            if self.is_switch_time():
                self._last_switch_t = system.switch_times[self._switch_idx]
                system.switch_action(self.pflow_tds_models)

        self.pbar.close()
        _, s1 = elapsed(t0)
        logger.info(f'Simulation completed in {s1}.')

        # load data into ``TDS.plot``
        from andes.plot import TDSData  # NOQA
        self.plotter = TDSData(mode='memory', dae=system.dae)

    def _implicit_step(self, verbose=False):
        """
        Integrate for a single given step.

        This function has an internal Newton-Raphson loop for algebraized semi-explicit DAE.
        The function returns the convergence status when done but does NOT progress simulation time.

        Returns
        -------
        bool
            Convergence status in ``self.converged``.

        """
        system = self.system
        dae = self.system.dae

        self.mis = []
        self.niter = 0
        self.converged = False

        self.x0 = np.array(dae.x)
        self.y0 = np.array(dae.y)
        self.f0 = np.array(dae.f)

        while True:
            system.e_clear(models=self.pflow_tds_models)
            system.l_update_var(models=self.pflow_tds_models)
            system.f_update(models=self.pflow_tds_models)
            system.g_update(models=self.pflow_tds_models)
            system.l_update_eq(models=self.pflow_tds_models)
            # lazy jacobian update
            if dae.t == 0 or self.niter > 3 or (dae.t - self._last_switch_t < 0.2):
                system.j_update(models=self.pflow_tds_models)

            # solve
            In = spdiag([1] * dae.n)
            self.Ac = sparse([[In - self.h * 0.5 * dae.fx, dae.gx],
                              [-self.h * 0.5 * dae.fy, dae.gy]], 'd')
            q = dae.x - self.x0 - self.h * 0.5 * (dae.f + self.f0)
            qg = np.hstack((q, dae.g))

            # set new values
            inc = self.solver.solve(self.Ac, -matrix(qg))
            dae.x += np.ravel(np.array(inc[:dae.n]))
            dae.y += np.ravel(np.array(inc[dae.n: dae.n + dae.m]))
            system.vars_to_models()

            # calculate correction
            mis = np.max(np.abs(inc))
            self.mis.append(mis)
            self.niter += 1

            if mis <= self.config.tol:
                self.converged = True
                break
            if np.isnan(inc).any():
                logger.error(f'NaN found in solution. Convergence not likely')
                self.niter = self.config.max_iter + 1
                self.busted = True
                break
            if self.niter > 5:
                logger.debug('debug - niter > 5')
            if self.niter > self.config.max_iter:
                logger.debug(f'Maximum iteration {self.config.max_iter} reached for t={dae.t}, h={self.h:.4f}')
                break
            if mis > 1000 and (mis > 1e4 * self.mis[0]):
                logger.error(f'Error increased too quickly. Convergence not likely.')
                self.busted = True
                break

        if not self.converged:
            dae.x = np.array(self.x0)
            dae.y = np.array(self.y0)
            dae.f = np.array(self.f0)
            system.vars_to_models()

            if verbose:
                logger.info(f'  Not converged, time={dae.t:.4f}s, h={self.h:.4f}, mis={mis:.4g}')

        return self.converged

    def save_output(self):
        if self.system.files.no_output:
            return False
        else:
            t0, _ = elapsed()
            self.system.dae.write_lst(self.system.files.lst)
            self.system.dae.write_npy(self.system.files.npy)
            _, s1 = elapsed(t0)
            logger.info(f'TDS outputs saved in {s1}.')
            return True

    def _run_odeint(self, tspan, x0=None, asolver=None, verbose=False, h=0.05, hmax=0, hmin=0):
        """
        Run integration with ``scipy.odeint``.

        Warnings
        --------
        Function is NOT working. The time-based switching is not handled correctly.
        """
        self._initialize()
        if x0 is None:
            x0 = self.system.dae.x
        times = np.arange(tspan[0], tspan[1], h)

        # build critical time list
        tcrit = np.hstack([np.linspace(i, i+0.5, 100) for i in self.system.switch_times])
        ret = odeint(self._solve_ivp_wrapper,
                     x0,
                     times,
                     tfirst=True,
                     args=(asolver, verbose),
                     full_output=True,
                     hmax=hmax,
                     hmin=hmin,
                     tcrit=tcrit
                     )

        # store the last step algebraic variables
        self.system.dae.store_yt_single()
        self.system.dae.store_xt_array(ret[0], times)
        return ret

    def _run_solve_ivp(self, tspan, x0=None, asolver=None, method='RK45', verbose=False):
        """
        Run integration with ``scipy.solve_ivp``.

        Warnings
        --------
        Function not fully implemented. Discontinuities are not properly handled by this wrapper.

        """
        self._initialize()
        if x0 is None:
            x0 = self.system.dae.x
        ret = solve_ivp(lambda t, x: self._solve_ivp_wrapper(t, x, asolver, verbose=verbose),
                        tspan,
                        x0,
                        method=method)

        # store the last step algebraic variables
        self.system.dae.store_yt_single()

        self.system.dae.store_xt_array(np.transpose(ret.y), ret.t)
        return ret

    def calc_h(self):
        """
        Calculate the time step size during the TDS.

        Notes
        -----
        A heuristic function is used for variable time step size ::

                 min(0.50 * h, hmin), if niter >= 15
            h =  max(1.10 * h, hmax), if niter <= 6
                 min(0.95 * h, hmin), otherwise

        Returns
        -------
        float
            computed time step size stored in ``self.h``
        """
        system = self.system
        config = self.config

        if system.dae.t == 0:
            return self._calc_h_first()

        if self.converged:
            if self.niter >= 15:
                self.deltat = max(self.deltat * 0.5, self.deltatmin)
            elif self.niter <= 6:
                self.deltat = min(self.deltat * 1.1, self.deltatmax)
            else:
                self.deltat = max(self.deltat * 0.95, self.deltatmin)

            # adjust fixed time step if niter is high
            if config.fixt:
                self.deltat = min(config.tstep, self.deltat)
        else:
            self.deltat *= 0.9
            if self.deltat < self.deltatmin:
                self.deltat = 0

        # last step size
        if system.dae.t + self.deltat > config.tf:
            self.deltat = config.tf - system.dae.t

        self.h = self.deltat
        # do not skip event switch_times
        if self._has_more_switch():
            if (system.dae.t + self.h) > system.switch_times[self._switch_idx + 1]:
                self.h = system.switch_times[self._switch_idx + 1] - system.dae.t
        return self.h

    def _calc_h_first(self):
        """
        Compute the first time step and save to ``self.h``.
        """
        system = self.system
        config = self.config

        if not system.dae.n:
            freq = 1.0
        elif system.dae.n == 1:
            B = matrix(system.dae.gx)
            self.solver.linsolve(system.dae.gy, B)
            As = system.dae.fx - system.dae.fy * B
            freq = abs(As[0, 0])
        else:
            freq = 20.0

        if freq > system.config.freq:
            freq = float(system.config.freq)

        tspan = abs(config.tf - config.t0)
        tcycle = 1 / freq

        self.deltatmax = min(3 * tcycle, tspan / 100.0)
        self.deltat = min(tcycle, tspan / 100.0)
        self.deltatmin = min(tcycle / 64, self.deltatmax / 20)

        if config.tstep <= 0:
            logger.warning('Fixed time step is negative or zero')
            logger.warning('Switching to automatic time step')
            config.fixt = False

        if config.fixt:
            self.deltat = config.tstep
            if config.tstep < self.deltatmin:
                logger.warning('Fixed time step is smaller than the estimated minimum')

        self.h = self.deltat
        return self.h

    def _has_more_switch(self):
        """
        Check if there are more switching events in the ``System.switch_times`` list.
        """
        ret = False
        if len(self.system.switch_times) > 0:
            if self._switch_idx + 1 < len(self.system.switch_times):
                ret = True
        return ret

    def is_switch_time(self):
        """
        Return if the current time is a switching time for time domain simulation.

        Time is approximated with a tolerance of 1e-8.

        Returns
        -------
        bool
            ``True`` if is a switching time; ``False`` otherwise.
        """
        ret = False
        if self._has_more_switch():
            next_idx = self._switch_idx + 1
            if abs(self.system.dae.t - self.system.switch_times[next_idx]) < 1e-8:
                ret = True
                self._switch_idx += 1
        return ret

    def _reset(self):
        # reset states
        self.deltat = 0
        self.deltatmin = 0
        self.deltatmax = 0
        self.h = 0
        self.next_pc = 0.1

        self.converged = False
        self.busted = False
        self.niter = 0
        self._switch_idx = -1  # index into `System.switch_times`
        self._last_switch_t = -999  # the last critical time
        self.mis = []
        self.system.dae.t = 0.0
        self.pbar = tqdm(total=100, ncols=80, unit='%')
        self.plotter = None

        self.initialized = False

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

        # check if the next step is critical time
        if self.is_switch_time():
            self._last_switch_t = system.switch_times[self._switch_idx]
            system.switch_action(self.pflow_tds_models)

        while True:
            system.e_clear(models=self.pflow_tds_models)
            system.l_update_var(models=self.pflow_tds_models)
            system.g_update(models=self.pflow_tds_models)

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

    def _solve_ivp_wrapper(self, t, x, asolver, verbose):
        system = self.system
        dae = self.system.dae
        # store the values from k-1 (the last step)
        dae.x = x
        system.vars_to_models()
        system.dae.store_yt_single()
        # set new t must come after `store_xyt`
        dae.t = t

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

        return dae.f
