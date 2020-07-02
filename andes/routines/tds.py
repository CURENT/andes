import sys
import os
import importlib
from collections import OrderedDict

from andes.routines.base import BaseRoutine
from andes.utils.misc import elapsed, is_notebook
from andes.utils.tab import Tab
from andes.shared import tqdm, np
from andes.shared import matrix, sparse, spdiag

import logging
logger = logging.getLogger(__name__)


class TDS(BaseRoutine):
    """
    Time-domain simulation routine.
    """
    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.config.add(OrderedDict((('tol', 1e-6),
                                     ('t0', 0.0),
                                     ('tf', 20.0),
                                     ('fixt', 1),
                                     ('shrinkt', 1),
                                     ('tstep', 1/30),
                                     ('max_iter', 15),
                                     )))
        self.config.add_extra("_help",
                              tol="convergence tolerance",
                              t0="simulation starting time",
                              tf="simulation ending time",
                              fixt="use fixed step size (1) or variable (0)",
                              shrinkt='shrink step size for fixed method if not converged',
                              tstep='the initial step step size',
                              max_iter='maximum number of iterations',
                              )
        self.config.add_extra("_alt",
                              tol="float",
                              t0=">=0",
                              tf=">t0",
                              fixt=(0, 1),
                              shrinkt=(0, 1),
                              tstep='float',
                              max_iter='>=10',
                              )
        # overwrite `tf` from command line
        if system.options.get('tf') is not None:
            self.config.tf = system.options.get('tf')

        # to be computed
        self.deltat = 0
        self.deltatmin = 0
        self.deltatmax = 0
        self.h = 0
        self.next_pc = 0
        self.eye = None
        self.Teye = None
        self.qg = np.array([])
        self.tol_zero = self.config.tol / 100

        # internal status
        self.converged = False
        self.busted = False
        self.err_msg = ''
        self.niter = 0
        self._switch_idx = 0  # index into `System.switch_times`
        self._last_switch_t = -999  # the last critical time
        self.mis = 1
        self.pbar = None
        self.callpert = None
        self.plotter = None
        self.plt = None
        self.initialized = False

    def init(self):
        """
        Initialize the status, storage and values for TDS.

        Returns
        -------
        array-like
            The initial values of xy.

        """
        t0, _ = elapsed()
        system = self.system

        if self.initialized:
            return system.dae.xy

        self._reset()
        self._load_pert()
        system.set_address(models=system.exist.tds)
        system.set_dae_names(models=system.exist.tds)

        system.dae.clear_ts()
        system.store_sparse_pattern(models=system.exist.pflow_tds)
        system.store_adder_setter(models=system.exist.pflow_tds)
        system.vars_to_models()
        system.init(system.exist.tds)
        system.store_switch_times(system.exist.tds)
        self.eye = spdiag([1] * system.dae.n)
        self.Teye = spdiag(system.dae.Tf.tolist()) * self.eye
        self.qg = np.zeros(system.dae.n + system.dae.m)
        self.calc_h()

        self.initialized = self.test_init()
        _, s1 = elapsed(t0)

        if self.initialized is True:
            logger.info(f"Initialization was successful in {s1}.")
        else:
            logger.error(f"Initialization failed in {s1}.")

        if system.dae.n == 0:
            tqdm.write('No dynamic component loaded.')
        return system.dae.xy

    def summary(self):
        """
        Print out a summary to logger.info.

        Returns
        -------

        """
        out = list()
        out.append('')
        out.append('-> Time Domain Simulation Summary:')
        out.append(f'Sparse Solver: {self.solver.sparselib.upper()}')
        out.append(f'Simulation time: {self.system.dae.t}-{self.config.tf}sec.')
        if self.config.fixt == 1:
            msg = f'Fixed step size: h={1000 * self.config.tstep:.4g}msec.'
            if self.config.shrinkt == 1:
                msg += ', shrink if not converged'
            out.append(msg)
        else:
            out.append(f'Variable step size: h0={1000 * self.config.tstep:.4g}msec.')

        out_str = '\n'.join(out)
        logger.info(out_str)

    def run(self, no_pbar=False, no_summary=False, **kwargs):
        """
        Run the implicit numerical integration for TDS.

        Parameters
        ----------
        no_pbar : bool
            True to disable progress bar
        no_summary : bool, optional
            True to disable the display of summary
        """
        system = self.system
        dae = self.system.dae
        config = self.config

        succeed = False
        resume = False

        if system.PFlow.converged is False:
            logger.warning('Power flow not solved. Simulation will not continue.')
            system.exit_code += 1
            return succeed

        if no_summary is False:
            self.summary()

        # only initializing at t=0 allows to continue when `run` is called again.
        if system.dae.t == 0:
            self.init()
        else:  # resume simulation
            resume = True

        self.pbar = tqdm(total=100, ncols=70, unit='%', file=sys.stdout, disable=no_pbar)

        if resume:
            perc = round((dae.t - config.t0) / (config.tf - config.t0) * 100, 0)
            self.next_pc = perc + 1
            self.pbar.update(perc)

        t0, _ = elapsed()

        while (system.dae.t < self.config.tf) and (not self.busted):
            if self.callpert is not None:
                self.callpert(dae.t, system)

            if self._itm_step():  # simulate the current step
                # store values
                dae.ts.store_txyz(dae.t.tolist(),
                                  dae.xy,
                                  self.system.get_z(models=system.exist.pflow_tds),
                                  )
                # check if the next step is critical time
                self.do_switch()

                if self.calc_h() == 0:
                    logger.error('Time step to zero...')
                    break

                dae.t += self.h

                # show progress in percentage
                perc = max(min((dae.t - config.t0) / (config.tf - config.t0) * 100, 100), 0)
                if perc >= self.next_pc:
                    self.pbar.update(1)
                    self.next_pc += 1

        self.pbar.close()
        delattr(self, 'pbar')  # removed `pbar` so that System object can be dilled

        if self.busted:
            logger.error(self.err_msg)
            logger.error(f"Simulation terminated at t={system.dae.t:.4f}.")
            system.exit_code += 1

        elif system.dae.t == self.config.tf:
            succeed = True   # success flag
            system.exit_code += 0
        else:
            system.exit_code += 1

        _, s1 = elapsed(t0)
        logger.info(f'Simulation completed in {s1}.')
        system.TDS.save_output()

        # load data into `TDS.plotter` in the notebook mode
        if is_notebook():
            self.load_plotter()

        return succeed

    def rewind(self, t):
        """
        TODO: rewind to a past time.
        """
        pass

    def load_plotter(self):
        """
        Manually load a plotter into ``TDS.plotter``.
        """
        from andes.plot import TDSData  # NOQA
        self.plotter = TDSData(mode='memory', dae=self.system.dae)
        self.plt = self.plotter

    def test_init(self):
        """
        Update f and g to see if initialization is successful.
        """
        system = self.system
        self._fg_update(system.exist.pflow_tds)
        system.j_update(models=system.exist.pflow_tds)

        # warn if variables are initialized at limits
        if system.config.warn_limits:
            for model in system.exist.pflow_tds.values():
                for item in model.discrete.values():
                    item.warn_init_limit()

        if np.max(np.abs(system.dae.fg)) < self.config.tol:
            logger.debug('Initialization tests passed.')
            return True
        else:

            fail_idx = np.where(abs(system.dae.fg) >= self.config.tol)
            fail_names = [system.dae.xy_name[int(i)] for i in np.ravel(fail_idx)]

            title = 'Suspect initialization issue! Simulation may crash!'
            err_data = {'Name': fail_names,
                        'Var. Value': system.dae.xy[fail_idx],
                        'Eqn. Mismatch': system.dae.fg[fail_idx],
                        }
            tab = Tab(title=title,
                      header=err_data.keys(),
                      data=list(map(list, zip(*err_data.values()))))

            logger.error(tab.draw())

            if system.options.get('verbose') == 1:
                breakpoint()
            system.exit_code += 1
            return False

    def _fg_update(self, models):
        """
        Update `f` and `g` equations.
        """
        system = self.system
        system.dae.clear_fg()
        system.l_update_var(models=models)
        system.s_update_var(models=models)  # update VarService
        system.f_update(models=models)
        system.g_update(models=models)
        system.l_update_eq(models=models)
        system.fg_to_dae()

    def _itm_step(self):
        """
        Integrate with Implicit Trapezoidal Method (ITM) to the current time.

        This function has an internal Newton-Raphson loop for algebraized semi-explicit DAE.
        The function returns the convergence status when done but does NOT progress simulation time.

        Returns
        -------
        bool
            Convergence status in ``self.converged``.

        """
        system = self.system
        dae = self.system.dae

        self.mis = 1
        self.niter = 0
        self.converged = False

        self.x0 = np.array(dae.x)
        self.y0 = np.array(dae.y)
        self.f0 = np.array(dae.f)

        while True:
            self._fg_update(models=system.exist.pflow_tds)

            # lazy Jacobian update
            if dae.t == 0 or self.niter > 3 or (dae.t - self._last_switch_t < 0.2):
                system.j_update(models=system.exist.pflow_tds)
                self.solver.factorize = True

            # TODO: set the `Tf` corresponding to the pegged anti-windup limiters to zero.
            # Although this should not affect anything since corr. mismatches in `self.qg` are reset to zero

            # solve implicit trapezoidal method (ITM) integration
            self.Ac = sparse([[self.Teye - self.h * 0.5 * dae.fx, dae.gx],
                              [-self.h * 0.5 * dae.fy, dae.gy]], 'd')

            # equation `self.qg[:dae.n] = 0` is the implicit form of differential equations using ITM
            self.qg[:dae.n] = dae.Tf * (dae.x - self.x0) - self.h * 0.5 * (dae.f + self.f0)

            # reset the corresponding q elements for pegged anti-windup limiter
            for item in system.antiwindups:
                for key, val in item.x_set:
                    np.put(self.qg, key, 0)

            self.qg[dae.n:] = dae.g

            if not self.config.linsolve:
                inc = self.solver.solve(self.Ac, matrix(self.qg))
            else:
                inc = self.solver.linsolve(self.Ac, matrix(self.qg))

            # check for np.nan first
            if np.isnan(inc).any():
                self.err_msg = 'NaN found in solution. Convergence not likely'
                self.niter = self.config.max_iter + 1
                self.busted = True
                break

            # reset small values to reduce chattering
            inc[np.where(np.abs(inc) < self.tol_zero)] = 0

            # set new values
            dae.x -= inc[:dae.n].ravel()
            dae.y -= inc[dae.n: dae.n + dae.m].ravel()

            system.vars_to_models()

            # calculate correction
            mis = np.max(np.abs(inc))
            if self.niter == 0:
                self.mis = mis

            self.niter += 1

            # converged
            if mis <= self.config.tol:
                self.converged = True
                break
            # non-convergence cases
            if self.niter > self.config.max_iter:
                logger.debug(f'Max. iter. {self.config.max_iter} reached for t={dae.t:.6f}, '
                             f'h={self.h:.6f}, mis={mis:.4g} ')

                # debug helpers
                g_max = np.argmax(abs(dae.g))
                inc_max = np.argmax(abs(inc))
                self._debug_g(g_max)
                self._debug_ac(inc_max)

                break
            if mis > 1000 and (mis > 1e8 * self.mis):
                self.err_msg = 'Error increased too quickly. Convergence not likely.'
                self.busted = True
                break

        if not self.converged:
            dae.x = np.array(self.x0)
            dae.y = np.array(self.y0)
            dae.f = np.array(self.f0)
            system.vars_to_models()

        return self.converged

    def _debug_g(self, y_idx):
        """
        Print out the associated variables with the given algebraic equation index.

        Parameters
        ----------
        y_idx
            Index of the equation into the `g` array. Diff. eqns. are not counted in.
        """
        y_idx = y_idx.tolist()
        logger.debug(f'Max. algebraic mismatch associated with {self.system.dae.y_name[y_idx]} [y_idx={y_idx}]')
        assoc_vars = self.system.dae.gy[y_idx, :]
        vars_idx = np.where(np.ravel(matrix(assoc_vars)))[0]

        logger.debug('')
        logger.debug(f'{"y_index":<10} {"Variable":<20} {"Derivative":<20}')
        for v in vars_idx:
            v = v.tolist()
            logger.debug(f'{v:<10} {self.system.dae.y_name[v]:<20} {assoc_vars[v]:<20g}')

        pass

    def _debug_ac(self, xy_idx):
        """
        Debug Ac matrix by printing out equations and derivatives associated with the max. mismatch variable.

        Parameters
        ----------
        xy_idx
            Index of the maximum mismatch into the `xy` array.
        """

        xy_idx = xy_idx.tolist()
        assoc_eqns = self.Ac[:, xy_idx]
        assoc_vars = self.Ac[xy_idx, :]

        eqns_idx = np.where(np.ravel(matrix(assoc_eqns)))[0]
        vars_idx = np.where(np.ravel(matrix(assoc_vars)))[0]

        logger.debug(f'Max. correction is for variable {self.system.dae.xy_name[xy_idx]} [{xy_idx}]')
        logger.debug(f'Associated equation value is {self.system.dae.fg[xy_idx]:<20g}.')
        logger.debug('')

        logger.debug(f'{"xy_index":<10} {"Equation":<20} {"Derivative":<20} {"Eq. Mismatch":<20}')
        for eq in eqns_idx:
            eq = eq.tolist()
            logger.debug(f'{eq:<10} {self.system.dae.xy_name[eq]:<20} {assoc_eqns[eq]:<20g} '
                         f'{self.system.dae.fg[eq]:<20g}')

        logger.debug('')
        logger.debug(f'{"xy_index":<10} {"Variable":<20} {"Derivative":<20} {"Eq. Mismatch":<20}')
        for v in vars_idx:
            v = v.tolist()
            logger.debug(f'{v:<10} {self.system.dae.xy_name[v]:<20} {assoc_vars[v]:<20g} '
                         f'{self.system.dae.fg[v]:<20g}')

    def save_output(self, npz=True):
        """
        Save the simulation data into two files: a lst file and a npz file.

        Parameters
        ----------
        npz : bool
            True to save in npz format; False to save in npy format.

        Returns
        -------
        bool
            True if files are written. False otherwise.
        """
        if self.system.files.no_output:
            return False
        else:
            t0, _ = elapsed()

            self.system.dae.write_lst(self.system.files.lst)

            if npz is True:
                self.system.dae.write_npz(self.system.files.npz)
            else:
                self.system.dae.write_npy(self.system.files.npy)

            _, s1 = elapsed(t0)
            logger.info(f'TDS outputs saved in {s1}.')
            return True

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

        if config.fixt and not config.shrinkt:
            if not self.converged:
                self.deltat = 0
                self.busted = True
                self.err_msg = f"Simulation did not converge with step size h={self.config.tstep:.4f}.\n"
                self.err_msg += "Reduce the step size `tstep`, or set `shrinkt = 1` to let it shrink."
        else:
            if self.converged:
                if self.niter >= 15:
                    self.deltat = max(self.deltat * 0.5, self.deltatmin)
                elif self.niter <= 6:
                    self.deltat = min(self.deltat * 1.1, self.deltatmax)
                else:
                    self.deltat = max(self.deltat * 0.95, self.deltatmin)

                # for converged cases, set step size back to the initial `config.tstep`
                if config.fixt:
                    self.deltat = min(config.tstep, self.deltat)
            else:
                self.deltat *= 0.9
                if self.deltat < self.deltatmin:
                    self.deltat = 0
                    self.err_msg = "Time step reduced to zero. Convergence not likely."
                    self.busted = True

        # last step size
        if system.dae.t + self.deltat > config.tf:
            self.deltat = config.tf - system.dae.t

        self.h = self.deltat
        # do not skip event switch_times
        if self._switch_idx < system.n_switches:
            if (system.dae.t + self.h) > system.switch_times[self._switch_idx]:
                self.h = system.switch_times[self._switch_idx] - system.dae.t
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
            freq = max(abs(As[0, 0]), 1)
        else:
            freq = 30.0

        if freq > system.config.freq:
            freq = float(system.config.freq)

        tspan = abs(config.tf - config.t0)
        tcycle = 1 / freq

        self.deltatmax = min(tcycle, tspan / 100.0)
        self.deltat = min(tcycle, tspan / 100.0)
        self.deltatmin = min(tcycle / 500, self.deltatmax / 20)

        if config.tstep <= 0:
            logger.warning('Fixed time step is negative or zero')
            logger.warning('Switching to automatic time step')
            config.fixt = False

        if config.fixt:
            self.deltat = config.tstep
            if config.tstep < self.deltatmin:
                logger.warning('Fixed time step is smaller than the estimated minimum.')

        self.h = self.deltat
        return self.h

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
        if self._switch_idx < self.system.n_switches:
            if abs(self.system.dae.t - self.system.switch_times[self._switch_idx]) < 1e-8:
                ret = True
        return ret

    def do_switch(self):
        """Perform switch if is switch time"""
        ret = False

        system = self.system
        if self.is_switch_time():
            self._last_switch_t = system.switch_times[self._switch_idx]
            system.switch_action(system.exist.pflow_tds)
            self._switch_idx += 1
            system.vars_to_models()

            ret = True

        return ret

    def _reset(self):
        # reset states
        self.deltat = 0
        self.deltatmin = 0
        self.deltatmax = 0
        self.h = 0
        self.next_pc = 0.1
        self.eye = None
        self.Teye = None
        self.qg = np.array([])

        self.converged = False
        self.busted = False
        self.niter = 0
        self._switch_idx = 0       # index into `System.switch_times`
        self._last_switch_t = -999  # the last critical time
        self.mis = 1
        self.system.dae.t = np.array(0.0)
        self.pbar = None
        self.plotter = None
        self.plt = None             # short name for `plotter`

        self.initialized = False

    def _load_pert(self):
        """
        Load perturbation files to ``self.callpert``.
        """
        system = self.system
        if system.files.pert:
            if not os.path.isfile(system.files.pert):
                logger.warning(f'Pert file not found at <{system.files.pert}>.')
                return False

            sys.path.append(system.files.case_path)
            _, full_name = os.path.split(system.files.pert)
            name, ext = os.path.splitext(full_name)

            module = importlib.import_module(name)
            self.callpert = getattr(module, 'pert')
            logger.info(f'Perturbation file <{system.files.pert}> loaded.')
            return True

    def _fg_wrapper(self, xy):
        system = self.system
        system.dae.x[:] = xy[:system.dae.n]
        system.dae.y[:] = xy[system.dae.n:]
        system.vars_to_models()

        self._fg_update(system.exist.pflow_tds)

        return system.dae.fg
