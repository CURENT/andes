import sys
import os
import importlib
from collections import OrderedDict

from andes.routines.base import BaseRoutine
from andes.utils.misc import elapsed, is_notebook
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
        self.config.add(OrderedDict((('tol', 1e-4),
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

        self.tds_models = system.find_models('tds')
        self.pflow_tds_models = system.find_models(('tds', 'pflow'))

        # to be computed
        self.deltat = 0
        self.deltatmin = 0
        self.deltatmax = 0
        self.h = 0
        self.next_pc = 0

        # internal status
        self.converged = False
        self.busted = False
        self.niter = 0
        self._switch_idx = -1  # index into `System.switch_times`
        self._last_switch_t = -999  # the last critical time
        self.mis = []
        self.pbar = None
        self.callpert = None
        self.plotter = None
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
        self._reset()
        self._load_pert()
        system.set_address(models=self.tds_models)
        system.set_dae_names(models=self.tds_models)

        system.dae.resize_array()
        system.dae.clear_ts()
        system.store_sparse_pattern(models=self.pflow_tds_models)
        system.store_adder_setter(models=self.pflow_tds_models)
        system.vars_to_models()
        system.init(self.tds_models)
        system.store_switch_times(self.tds_models)
        self.eye = spdiag([1] * system.dae.n)
        self.Teye = spdiag(system.dae.zf.tolist()) * self.eye

        self.initialized = self.test_initialization()
        _, s1 = elapsed(t0)

        if self.initialized is True:
            logger.info(f"Initialization was successful in {s1}.")
        else:
            logger.info(f"Initialization failed in {s1}.")

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
        out.append(f'Simulation time: {self.config.t0}-{self.config.tf}sec.')
        if self.config.fixt == 1:
            msg = f'Fixed step size: h={1000 * self.config.tstep:.4g}msec.'
            if self.config.shrinkt == 1:
                msg += ', shrink if not converged'
            out.append(msg)
        else:
            out.append(f'Variable step size: h0={1000 * self.config.tstep:.4g}msec.')

        out_str = '\n'.join(out)
        logger.info(out_str)

    def run(self, disable_pbar=False, **kwargs):
        """
        Run the implicit numerical integration for TDS.

        Parameters
        ----------
        disable_pbar : bool
            True to disable progress bar
        """
        system = self.system
        dae = self.system.dae
        config = self.config

        ret = False
        if system.PFlow.converged is False:
            logger.warning('Power flow not solved. Simulation will not continue.')
            return ret

        self.summary()
        self.init()
        self.pbar = tqdm(total=100, ncols=70, unit='%', file=sys.stdout, disable=disable_pbar)

        t0, _ = elapsed()
        while (system.dae.t < self.config.tf) and (not self.busted):
            if self.calc_h() == 0:
                self.pbar.close()
                logger.error(f"Simulation terminated at t={system.dae.t:.4f}.")
                ret = False   # FIXME: overwritten
                break

            if self.callpert is not None:
                self.callpert(dae.t, system)

            if self._implicit_step():
                # store values
                dae.ts.store_txyz(dae.t, dae.xy, self.system.get_z(models=self.pflow_tds_models))
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
                system.vars_to_models()

        self.pbar.close()
        delattr(self, 'pbar')  # removed `pbar` so that System can be dilled

        _, s1 = elapsed(t0)
        logger.info(f'Simulation completed in {s1}.')
        system.TDS.save_output()
        ret = True

        # load data into ``TDS.plotter`` in the notebook mode
        if is_notebook():
            self.load_plotter()

        return ret

    def load_plotter(self):
        """
        Manually load a plotter into ``TDS.plotter``.
        """
        from andes.plot import TDSData  # NOQA
        self.plotter = TDSData(mode='memory', dae=self.system.dae)

    def test_initialization(self):
        """
        Update f and g to see if initialization is successful.
        """
        system = self.system
        system.e_clear(models=self.pflow_tds_models)
        system.l_update_var(models=self.pflow_tds_models)
        system.f_update(models=self.pflow_tds_models)
        system.g_update(models=self.pflow_tds_models)
        system.l_check_eq(models=self.pflow_tds_models)
        system.l_set_eq(models=self.pflow_tds_models)
        system.fg_to_dae()
        system.j_update(models=self.pflow_tds_models)

        if np.max(np.abs(system.dae.fg)) < self.config.tol:
            logger.debug('Initialization tests passed.')
            return True
        else:
            logger.error('Suspect initialization issue!')
            fail_idx = np.where(abs(system.dae.fg) >= self.config.tol)
            fail_names = [system.dae.xy_name[int(i)] for i in np.ravel(fail_idx)]
            logger.error(f"Check variables {', '.join(fail_names)}")
            return False

    def _implicit_step(self):
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
            system.l_check_eq(models=self.pflow_tds_models)
            system.l_set_eq(models=self.pflow_tds_models)
            system.fg_to_dae()

            # lazy jacobian update
            if dae.t == 0 or self.niter > 3 or (dae.t - self._last_switch_t < 0.2):
                system.j_update(models=self.pflow_tds_models)
                self.solver.factorize = True

            # solve implicit trapezoidal method (ITM) integration
            self.Ac = sparse([[self.Teye - self.h * 0.5 * dae.fx, dae.gx],
                              [-self.h * 0.5 * dae.fy, dae.gy]], 'd')
            # equation `q = 0` is the implicit form of differential equations using ITM
            q = dae.zf * (dae.x - self.x0) - self.h * 0.5 * (dae.f + self.f0)

            # reset the corresponding q elements for pegged anti-windup limiter
            for item in system.antiwindups:
                if len(item.x_set) > 0:
                    for key, val in item.x_set:
                        np.put(q, key[np.where(item.zi == 0)], 0)

            qg = np.hstack((q, dae.g))

            inc = self.solver.solve(self.Ac, -matrix(qg))

            # check for np.nan first
            if np.isnan(inc).any():
                logger.error(f'NaN found in solution. Convergence not likely')
                self.niter = self.config.max_iter + 1
                self.busted = True
                break

            # reset really small values to avoid anti-windup limiter flag jumps
            inc[np.where(np.abs(inc) < 1e-12)] = 0
            # set new values
            dae.x += np.ravel(np.array(inc[:dae.n]))
            dae.y += np.ravel(np.array(inc[dae.n: dae.n + dae.m]))
            system.vars_to_models()

            # calculate correction
            mis = np.max(np.abs(inc))
            self.mis.append(mis)
            self.niter += 1

            # converged
            if mis <= self.config.tol:
                self.converged = True
                break
            # non-convergence cases
            if self.niter > self.config.max_iter:
                logger.debug(f'Max. iter. {self.config.max_iter} reached for t={dae.t:.6f}, '
                             f'h={self.h:.6f}, mis={mis:.4g} '
                             f'({system.dae.xy_name[np.argmax(inc)]})')
                break
            if mis > 1000 and (mis > 1e8 * self.mis[0]):
                self.pbar.close()
                logger.error(f'Error increased too quickly. Convergence not likely.')
                self.busted = True
                break

        if not self.converged:
            dae.x = np.array(self.x0)
            dae.y = np.array(self.y0)
            dae.f = np.array(self.f0)
            system.vars_to_models()

        return self.converged

    def save_output(self):
        """
        Save the simulation data into two files: a lst file and a npy file.

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
                self.pbar.close()
                logger.error(f"Simulation does not converge for the given step size h={self.config.tstep:.4f}.")
                logger.error(f"Reduce the step size `tstep`, or set `shrinkt = 1` to let it shrink.")
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
                    self.pbar.close()
                    logger.error(f"Time step calculated to zero. Convergence not likely.")

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
        self.pbar = None
        self.plotter = None

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
