import sys
import os
import time
import importlib
from collections import OrderedDict

from andes.routines.base import BaseRoutine
from andes.utils.misc import elapsed, is_notebook, is_interactive
from andes.utils.tab import Tab
from andes.shared import tqdm, np, pd
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
                                     ('honest', 0),
                                     ('tstep', 1/30),
                                     ('max_iter', 15),
                                     ('refresh_event', 0),
                                     ('g_scale', 1),
                                     ('qrt', 0),
                                     ('kqrt', 1.0),
                                     )))
        self.config.add_extra("_help",
                              tol="convergence tolerance",
                              t0="simulation starting time",
                              tf="simulation ending time",
                              fixt="use fixed step size (1) or variable (0)",
                              shrinkt='shrink step size for fixed method if not converged',
                              honest='honest Newton method that updates Jac at each step',
                              tstep='the initial step step size',
                              max_iter='maximum number of iterations',
                              refresh_event='refresh events at each step',
                              g_scale='scale algebraic residuals with time step size',
                              qrt='quasi-real-time stepping',
                              kqrt='quasi-real-time scaling factor; kqrt > 1 means slowing down',
                              )
        self.config.add_extra("_alt",
                              tol="float",
                              t0=">=0",
                              tf=">t0",
                              fixt=(0, 1),
                              shrinkt=(0, 1),
                              honest=(0, 1),
                              tstep='float',
                              max_iter='>=10',
                              refresh_event=(0, 1),
                              g_scale=(0, 1),
                              qrt='bool',
                              kqrt="positive",
                              )
        # overwrite `tf` from command line
        if system.options.get('tf') is not None:
            self.config.tf = system.options.get('tf')
        if system.options.get('qrt') is True:
            self.config.qrt = system.options.get('qrt')
        if system.options.get('kqrt') is not None:
            self.config.kqrt = system.options.get('kqrt')

        # if data is from a CSV file instead of simulation
        self.from_csv = system.options.get('from_csv')
        self.data_csv = None
        self.k_csv = 0    # row number

        # to be computed
        self.deltat = 0
        self.deltatmin = 0
        self.deltatmax = 0
        self.h = 0
        self.next_pc = 0
        self.Teye = None
        self.qg = np.array([])
        self.tol_zero = self.config.tol / 1000

        # internal status
        self.converged = False
        self.last_converged = False   # True if the previous step converged
        self.busted = False           # True if in a non-recoverable error state
        self.err_msg = ''
        self.niter = 0
        self._switch_idx = 0          # index into `System.switch_times`
        self._last_switch_t = -999    # the last critical time
        self.custom_event = False
        self.mis = 1
        self.pbar = None
        self.callpert = None
        self.plotter = None
        self.plt = None
        self.initialized = False
        self.qrt_start = None
        self.headroom = 0.0

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

        self.reset()
        self._load_pert()

        # restore power flow solutions
        system.dae.x[:len(system.PFlow.x_sol)] = system.PFlow.x_sol
        system.dae.y[:len(system.PFlow.y_sol)] = system.PFlow.y_sol

        # Note:
        #   calling `set_address` on `system.exist.pflow_tds` will point all variables
        #   to the new array after extending `dae.y`.
        system.set_address(models=system.exist.pflow_tds)
        system.set_dae_names(models=system.exist.tds)

        system.dae.clear_ts()
        system.store_sparse_pattern(models=system.exist.pflow_tds)
        system.store_adder_setter(models=system.exist.pflow_tds)
        system.vars_to_models()

        system.init(system.exist.tds, routine='tds')

        # only store switch times when not replaying CSV data
        if self.data_csv is None:
            system.store_switch_times(system.exist.tds)

        # Build mass matrix into `self.Teye`
        self.Teye = spdiag(system.dae.Tf.tolist())
        self.qg = np.zeros(system.dae.n + system.dae.m)

        # test if residuals are close enough to zero
        self.initialized = self.test_init()

        # discard initialized values and use that from CSV if provided
        if self.data_csv is not None:
            system.dae.x[:] = self.data_csv[0, 1:system.dae.n + 1]
            system.dae.y[:] = self.data_csv[0, system.dae.n + 1:system.dae.n + system.dae.m + 1]
            system.vars_to_models()

        # connect to data streaming server
        if system.config.dime_enabled:
            if system.streaming.dimec is None:
                system.streaming.connect()

        # send out system data using DiME
        self.streaming_init()
        self.streaming_step()

        # if `dae.n == 1`, `calc_h_first` depends on new `dae.gy`
        self.calc_h()

        _, s1 = elapsed(t0)

        if self.initialized is True:
            logger.info(f"Initialization for dynamics was successful in {s1}.")
        else:
            logger.error(f"Initialization for dynamics failed in {s1}.")

        if system.dae.n == 0:
            tqdm.write('No dynamic component loaded.')
        return system.dae.xy

    def summary(self):
        """
        Print out a summary of TDS options to logger.info.

        Returns
        -------
        None
        """
        out = list()
        out.append('')
        out.append('-> Time Domain Simulation Summary:')

        if self.data_csv is not None:
            out.append(f'Loaded data from CSV file "{self.from_csv}".')
            out.append('Replaying from CSV data.')
            out.append(f'Replay time: {self.system.dae.t}-{self.config.tf} sec.')
        else:
            out.append(f'Sparse Solver: {self.solver.sparselib.upper()}')
            out.append(f'Simulation time: {self.system.dae.t}-{self.config.tf} sec.')
            if self.config.fixt == 1:
                msg = f'Fixed step size: h={1000 * self.config.tstep:.4g} msec.'
                if self.config.shrinkt == 1:
                    msg += ', shrink if not converged'
                out.append(msg)
            else:
                out.append(f'Variable step size: h0={1000 * self.config.tstep:.4g} msec.')

        out_str = '\n'.join(out)
        logger.info(out_str)

        if self.config.honest == 1:
            logger.warning("The honest Newton method is used and will slow down the simulation.")
            logger.warning("For significant speed up, set `honest=0` in TDS.config.")

    def run(self, no_pbar=False, no_summary=False, **kwargs):
        """
        Run time-domain simulation using numerical integration.

        The default method is the Implicit Trapezoidal Method (ITM).

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

        # load from csv is provided
        if self.from_csv is not None:
            self.data_csv = self._load_csv(self.from_csv)

        if no_summary is False:
            self.summary()

        # only initializing at t=0 allows to continue when `run` is called again.
        if system.dae.t == 0:
            self.init()
        else:  # resume simulation
            resume = True
            self._calc_h_first()

        self.pbar = tqdm(total=100, ncols=70, unit='%', file=sys.stdout, disable=no_pbar)

        if resume:
            perc = round((dae.t - config.t0) / (config.tf - config.t0) * 100, 0)
            self.next_pc = perc + 1
            self.pbar.update(perc)

        self.qrt_start = time.time()
        self.headroom = 0.0

        t0, _ = elapsed()

        while (system.dae.t - self.h < self.config.tf) and (not self.busted):
            if self.callpert is not None:
                self.callpert(dae.t, system)

            step_status = False
            # call the stepping method of the integration method (or data replay)
            if self.data_csv is None:
                step_status = self._itm_step()  # compute for the current step
            else:
                step_status = self._csv_step()

            if step_status:
                # store values
                dae.ts.store_txyz(dae.t.tolist(),
                                  dae.xy,
                                  self.system.get_z(models=system.exist.pflow_tds),
                                  )

                self.streaming_step()

                # check if the next step is critical time
                self.do_switch()
                self.calc_h()
                dae.t += self.h

                # show progress in percentage
                perc = max(min((dae.t - config.t0) / (config.tf - config.t0) * 100, 100), 0)
                if perc >= self.next_pc:
                    self.pbar.update(1)
                    self.next_pc += 1

                # quasi-real-time check and wait (except for the last step)
                if config.qrt and self.h > 0:
                    rt_end = self.qrt_start + self.h * config.kqrt

                    # if the ending time has passed
                    if time.time() - rt_end > 0:
                        logger.debug(f'Simulation over-run at t={dae.t:4.4g} s.')
                    else:
                        self.headroom += (rt_end - time.time())

                        while time.time() - rt_end < 0:
                            time.sleep(1e-4)

                    self.qrt_start = time.time()

            else:
                if self.calc_h() == 0:
                    self.err_msg = "Time step reduced to zero. Convergence is not likely."
                    self.busted = True
                    break

        self.pbar.close()
        delattr(self, 'pbar')  # removed `pbar` so that System object can be serialized

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

        if config.qrt:
            logger.debug(f'QRT headroom time: {self.headroom} s.')

        system.TDS.save_output()

        # end data streaming
        if system.config.dime_enabled:
            system.streaming.finalize()

        # load data into `TDS.plotter` in a notebook or in an interactive mode
        if is_notebook() or is_interactive():
            self.load_plotter()

        return succeed

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

            reason = ''
            if dae.t == 0:
                reason = 't=0'
            elif self.config.honest:
                reason = 'using honest method'
            elif self.custom_event:
                reason = 'custom event set'
            elif not self.last_converged:
                reason = 'last step did not converge'
            elif self.niter > 4 and (self.niter + 1) % 3 == 0:
                reason = 'update every 6 iterations'
            elif dae.t - self._last_switch_t < 0.1:
                reason = 'within 0.1s of event'

            if reason:
                system.j_update(models=system.exist.pflow_tds, info=reason)

                # set flag in `solver.worker.factorize`, not `solver.factorize`.
                self.solver.worker.factorize = True

            # `Tf` should remain constant throughout the simulation, even if the corresponding diff. var.
            # is pegged by the anti-windup limiters.

            # solve implicit trapezoidal method (ITM) integration
            if self.config.g_scale == 1:
                gxs = self.h * dae.gx
                gys = self.h * dae.gy
            else:
                gxs = dae.gx
                gys = dae.gy

            self.Ac = sparse([[self.Teye - self.h * 0.5 * dae.fx, gxs],
                              [-self.h * 0.5 * dae.fy, gys]], 'd')

            # equation `self.qg[:dae.n] = 0` is the implicit form of differential equations using ITM
            self.qg[:dae.n] = dae.Tf * (dae.x - self.x0) - self.h * 0.5 * (dae.f + self.f0)

            # reset the corresponding q elements for pegged anti-windup limiter
            for item in system.antiwindups:
                for key, _, eqval in item.x_set:
                    np.put(self.qg, key, eqval)

            # set or scale the algebraic residuals
            if self.config.g_scale == 1:
                self.qg[dae.n:] = self.h * dae.g
            else:
                self.qg[dae.n:] = dae.g

            # calculate variable corrections
            if not self.config.linsolve:
                inc = self.solver.solve(self.Ac, matrix(self.qg))
            else:
                inc = self.solver.linsolve(self.Ac, matrix(self.qg))

            # check for np.nan first
            if np.isnan(inc).any():
                self.err_msg = 'NaN found in solution. Convergence is not likely'
                self.niter = self.config.max_iter + 1
                self.busted = True
                break

            # reset small values to reduce chattering
            inc[np.where(np.abs(inc) < self.tol_zero)] = 0

            # set new values
            dae.x -= inc[:dae.n].ravel()
            dae.y -= inc[dae.n: dae.n + dae.m].ravel()

            # synchronize solutions to model internal storage
            system.vars_to_models()

            # store `inc` to self for debugging
            self.inc = inc

            mis = np.max(np.abs(inc))
            # store initial maximum mismatch
            if self.niter == 0:
                self.mis = mis

            self.niter += 1

            # converged
            if mis <= self.config.tol:
                self.converged = True
                break
            # non-convergence cases
            if self.niter > self.config.max_iter:
                tqdm.write(f'* Max. iter. {self.config.max_iter} reached for t={dae.t:.6f}s, '
                           f'h={self.h:.6f}s, max inc={mis:.4g} ')

                # debug helpers
                g_max = np.argmax(abs(dae.g))
                inc_max = np.argmax(abs(inc))
                self._debug_g(g_max)
                self._debug_ac(inc_max)
                break

            if mis > 1e6 and (mis > 1e6 * self.mis):
                self.err_msg = 'Error increased too quickly. Convergence not likely.'
                self.busted = True
                break

        if not self.converged:
            dae.x[:] = np.array(self.x0)
            dae.y[:] = np.array(self.y0)
            dae.f[:] = np.array(self.f0)
            system.vars_to_models()

        self.last_converged = self.converged

        return self.converged

    def _csv_step(self):
        """
        Fetch data for the next step from ``data_csv``.
        """
        system = self.system
        if self.data_csv is not None:
            system.dae.x[:] = self.data_csv[self.k_csv, 1:system.dae.n + 1]
            system.dae.y[:] = self.data_csv[self.k_csv, system.dae.n + 1:system.dae.n + system.dae.m + 1]
            system.vars_to_models()

        self.converged = True
        return self.converged

    def calc_h(self, resume=False):
        """
        Calculate the time step size during the TDS.

        Parameters
        ----------
        resume : bool
            If True, calculate the initial step size.

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

        # t=0, first iteration (not previously failed)
        if (system.dae.t == 0 and self.niter == 0) or resume:
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

        if self.data_csv is not None:
            if self.k_csv + 1 < self.data_csv.shape[0]:
                self.k_csv += 1
                self.h = self.data_csv[self.k_csv, 0] - system.dae.t
            else:
                self.h = 0

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

        # if from CSV, determine `h` from data
        if self.data_csv is not None:
            if self.data_csv.shape[0] > 1:
                self.h = self.data_csv[1, 0] - self.data_csv[0, 0]
            else:
                logger.warning("CSV data does not contain more than one time step.")
                self.h = 0

        return self.h

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

    def do_switch(self):
        """
        Checks if is an event time and perform switch if true.

        Time is approximated with a tolerance of 1e-8.
        """
        ret = False
        system = self.system

        # refresh switch times if enabled
        if self.config.refresh_event:
            system.store_switch_times(system.exist.pflow_tds)

        # if not all events have been processed
        if self._switch_idx < system.n_switches:

            # if the current time is close enough to the next event time
            if np.isclose(system.dae.t, system.switch_times[self._switch_idx]):

                # `_last_switch_t` is used by the Jacobian updater
                self._last_switch_t = system.switch_times[self._switch_idx]

                # only call `switch_action` on the models that defined the time
                system.switch_action(system.switch_dict[self._last_switch_t])

                # progress `_switch_idx` to avoid calling the same event if time gets stuck
                self._switch_idx += 1
                system.vars_to_models()

                ret = True

        # if a `custom_event` flag is set (without a specific callback)
        if self.custom_event is True:
            system.switch_action(system.exist.pflow_tds)
            self._last_switch_t = system.dae.t.tolist()
            system.vars_to_models()
            self.custom_event = False
            ret = True

        return ret

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

    def _fg_wrapper(self, xy):
        """
        Wrapper function for equations. Callable by general-purpose DAE solvers.

        Parameters
        ----------
        xy : np.ndarray
            Input values for evaluating equations.

        Returns
        -------
        np.ndarray
            RHS of diff. and algeb. equations.

        """
        system = self.system
        system.dae.x[:] = xy[:system.dae.n]
        system.dae.y[:] = xy[system.dae.n:]
        system.vars_to_models()

        self._fg_update(system.exist.pflow_tds)

        return system.dae.fg

    def _load_pert(self):
        """
        Load perturbation files to ``self.callpert``.
        """
        system = self.system
        if system.files.pert:
            if not os.path.isfile(system.files.pert):
                logger.warning(f'Pert file not found at "{system.files.pert}".')
                return False

            sys.path.append(system.files.case_path)
            _, full_name = os.path.split(system.files.pert)
            name, ext = os.path.splitext(full_name)

            module = importlib.import_module(name)
            self.callpert = getattr(module, 'pert')
            logger.info(f'Perturbation file "{system.files.pert}" loaded.')
            return True

    def _load_csv(self, csv_file):
        """
        Load simulation data from CSV file and return a numpy array.
        """
        if csv_file is None:
            return None

        df = pd.read_csv(csv_file)

        if df.isnull().values.any():
            raise ValueError("CSV file contains missing values. Please check data consistency.")

        data = df.to_numpy()

        if data.ndim != 2:
            raise ValueError("Data from CSV is not 2-dimentional (time versus variable)")
        if data.shape[0] < 2:
            logger.warning("CSV data does not contain more than one time step.")

        # set start and end times from data
        self.config.t0 = data[0, 0]
        self.config.tf = data[-1, 0]

        return data

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
        logger.debug(f'Associated equation value is {self.system.dae.fg[xy_idx]:<20g}')
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

    def reset(self):
        """
        Reset internal states to pre-init condition.
        """
        self.deltat = 0
        self.deltatmin = 0
        self.deltatmax = 0
        self.h = 0
        self.next_pc = 0.1
        self.Teye = None
        self.qg = np.array([])

        self.converged = False
        self.last_converged = False
        self.busted = False
        self.niter = 0
        self._switch_idx = 0        # index into `System.switch_times`
        self._last_switch_t = -999  # the last event time
        self.custom_event = False
        self.mis = 1
        self.system.dae.t = np.array(0.0)
        self.pbar = None
        self.plotter = None
        self.plt = None             # short name for `plotter`

        self.initialized = False

    def rewind(self, t):
        """
        TODO: rewind to a past time.
        """
        raise NotImplementedError("TDS.rewind() not implemented")

    def streaming_init(self):
        """
        Send out initialization variables and process init from modules.

        Returns
        -------
        None
        """
        system = self.system
        if system.config.dime_enabled:
            system.streaming.send_init(recepient='all')
            logger.info('Broadcast system data. Waiting to receive modules init info...')
            time.sleep(0.5)
            system.streaming.sync_and_handle()

    def streaming_step(self):
        """
        Sync, handle and streaming for each integration step.

        Returns
        -------
        None
        """
        system = self.system
        if system.config.dime_enabled:
            system.streaming.sync_and_handle()
            system.streaming.vars_to_modules()
            system.streaming.vars_to_pmu()
