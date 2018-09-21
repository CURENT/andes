import logging
from math import isnan
from time import monotonic as time, sleep
import importlib
import sys

import progressbar
from cvxopt import matrix, sparse, spdiag

from .base import RoutineBase
from andes.config.tds import Tds
from andes.utils import elapsed
from andes.utils.math import zeros
from andes.utils.solver import Solver

logger = logging.getLogger(__name__)
__cli__ = 'tds'


class TDS(RoutineBase):
    """
    Time domain simulation (TDS) routine
    """
    def __init__(self, system, rc=None):
        self.system = system
        self.config = Tds(rc=rc)
        self.solver = Solver(system.config.sparselib)

        # internal states
        self.F = None
        self.bar = progressbar.ProgressBar(maxval=100,
                                           widgets=[
                                               ' [',
                                               progressbar.Percentage(),
                                               progressbar.Bar(),
                                               progressbar.AdaptiveETA(), '] '
                                           ])

        self.switch = False
        self.next_pc = 0.1
        self.step = 0
        self.t = self.config.t0
        self.h = 0
        self.headroom = 0
        self.t_jac = 0
        self.inc = None
        self.callpert = None
        self.solved = False
        self.fixed_times = []
        self.convergence = True
        self.niter = 0
        self.err = 1
        self.x0 = None
        self.y0 = None
        self.f0 = None
        self.success = False

    def _calc_time_step_first(self):
        """
        Compute the first time step and save to ``self.h``

        Returns
        -------
        None
        """
        system = self.system
        config = self.config

        if not system.dae.n:
            freq = 1.0
        elif system.dae.n == 1:
            B = matrix(system.dae.Gx)
            self.solver.linsolve(system.dae.Gy, B)
            As = system.dae.Fx - system.dae.Fy * B
            freq = abs(As[0, 0])
        else:
            freq = 20.0

        if freq > system.freq:
            freq = float(system.freq)

        tspan = abs(config.tf - config.t0)
        tcycle = 1 / freq
        config.deltatmax = min(5 * tcycle, tspan / 100.0)
        config.deltat = min(tcycle, tspan / 100.0)
        config.deltatmin = min(tcycle / 64, config.deltatmax / 20)

        if config.fixt:
            if config.tstep <= 0:
                logger.warning('Fixed time step is negative or zero')
                logger.warning('Switching to automatic time step')
                config.fixt = False
            else:
                config.deltat = config.tstep
                if config.tstep < config.deltatmin:
                    logger.warning(
                        'Fixed time step is below the estimated minimum')

        self.h = config.deltat

    def calc_time_step(self):
        """
        Set the time step during time domain simulations

        Parameters
        ----------
        convergence: bool
            truth value of the convergence of the last step
        niter: int
            current iteration count
        t: float
            current simulation time

        Returns
        -------
        float
            computed time step size
        """
        system = self.system
        config = self.config
        convergence = self.convergence
        niter = self.niter
        t = self.t

        if t == 0:
            self._calc_time_step_first()
            return

        if convergence:
            if niter >= 15:
                config.deltat = max(config.deltat * 0.5, config.deltatmin)
            elif niter <= 6:
                config.deltat = min(config.deltat * 1.1, config.deltatmax)
            else:
                config.deltat = max(config.deltat * 0.95, config.deltatmin)

            # adjust fixed time step if niter is high
            if config.fixt:
                config.deltat = min(config.tstep, config.deltat)
        else:
            config.deltat *= 0.9
            if config.deltat < config.deltatmin:
                config.deltat = 0

        if system.Fault.is_time(t) or system.Breaker.is_time(t):
            config.deltat = min(config.deltat, 0.002778)
        elif system.check_event(t):
            config.deltat = min(config.deltat, 0.002778)

        if config.method == 'fwdeuler':
            config.deltat = min(config.deltat, config.tstep)

        # last step size
        if self.t + config.deltat > config.tf:
            config.deltat = config.tf - self.t

        # reduce time step for fixed_times events
        for fixed_t in self.fixed_times:
            if (fixed_t > self.t) and (fixed_t < self.t + config.deltat):
                config.deltat = fixed_t - self.t
                self.switch = True
                break

        self.h = config.deltat

    def init(self):
        """
        Initialize time domain simulation

        Returns
        -------
        None
        """
        system = self.system
        config = self.config
        dae = self.system.dae
        if system.pflow.solved is False:
            return

        t, s = elapsed()

        # Assign indices for post-powerflow device variables
        system.xy_addr1()

        # Assign variable names for bus injections and line flows if enabled
        system.varname.resize_for_flows()
        system.varname.bus_line_names()

        # Reshape dae to retain power flow solutions
        system.dae.init1()

        # Initialize post-powerflow device variables
        for device, init1 in zip(system.devman.devices, system.call.init1):
            if init1:
                system.__dict__[device].init1(system.dae)

        # compute line and area flow
        if config.compute_flows:
            dae.init_fg()
            self.compute_flows()  # TODO: move to PowerSystem

        t, s = elapsed(t)

        if system.dae.n:
            logger.debug('Dynamic models initialized in {:s}.'.format(s))
        else:
            logger.debug('No dynamic model loaded.')

        # system.dae flags initialize
        system.dae.factorize = True
        system.dae.mu = 1.0
        system.dae.kg = 0.0

    def run(self):
        """
        Run time domain simulation

        Returns
        -------
        bool
            Success flag
        """
        ret = False
        system = self.system
        config = self.config
        dae = self.system.dae

        # maxit = config.maxit
        # tol = config.tol

        if system.pflow.solved is False:
            logger.warning('Power flow not solved. Simulation cannot continue.')
            return ret
        t0, _ = elapsed()

        logger.info('-> Time Domain Simulation: {} method, t={} s'
                    .format(self.config.method, self.config.tf))

        self.load_pert()

        self.run_step0()

        config.qrtstart = time()
        self.bar.start()

        while self.t < config.tf:
            self.calc_time_step()
            self.check_fixed_times()

            if self.h == 0:
                break
            # progress time and set time in dae
            self.t += self.h
            dae.t = self.t

            # backup actual variables
            self.x0 = matrix(dae.x)
            self.y0 = matrix(dae.y)
            self.f0 = matrix(dae.f)

            # apply fixed_time interventions and perturbations
            self.event_actions()

            # reset flags used in each step
            self.err = 1
            self.niter = 0
            self.convergence = False

            self.implicit_step()

            if self.convergence is False:
                self.restore_values()
                continue

            self.step += 1
            self.compute_flows()
            system.varout.store(self.t, self.step)

            # plot variables and display iteration status
            perc = max(min((self.t - config.t0) / (config.tf - config.t0) * 100, 100), 0)

            if self.bar is not None:
                self.bar.update(perc)

            if perc > self.next_pc or self.t == config.tf:
                self.next_pc += 20
                if self.bar is None:
                    logger.info(' ({:.0f}%) time = {:.4f}s, step = {}, niter = {}'
                                .format(100 * self.t / config.tf, self.t, self.step, self.niter))

            # compute max rotor angle difference
            # diff_max = anglediff()

            # quasi-real-time check and wait
            rt_end = config.qrtstart + (self.t - config.t0) * config.kqrt

            if config.qrt:
                # the ending time has passed
                if time() - rt_end > 0:
                    # simulation is too slow
                    if time() - rt_end > config.kqrt:
                        logger.debug('Simulation over-run at t={:4.4g} s.'.format(self.t))
                # wait to finish
                else:
                    self.headroom += (rt_end - time())
                    while time() - rt_end < 0:
                        sleep(1e-5)

        if self.bar is not None:
            self.bar.finish()

        if config.qrt:
            logger.debug('RT headroom time: {} s.'.format(str(self.headroom)))

        if self.t != config.tf:
            logger.error('Reached minimum time step. Convergence is not likely.')
            ret = False
        else:
            ret = True

        _, s = elapsed(t0)

        if ret is True:
            logger.info(' Time domain simulation finished in {:s}.'.format(s))
        else:
            logger.info(' Time domain simulation failed in {:s}.'.format(s))

        self.success = ret

        self.dump_results()

        return ret

    def restore_values(self):
        """
        Restore x, y, and f values if not converged

        Returns
        -------
        None
        """
        if self.convergence is True:
            return
        dae = self.system.dae
        system = self.system

        inc_g = self.inc[dae.n:dae.m + dae.n]
        max_g_err_sign = 1 if abs(max(inc_g)) > abs(min(inc_g)) else -1
        if max_g_err_sign == 1:
            max_g_err_idx = list(inc_g).index(max(inc_g))
        else:
            max_g_err_idx = list(inc_g).index(min(inc_g))
        logger.debug(
            'Maximum mismatch = {:.4g} at equation <{}>'.format(
                max(abs(inc_g)), system.varname.unamey[max_g_err_idx]))
        logger.debug(
            'Reducing time step h={:.4g}s for t={:.4g}'.format(self.h, self.t))

        # restore initial variable data
        dae.x = matrix(self.x0)
        dae.y = matrix(self.y0)
        dae.f = matrix(self.f0)

    def implicit_step(self):
        """
        Integrate one step using trapezoidal method. Sets convergence and niter flags.

        Returns
        -------
        None
        """
        config = self.config
        system = self.system
        dae = self.system.dae

        # constant short names
        In = spdiag([1] * dae.n)
        h = self.h

        while self.err > config.tol and self.niter < config.maxit:
            if self.t - self.t_jac >= 5:
                dae.rebuild = True
                self.t_jac = self.t
            elif self.niter > 4:
                dae.rebuild = True
            elif dae.factorize:
                dae.rebuild = True

            # rebuild Jacobian
            if dae.rebuild:
                exec(system.call.int)
                dae.rebuild = False
            else:
                exec(system.call.int_fg)

            # complete Jacobian matrix dae.Ac
            if config.method == 'euler':
                dae.Ac = sparse(
                    [[In - h * dae.Fx, dae.Gx], [-h * dae.Fy, dae.Gy]],
                    'd')
                dae.q = dae.x - self.x0 - h * dae.f

            elif config.method == 'trapezoidal':
                dae.Ac = sparse([[In - h * 0.5 * dae.Fx, dae.Gx],
                                 [-h * 0.5 * dae.Fy, dae.Gy]], 'd')
                dae.q = dae.x - self.x0 - h * 0.5 * (dae.f + self.f0)

            # windup limiters
            dae.reset_Ac()

            if dae.factorize:
                self.F = self.solver.symbolic(dae.Ac)
                dae.factorize = False
            self.inc = -matrix([dae.q, dae.g])

            try:
                N = self.solver.numeric(dae.Ac, self.F)
                self.solver.solve(dae.Ac, self.F, N, self.inc)
            except ArithmeticError:
                logger.error('Singular matrix')
                dae.check_diag(dae.Gy, 'unamey')
                dae.check_diag(dae.Fx, 'unamex')
                # force quit
                self.niter = config.maxit + 1
                break
            except ValueError:
                logger.warning('Unexpected symbolic factorization')
                dae.factorize = True
                continue
            else:
                inc_x = self.inc[:dae.n]
                inc_y = self.inc[dae.n:dae.m + dae.n]
                dae.x += inc_x
                dae.y += inc_y

            self.err = max(abs(self.inc))
            if isnan(config.error):
                logger.error('Iteration error: NaN detected.')
                self.niter = config.maxit + 1
                break

            self.niter += 1

        if self.niter <= config.maxit:
            self.convergence = True

    def event_actions(self):
        """
        Take actions for timed events

        Returns
        -------
        None
        """
        system = self.system
        dae = system.dae
        if self.switch:
            system.Breaker.apply(self.t)
            for item in system.check_event(self.t):
                system.__dict__[item].apply(self.t)

            dae.rebuild = True
            self.switch = False

    def check_fixed_times(self):
        """
        Check for fixed times and store in ``self.fixed_times``.

        Returns
        -------
        None
        """
        self.fixed_times = self.system.get_event_times()

    def load_pert(self):
        """
        Load perturbation files to ``self.callpert``

        Returns
        -------
        None
        """
        system = self.system

        if system.files.pert:
            try:
                sys.path.append(system.files.path)
                module = importlib.import_module(system.files.pert[:-3])
                self.callpert = getattr(module, 'pert')
            except ImportError:
                logger.warning('Pert file is discarded due to import errors.')
                self.callpert = None

    def run_step0(self):
        """
        For the 0th step, store the data and stream data

        Returns
        -------
        None
        """
        dae = self.system.dae
        system = self.system

        self.inc = zeros(dae.m + dae.n, 1)
        system.varout.store(self.t, self.step)

    def angle_diff(self):
        """
        Compute the maximum angle difference between generators

        Returns
        -------
        float
            maximum angular difference
        """
        return 0

    def compute_flows(self):
        """
        If enabled, compute the line flows after each step

        Returns
        -------
        None
        """
        system = self.system
        config = self.config
        dae = system.dae

        if config.compute_flows:
            # compute and append series injections on buses

            exec(system.call.bus_injection)
            bus_inj = dae.g[:2 * system.Bus.n]

            exec(system.call.seriesflow)
            system.Area.seriesflow(system.dae)
            system.Area.interchange_varout()
            dae.y = matrix([
                dae.y, bus_inj, system.Line._line_flows, system.Area.inter_varout
            ])

    def dump_results(self):
        """
        Dump simulation results to ``dat`` and ``lst`` files

        Returns
        -------
        None
        """
        system = self.system

        t, _ = elapsed()

        if self.success and (not system.files.no_output):
            system.varout.dump()
            _, s = elapsed(t)
            logger.info('Simulation data dumped in {:s}.'.format(s))
