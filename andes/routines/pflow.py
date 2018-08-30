from cvxopt import matrix, sparse, div
from .base import RoutineBase
from ..config.pflow import Pflow
from ..utils import elapsed
from ..utils.solver import Solver

import logging
logger = logging.getLogger(__name__)


class PowerFlow(RoutineBase):
    """
    Power flow calculation routine
    """
    def __init__(self, system, rc=None):
        self.system = system
        self.config = Pflow(rc=rc)
        self.solver = Solver(system.config.sparselib)

        # store status and internal flags
        self.solved = False
        self.niter = 0
        self.iter_mis = []
        self.F = None

    def reset(self):
        """
        Reset all internal storage to initial status

        Returns
        -------
        None
        """
        self.solved = False
        self.niter = 0
        self.iter_mis = []
        self.F = None

    def pre(self):
        """
        Initialize system for power flow study

        Returns
        -------
        None
        """
        logger.info('')
        logger.info('Power flow study: {} method, {} start'.format(
            self.config.method.upper(), 'Flat' if self.config.flatstart else 'Non-flat')
        )

        t, s = elapsed()

        system = self.system
        dae = self.system.DAE

        system.DAE.init_xy()

        for device, pflow, init0 in zip(system.DevMan.devices, system.Call.pflow, system.Call.init0):
            if pflow and init0:
                system.__dict__[device].init0(dae)

        # check for islands
        system.check_islands(show_info=True)

        t, s = elapsed(t)
        logger.info('Power flow initialized in {:s}.'.format(s))

    def run(self):
        """
        Call the power flow solution routine
        Returns
        -------
        bool:
            True for success, False for fail
        """
        ret = None

        self.pre()
        t, _ = elapsed()

        # call solution methods
        if self.config.method == 'NR':
            ret = self.newton()
        elif self.config.method in ('FDPF', 'FDBX', 'FDXB'):
            ret = self.fdpf()

        self.post()
        _, s = elapsed(t)

        if self.solved:
            logger.info('Power flow converged in {}'.format(s))
        else:
            logger.warn('Power flow failed in {}'.format(s))
        return ret

    def newton(self):
        """
        Newton power flow routine
        Returns
        -------
        (bool, int)
            success flag, number of iterations
        """
        dae = self.system.DAE

        while True:
            inc = self.calc_inc()
            dae.x += inc[:dae.n]
            dae.y += inc[dae.n:dae.n + dae.m]

            self.niter += 1

            max_mis = max(abs(inc))
            self.iter_mis.append(max_mis)

            self._iter_info(self.niter)

            if max_mis < self.config.tol:
                self.solved = True
                break
            elif self.niter > 5 and max_mis > 1000 * self.iter_mis[0]:
                logger.warning('Blown up in {0} iterations.'.format(self.niter))
                break
            if self.niter > self.config.maxit:
                logger.warning('Reached maximum number of iterations.')
                break

        return self.solved, self.niter

    def _iter_info(self, niter, level=logging.INFO):
        """
        Log iteration number and mismatch

        Parameters
        ----------
        level
            logging level
        Returns
        -------
        None
        """
        max_mis = self.iter_mis[niter - 1]
        msg = ' Iter {:<d}.  max mismatch = {:8.7f}'.format(
            niter, max_mis)

        logger.log(level, msg)

    def calc_inc(self):
        """
        Calculate the Newton incrementals for each step

        Returns
        -------
        matrix
            The solution to ``x = -A\b``
        """
        system = self.system
        self.newton_call()

        A = sparse([[system.DAE.Fx, system.DAE.Gx],
                    [system.DAE.Fy, system.DAE.Gy]])

        inc = matrix([system.DAE.f, system.DAE.g])

        if system.DAE.factorize:
            self.F = self.solver.symbolic(A)
            system.DAE.factorize = False

        try:
            N = self.solver.numeric(A, self.F)
            self.solver.solve(A, self.F, N, inc)
        except ValueError:
            logger.warning('Unexpected symbolic factorization.')
            system.DAE.factorize = True
        except ArithmeticError:
            logger.warning('Jacobian matrix is singular.')
            system.DAE.check_diag(system.DAE.Gy, 'unamey')

        return -inc

    def newton_call(self):
        """
        Function calls for Newton power flow

        Returns
        -------
        None

        """
        # system = self.system
        # exec(system.Call.newton)

        system = self.system
        dae = self.system.DAE

        system.DAE.init_fg()

        # evaluate algebraic equation mismatches
        for model, pflow, gcall in zip(system.DevMan.devices, system.Call.pflow, system.Call.gcall):
            if pflow and gcall:
                system.__dict__[model].gcall(dae)

        # eval differential equations
        for model, pflow, fcall in zip(system.DevMan.devices, system.Call.pflow, system.Call.fcall):
            if pflow and fcall:
                system.__dict__[model].fcall(dae)

        # reset islanded buses mismatches
        system.Bus.gisland(dae)

        if system.DAE.factorize:
            system.DAE.init_jac0()
            # evaluate constant Jacobian elements
            for model, pflow, jac0 in zip(system.DevMan.devices, system.Call.pflow, system.Call.jac0):
                if pflow and jac0:
                    system.__dict__[model].jac0(dae)
            dae.temp_to_spmatrix('jac0')

        dae.setup_FxGy()

        # evaluate Gy
        for model, pflow, gycall in zip(system.DevMan.devices, system.Call.pflow, system.Call.gycall):
            if pflow and gycall:
                system.__dict__[model].gycall(dae)

        # evaluate Fx
        for model, pflow, fxcall in zip(system.DevMan.devices, system.Call.pflow, system.Call.fxcall):
            if pflow and fxcall:
                system.__dict__[model].fxcall(dae)

        # reset islanded buses Jacobians
        system.Bus.gyisland(dae)

        dae.temp_to_spmatrix('jac')

    def post(self):
        """
        Post processing for solved systems.

        Store load, generation data on buses. Store reactive power generation on PVs and slack generators.
        Calculate series flows and area flows.
        Returns
        -------
        None
        """
        if not self.solved:
            return

        system = self.system

        exec(system.Call.pfload)
        system.Bus.Pl = system.DAE.g[system.Bus.a]
        system.Bus.Ql = system.DAE.g[system.Bus.v]

        exec(system.Call.pfgen)
        system.Bus.Pg = system.DAE.g[system.Bus.a]
        system.Bus.Qg = system.DAE.g[system.Bus.v]

        if system.PV.n:
            system.PV.qg = system.DAE.y[system.PV.q]
        if system.SW.n:
            system.SW.pg = system.DAE.y[system.SW.p]
            system.SW.qg = system.DAE.y[system.SW.q]

        exec(system.Call.seriesflow)

        system.Area.seriesflow(system.DAE)

    def fdpf(self):
        """
        Fast Decoupled Power Flow

        Returns
        -------
        bool, int
            Success flag, number of iterations
        """
        system = self.system

        # general settings
        self.niter = 1
        iter_max = self.config.maxit
        self.solved = True
        tol = self.config.tol
        error = tol + 1

        self.iter_mis = []
        if (not system.Line.Bp) or (not system.Line.Bpp):
            system.Line.build_b()

        # initialize indexing and Jacobian
        # ngen = system.SW.n + system.PV.n
        sw = system.SW.a
        sw.sort(reverse=True)
        no_sw = system.Bus.a[:]
        no_swv = system.Bus.v[:]
        for item in sw:
            no_sw.pop(item)
            no_swv.pop(item)
        gen = system.SW.a + system.PV.a
        gen.sort(reverse=True)
        no_g = system.Bus.a[:]
        no_gv = system.Bus.v[:]
        for item in gen:
            no_g.pop(item)
            no_gv.pop(item)
        Bp = system.Line.Bp[no_sw, no_sw]
        Bpp = system.Line.Bpp[no_g, no_g]

        Fp = self.solver.symbolic(Bp)
        Fpp = self.solver.symbolic(Bpp)
        Np = self.solver.numeric(Bp, Fp)
        Npp = self.solver.numeric(Bpp, Fpp)
        exec(system.Call.fdpf)

        # main loop
        while error > tol:
            # P-theta
            da = matrix(div(system.DAE.g[no_sw], system.DAE.y[no_swv]))
            self.solver.solve(Bp, Fp, Np, da)
            system.DAE.y[no_sw] += da

            exec(system.Call.fdpf)
            normP = max(abs(system.DAE.g[no_sw]))

            # Q-V
            dV = matrix(div(system.DAE.g[no_gv], system.DAE.y[no_gv]))
            self.solver.solve(Bpp, Fpp, Npp, dV)
            system.DAE.y[no_gv] += dV

            exec(system.Call.fdpf)
            normQ = max(abs(system.DAE.g[no_gv]))

            err = max([normP, normQ])
            self.iter_mis.append(err)
            error = err

            self._iter_info(self.niter)
            self.niter += 1

            if self.niter > 4 and self.iter_mis[-1] > 1000 * self.iter_mis[0]:
                logger.warning('Blown up in {0} iterations.'.format(self.niter))
                self.solved = False
                break

            if self.niter > iter_max:
                logger.warning('Reached maximum number of iterations.')
                self.solved = False
                break

        return self.solved, self.niter
