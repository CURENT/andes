from .base import RoutineBase
from ..config.powerflow import SPF

from cvxopt import matrix, sparse, div

from ..utils import elapsed
from ..utils.jactools import diag0

from ..utils.solver import Solver

import logging
logger = logging.getLogger(__name__)


class PowerFlow(RoutineBase):
    """
    Power flow calculation routine
    """
    def __init__(self, system, rc=None):
        self.system = system
        self.config = SPF(rc=rc)
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

    def init(self):
        """
        Initialize system for power flow study

        Returns
        -------
        None
        """
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
        logger.info('')

    def run(self):
        """
        Call the power flow solution routine
        Returns
        -------
        bool:
            True for success, False for fail
        """
        t, _ = elapsed()

        ret = None
        if self.config.method == 'NR':
            ret = self.newton()

        if self.solved:
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
        system = self.system
        dae = self.system.DAE

        while True:
            inc = self.calc_inc()
            dae.x += inc[:dae.n]
            dae.y += inc[dae.n:dae.n + dae.m]

            self.niter += 1

            max_mis = max(abs(inc))
            self.iter_mis.append(max_mis)

            logger.info(' Iter{:3d}.  Max. Mismatch = {:8.7f}'.format(
                self.niter, max_mis))

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
            diag0(system.DAE.Gy, 'unamey', system)

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
