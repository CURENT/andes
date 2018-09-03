import logging
from cvxopt import matrix, sparse, div
from .base import RoutineBase
from andes.config.pflow import Pflow
from andes.utils import elapsed
from andes.utils.solver import Solver

logger = logging.getLogger(__name__)
__cli__ = 'pflow'


class PFLOW(RoutineBase):
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
        logger.info('-> Power flow study: {} method, {} start'.format(
            self.config.method.upper(), 'flat' if self.config.flatstart else 'non-flat')
        )

        t, s = elapsed()

        system = self.system
        dae = self.system.dae

        system.dae.init_xy()

        for device, pflow, init0 in zip(system.devman.devices, system.call.pflow, system.call.init0):
            if pflow and init0:
                system.__dict__[device].init0(dae)

        # check for islands
        system.check_islands(show_info=True)

        t, s = elapsed(t)
        logger.debug('Power flow initialized in {:s}.'.format(s))

    def run(self, **kwargs):
        """
        call the power flow solution routine

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
            logger.info(' Solution converged in {} in {} iterations'.format(s, self.niter))
        else:
            logger.warn(' Solution failed in {} in {} iterations'.format(s, self.niter))
        return ret

    def newton(self):
        """
        Newton power flow routine

        Returns
        -------
        (bool, int)
            success flag, number of iterations
        """
        dae = self.system.dae

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
        msg = ' Iter {:<d}.  max mismatch = {:8.7f}'.format(niter, max_mis)
        logger.info(msg)

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

        A = sparse([[system.dae.Fx, system.dae.Gx],
                    [system.dae.Fy, system.dae.Gy]])

        inc = matrix([system.dae.f, system.dae.g])

        if system.dae.factorize:
            self.F = self.solver.symbolic(A)
            system.dae.factorize = False

        try:
            N = self.solver.numeric(A, self.F)
            self.solver.solve(A, self.F, N, inc)
        except ValueError:
            logger.warning('Unexpected symbolic factorization.')
            system.dae.factorize = True
        except ArithmeticError:
            logger.warning('Jacobian matrix is singular.')
            system.dae.check_diag(system.dae.Gy, 'unamey')

        return -inc

    def newton_call(self):
        """
        Function calls for Newton power flow

        Returns
        -------
        None

        """
        # system = self.system
        # exec(system.call.newton)

        system = self.system
        dae = self.system.dae

        system.dae.init_fg()

        # evaluate algebraic equation mismatches
        for model, pflow, gcall in zip(system.devman.devices, system.call.pflow, system.call.gcall):
            if pflow and gcall:
                system.__dict__[model].gcall(dae)

        # eval differential equations
        for model, pflow, fcall in zip(system.devman.devices, system.call.pflow, system.call.fcall):
            if pflow and fcall:
                system.__dict__[model].fcall(dae)

        # reset islanded buses mismatches
        system.Bus.gisland(dae)

        if system.dae.factorize:
            system.dae.init_jac0()
            # evaluate constant Jacobian elements
            for model, pflow, jac0 in zip(system.devman.devices, system.call.pflow, system.call.jac0):
                if pflow and jac0:
                    system.__dict__[model].jac0(dae)
            dae.temp_to_spmatrix('jac0')

        dae.setup_FxGy()

        # evaluate Gy
        for model, pflow, gycall in zip(system.devman.devices, system.call.pflow, system.call.gycall):
            if pflow and gycall:
                system.__dict__[model].gycall(dae)

        # evaluate Fx
        for model, pflow, fxcall in zip(system.devman.devices, system.call.pflow, system.call.fxcall):
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

        exec(system.call.pfload)
        system.Bus.Pl = system.dae.g[system.Bus.a]
        system.Bus.Ql = system.dae.g[system.Bus.v]

        exec(system.call.pfgen)
        system.Bus.Pg = system.dae.g[system.Bus.a]
        system.Bus.Qg = system.dae.g[system.Bus.v]

        if system.PV.n:
            system.PV.qg = system.dae.y[system.PV.q]
        if system.SW.n:
            system.SW.pg = system.dae.y[system.SW.p]
            system.SW.qg = system.dae.y[system.SW.q]

        exec(system.call.seriesflow)

        system.Area.seriesflow(system.dae)

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
        exec(system.call.fdpf)

        # main loop
        while error > tol:
            # P-theta
            da = matrix(div(system.dae.g[no_sw], system.dae.y[no_swv]))
            self.solver.solve(Bp, Fp, Np, da)
            system.dae.y[no_sw] += da

            exec(system.call.fdpf)
            normP = max(abs(system.dae.g[no_sw]))

            # Q-V
            dV = matrix(div(system.dae.g[no_gv], system.dae.y[no_gv]))
            self.solver.solve(Bpp, Fpp, Npp, dV)
            system.dae.y[no_gv] += dV

            exec(system.call.fdpf)
            normQ = max(abs(system.dae.g[no_gv]))

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
