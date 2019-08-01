import logging

from cvxopt import matrix, sparse, div

from andes.config.pflow import Pflow
from andes.utils import elapsed
from andes.utils.solver import Solver
from .base import RoutineBase

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
        self.system.dae.factorize = True

    def pre(self):
        """
        Initialize system for power flow study

        Returns
        -------
        None
        """

        if self.solved and self.system.tds.initialized:
            logger.error('TDS has been initialized. Cannot solve power flow again.')
            return False

        logger.info('-> Power flow study: {} method, {} start'.format(
            self.config.method.upper(), 'flat' if self.config.flatstart else 'non-flat')
        )

        t, s = elapsed()

        system = self.system
        dae = self.system.dae

        system.dae.init_xy()

        for device, pflow, init0 in zip(system.devman.devices,
                                        system.call.pflow, system.call.init0):
            if pflow and init0:
                system.__dict__[device].init0(dae)

        # check for islands
        system.check_islands(show_info=True)

        # reset internal storage
        self.reset()

        t, s = elapsed(t)
        logger.debug('Power flow initialized in {:s}.'.format(s))

        return True

    def run(self, **kwargs):
        """
        call the power flow solution routine

        Returns
        -------
        bool
            True for success, False for fail
        """
        ret = None

        # initialization Y matrix and inital guess
        if not self.pre():
            return False

        t, _ = elapsed()

        # call solution methods
        if self.config.method == 'NR':
            ret = self.newton()
        elif self.config.method == 'DCPF':
            ret = self.dcpf()
        elif self.config.method in ('FDPF', 'FDBX', 'FDXB'):
            ret = self.fdpf()

        self.post()
        _, s = elapsed(t)

        if self.solved:
            logger.info(' Solution converged in {} in {} iterations'.format(s, self.niter))
        else:
            logger.warning(' Solution failed in {} in {} iterations'.format(s,
                           self.niter))
        return ret

    def newton(self):
        """
        Newton power flow routine

        Returns
        -------
        bool
            success flag
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

        return self.solved

    def dcpf(self):
        """
        Calculate linearized power flow

        Returns
        -------
        bool
            success flag, number of iterations
        """
        dae = self.system.dae

        self.system.Bus.init0(dae)
        self.system.dae.init_g()

        Va0 = self.system.Bus.angle
        for model, pflow, gcall in zip(self.system.devman.devices, self.system.call.pflow, self.system.call.gcall):
            if pflow and gcall:
                self.system.__dict__[model].gcall(dae)

        sw = self.system.SW.a
        sw.sort(reverse=True)
        no_sw = self.system.Bus.a[:]
        no_swv = self.system.Bus.v[:]

        for item in sw:
            no_sw.pop(item)
            no_swv.pop(item)

        Bp = self.system.Line.Bp[no_sw, no_sw]
        p = matrix(self.system.dae.g[no_sw], (no_sw.__len__(), 1))
        p = p-self.system.Line.Bp[no_sw, sw]*Va0[sw]

        Sp = self.solver.symbolic(Bp)
        N = self.solver.numeric(Bp, Sp)
        self.solver.solve(Bp, Sp, N, p)
        self.system.dae.y[no_sw] = p

        self.solved = True
        self.niter = 1

        return self.solved

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
            The solution to ``x = -A\\b``
        """
        system = self.system
        self.newton_call()

        A = sparse([[system.dae.Fx, system.dae.Gx],
                    [system.dae.Fy, system.dae.Gy]])

        inc = matrix([system.dae.f, system.dae.g])

        if system.dae.factorize:
            try:
                self.F = self.solver.symbolic(A)
                system.dae.factorize = False
            except NotImplementedError:
                pass

        try:
            N = self.solver.numeric(A, self.F)
            self.solver.solve(A, self.F, N, inc)
        except ValueError:
            logger.warning('Unexpected symbolic factorization.')
            system.dae.factorize = True
        except ArithmeticError:
            logger.warning('Jacobian matrix is singular.')
            system.dae.check_diag(system.dae.Gy, 'unamey')
        except NotImplementedError:
            inc = self.solver.linsolve(A, inc)

        return -inc

    def newton_call(self):
        """
        Function calls for Newton power flow

        Returns
        -------
        None

        """
        system = self.system
        dae = self.system.dae

        system.dae.init_fg()
        system.dae.reset_small_g()
        # evaluate algebraic equation mismatches
        for model, pflow, gcall in zip(system.devman.devices,
                                       system.call.pflow, system.call.gcall):
            if pflow and gcall:
                system.__dict__[model].gcall(dae)

        # eval differential equations
        for model, pflow, fcall in zip(system.devman.devices,
                                       system.call.pflow, system.call.fcall):
            if pflow and fcall:
                system.__dict__[model].fcall(dae)

        # reset islanded buses mismatches
        system.Bus.gisland(dae)

        if system.dae.factorize:
            system.dae.init_jac0()
            # evaluate constant Jacobian elements
            for model, pflow, jac0 in zip(system.devman.devices,
                                          system.call.pflow, system.call.jac0):
                if pflow and jac0:
                    system.__dict__[model].jac0(dae)
            dae.temp_to_spmatrix('jac0')

        dae.setup_FxGy()

        # evaluate Gy
        for model, pflow, gycall in zip(system.devman.devices,
                                        system.call.pflow, system.call.gycall):
            if pflow and gycall:
                system.__dict__[model].gycall(dae)

        # evaluate Fx
        for model, pflow, fxcall in zip(system.devman.devices,
                                        system.call.pflow, system.call.fxcall):
            if pflow and fxcall:
                system.__dict__[model].fxcall(dae)

        # reset islanded buses Jacobians
        system.Bus.gyisland(dae)

        dae.temp_to_spmatrix('jac')

    def post(self):
        """
        Post processing for solved systems.

        Store load, generation data on buses.
        Store reactive power generation on PVs and slack generators.
        Calculate series flows and area flows.

        Returns
        -------
        None
        """
        if not self.solved:
            return

        system = self.system

        system.call.pfload()
        system.Bus.Pl = system.dae.g[system.Bus.a]
        system.Bus.Ql = system.dae.g[system.Bus.v]

        system.call.pfgen()
        system.Bus.Pg = system.dae.g[system.Bus.a]
        system.Bus.Qg = system.dae.g[system.Bus.v]

        if system.PV.n:
            system.PV.qg = system.dae.y[system.PV.q]
        if system.SW.n:
            system.SW.pg = system.dae.y[system.SW.p]
            system.SW.qg = system.dae.y[system.SW.q]

        system.call.seriesflow()

        system.Area.seriesflow(system.dae)

    def fdpf(self):
        """
        Fast Decoupled Power Flow

        Returns
        -------
        bool
            Success flag
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

        # Fp = self.solver.symbolic(Bp)
        # Fpp = self.solver.symbolic(Bpp)
        # Np = self.solver.numeric(Bp, Fp)
        # Npp = self.solver.numeric(Bpp, Fpp)
        system.call.fdpf()

        # main loop
        while error > tol:
            # P-theta
            da = matrix(div(system.dae.g[no_sw], system.dae.y[no_swv]))
            # self.solver.solve(Bp, Fp, Np, da)
            da = self.solver.linsolve(Bp, da)
            system.dae.y[no_sw] += da

            system.call.fdpf()
            normP = max(abs(system.dae.g[no_sw]))

            # Q-V
            dV = matrix(div(system.dae.g[no_gv], system.dae.y[no_gv]))
            # self.solver.solve(Bpp, Fpp, Npp, dV)
            dV = self.solver.linsolve(Bpp, dV)
            system.dae.y[no_gv] += dV

            system.call.fdpf()
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

        return self.solved
