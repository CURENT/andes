"""
Module for power flow calculation.
"""

import logging
from collections import OrderedDict

from andes.utils.misc import elapsed
from andes.routines.base import BaseRoutine
from andes.variables.report import Report
from andes.shared import np, matrix, sparse, newton_krylov

logger = logging.getLogger(__name__)


class PFlow(BaseRoutine):
    """
    Power flow calculation routine.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.config.add(OrderedDict((('tol', 1e-6),
                                     ('max_iter', 25),
                                     ('method', 'NR'),
                                     ('check_conn', 1),
                                     ('n_factorize', 4),
                                     ('report', 1),
                                     ('degree', 0),
                                     ('init_tds', 0),
                                     )))
        self.config.add_extra("_help",
                              tol="convergence tolerance",
                              max_iter="max. number of iterations",
                              method="calculation method",
                              check_conn='check connectivity before power flow',
                              n_factorize="first N iterations to factorize Jacobian in dishonest method",
                              report="write output report",
                              degree='use degree in report',
                              init_tds="initialize TDS after PFlow",
                              )
        self.config.add_extra("_alt",
                              tol="float",
                              method=("NR", "dishonest", "NK"),
                              check_conn=(0, 1),
                              max_iter=">=10",
                              n_factorize=">0",
                              report=(0, 1),
                              degree=(0, 1),
                              init_tds=(0, 1),
                              )

        self.converged = False
        self.inc = None
        self.A = None
        self.niter = 0
        self.mis = [1]
        self.models = OrderedDict()

        self.x_sol = None
        self.y_sol = None

    def init(self):
        """
        Initialize variables for power flow.
        """

        system = self.system

        t0, _ = elapsed()

        self.models = system.find_models('pflow')
        self.converged = False

        self.res = matrix(0, (system.dae.n + system.dae.m, 1), 'd')
        self.A = None

        self.niter = 0
        self.mis = [1]
        self.exec_time = 0.0

        self.x_sol = None
        self.y_sol = None

        self.system.set_var_arrays(self.models, inplace=True, alloc=False)
        self.system.init(self.models, routine='pflow')

        _, s1 = elapsed(t0)
        logger.info('Power flow initialized in %s.', s1)

        # force compile if numba is on - improves timing accuracy
        if system.config.numba:
            t0, _ = elapsed()

            system.f_update(self.models)
            system.g_update(self.models)
            system.j_update(models=self.models)

            _, s1 = elapsed(t0)
            logger.info('Numba compilation for power flow finished in %s.', s1)

        return system.dae.xy

    def nr_step(self):
        """
        Solve a single iteration step using the Newton-Raphson method.

        Returns
        -------
        float
            maximum absolute mismatch
        """

        system = self.system

        # ---------- Build numerical DAE----------
        self.fg_update()

        # ---------- update the Jacobian on conditions ----------
        if self.config.method != 'dishonest' or (self.niter < self.config.n_factorize):
            system.j_update(self.models)
            self.solver.worker.new_A = True

        # ---------- prepare and solve linear equations ----------
        self.res[:system.dae.n] = -system.dae.f[:]
        self.res[system.dae.n:] = -system.dae.g[:]

        self.A = sparse([[system.dae.fx, system.dae.gx],
                         [system.dae.fy, system.dae.gy]])

        if not self.config.linsolve:
            self.inc = self.solver.solve(self.A, self.res)
        else:
            self.inc = self.solver.linsolve(self.A, self.res)

        system.dae.x += np.ravel(self.inc[:system.dae.n])
        system.dae.y += np.ravel(self.inc[system.dae.n:])

        # find out variables associated with maximum mismatches
        fmax = 0
        if system.dae.n > 0:
            fmax_idx = np.argmax(np.abs(system.dae.f))
            fmax = system.dae.f[fmax_idx]
            logger.debug("Max. diff mismatch %.10g on %s", fmax, system.dae.x_name[fmax_idx])

        gmax_idx = np.argmax(np.abs(system.dae.g))
        gmax = system.dae.g[gmax_idx]
        logger.debug("Max. algeb mismatch %.10g on %s", gmax, system.dae.y_name[gmax_idx])

        mis = max(abs(fmax), abs(gmax))
        system.vars_to_models()

        return mis

    def nr_solve(self):
        """
        Solve the power flow problem using itertive Newton's method.
        """

        self.niter = 0
        while True:
            mis = self.nr_step()
            logger.info('%d: |F(x)| = %.10g', self.niter, mis)

            # store the increment
            if self.niter == 0:
                self.mis[0] = mis
            else:
                self.mis.append(mis)

            # check for convergence
            if mis < self.config.tol:
                self.converged = True
                break

            if self.niter > self.config.max_iter:
                break

            if np.isnan(mis).any():
                logger.error('NaN found in solution. Convergence is not likely')
                break

            if mis > 1e4 * self.mis[0]:
                logger.error('Mismatch increased too fast. Convergence is not likely.')
                break

            self.niter += 1

        return self.converged

    def summary(self):
        """
        Output a summary for the PFlow routine.
        """

        # extract package name of the solver
        sp_module = sparse.__module__
        if '.' in sp_module:
            sp_module = sp_module.split('.')[0]

        out = list()
        out.append('')
        out.append('-> Power flow calculation')
        out.append(f'{"Numba":>16s}: {"On" if self.system.config.numba else "Off"}')
        out.append(f'{"Sparse solver":>16s}: {self.solver.sparselib.upper()}')
        out.append(f'{"Solution method":>16s}: {self.config.method} method')

        out_str = '\n'.join(out)
        logger.info(out_str)

    def run(self, **kwargs):
        """
        Solve the power flow using the selected method.

        Returns
        -------
        bool
            convergence status
        """

        system = self.system
        if self.config.check_conn == 1:
            self.system.connectivity()

        self.summary()
        self.init()

        if system.dae.m == 0:
            logger.error("Loaded case contains no power flow element.")
            system.exit_code = 1
            return False

        method = self.config.method.lower()

        t0, _ = elapsed()

        # ---------- Call solution methods ----------
        if method == 'nr':
            self.nr_solve()
        elif method == 'nk':
            self.newton_krylov()

        t1, s1 = elapsed(t0)
        self.exec_time = t1 - t0

        if not self.converged:
            if abs(self.mis[-1] - self.mis[-2]) < self.config.tol:
                max_idx = np.argmax(np.abs(system.dae.xy))
                name = system.dae.xy_name[max_idx]
                logger.error('Mismatch is not correctable possibly due to large load-generation imbalance.')
                logger.error('Largest mismatch on equation associated with <%s>', name)
            else:
                logger.error('Power flow failed after %d iterations for "%s".', self.niter + 1, system.files.case)

        else:
            logger.info('Converged in %d iterations in %s.', self.niter + 1, s1)

            # make a copy of power flow solutions
            self.x_sol = system.dae.x.copy()
            self.y_sol = system.dae.y.copy()

            if self.config.init_tds:
                system.TDS.init()
            if self.config.report:
                system.PFlow.report()

        system.exit_code = 0 if self.converged else 1
        return self.converged

    def report(self):
        """
        Write power flow report to a plain-text file.
        """
        if self.system.files.no_output is False:
            r = Report(self.system)
            r.write()

    def _set_xy(self, xy):
        """
        Helper function to set values for variables.
        """

        system = self.system
        system.dae.x[:] = xy[:system.dae.n]
        system.dae.y[:] = xy[system.dae.n:]
        system.vars_to_models()

    def _fg_wrapper(self, xy):
        """
        Wrapper for algebraic equations to be used with Newton-Krylov general solver

        Parameters
        ----------
        xy

        Returns
        -------

        """
        self._set_xy(xy)
        self.fg_update()

        return self.system.dae.fg

    def fg_update(self):
        """
        Evaluate the limiters and residual equations.
        """

        system = self.system

        system.dae.clear_fg()
        system.l_update_var(self.models, niter=self.niter, err=self.mis[-1])
        system.s_update_var(self.models)
        system.f_update(self.models)
        system.g_update(self.models)
        system.l_update_eq(self.models, niter=0)
        system.fg_to_dae()

    def newton_krylov(self, verbose=True):
        """
        Full Newton-Krylov method from SciPy.

        Warnings
        --------
        The result might be wrong if discrete are in use!

        Parameters
        ----------
        verbose
            True if verbose.

        Returns
        -------
        bool
            Convergence status
        """

        system = self.system
        v0 = system.dae.xy

        try:
            ret = newton_krylov(self._fg_wrapper, v0, verbose=verbose)
            self._set_xy(ret)
            self.converged = True

        except ValueError as e:
            logger.error('Mismatch is not correctable. Equations may be unsolvable.')
            logger.error(e)
            self.converged = False

        return self.converged
