"""
Module for eigenvalue analysis.
"""

import logging
from math import ceil, pi
from typing import Iterable

import numpy as np
import scipy.io
from scipy.linalg import solve

from andes.io.txt import dump_data
from andes.plot import set_latex, set_style
from andes.routines.base import BaseRoutine, check_conn_before_init
from andes.shared import div, matrix, plt, sparse, spdiag, spmatrix
from andes.utils.misc import elapsed
from andes.variables.report import report_info

logger = logging.getLogger(__name__)
DPI = None


class EIG(BaseRoutine):
    """
    Eigenvalue analysis routine
    """

    def __init__(self, system, config):
        super().__init__(system=system, config=config)

        self.config.add(plot=0, tol=1e-6, gy_tol=1e-6)
        self.config.add_extra("_help",
                              plot="show plot after computation",
                              tol="numerical tolerance to treat eigenvalues as zeros",
                              gy_tol="row norm threshold for eliminating dead algebraic variables")

        self.config.add_extra("_alt", plot=(0, 1))

        # internal flags and storage
        self.As = None     # reduced state matrix
        self.mu = None     # eigenvalues
        self.N = None      # right eigenvectors
        self.W = None      # left eigenvectors
        self.pfactors = None

        # --- related to states with zero time constants (zs) ---
        self.zstate_idx = np.array([], dtype=int)

        # --- related to dead algebraic variables ---
        self.dead_algeb_idx = np.array([], dtype=int)

        # --- statistics --
        self.n_positive = 0
        self.n_zeros = 0
        self.n_negative = 0

        self.x_name = []
        self.x_tex_name = []

    def calc_As(self, dense=True):
        r"""
        Return state matrix and store to ``self.As``.

        Notes
        -----
        For systems in the mass-matrix formulation,

        .. math ::

            T \dot{x} = f(x, y) \\
            0 = g(x, y)

        Assume `T` is non-singular, the state matrix is calculated from

        .. math ::

            A_s = T^{-1} (f_x - f_y * g_y^{-1} * g_x)

        States with zero time constants satisfy :math:`0 = f(x, y)` and
        are treated as algebraic equations.  They are folded into the
        algebraic system before reduction so that a single Schur
        complement handles both original algebraic variables and
        zero-Tf states uniformly.

        Returns
        -------
        kvxopt.matrix
            state matrix
        """
        dae = self.system.dae

        self.find_zero_states()
        self.x_name = np.array(dae.x_name)
        self.x_tex_name = np.array(dae.x_tex_name)

        fx, fy, gx, gy = dae.fx, dae.fy, dae.gx, dae.gy
        Tf = dae.Tf

        # Fold zero-Tf states into the algebraic system
        if len(self.zstate_idx) > 0:
            fx, fy, gx, gy, Tf = self._fold_zstates(fx, fy, gx, gy, Tf)

        # Find and eliminate dead algebraic variables (including dead zero-Tf states)
        self.find_dead_algebs(gy, gx)

        # Extract state constraints (0 = C δx) before dead rows are removed
        C = self._extract_state_constraints(gx)

        if len(self.dead_algeb_idx) > 0:
            fx, fy, gx, gy = self._eliminate_algebs(fx, fy, gx, gy)

        # Regularize dead columns (variables absent from all equations)
        self._regularize_dead_columns(gy)

        self.As = self._reduce(fx, fy, gx, gy, Tf, dense=dense)

        # Use state constraints to eliminate dependent states
        if C is not None:
            self.As = self._apply_state_constraints(self.As, C)

        if len(self.x_name) < dae.n:
            n_as = len(self.x_name)
            logger.info("State matrix is %d x %d (reduced from %d states).", n_as, n_as, dae.n)

        return self.As

    def _reduce(self, fx, fy, gx, gy, Tf, dense=True):
        """
        Reduce algebraic equations (or states associated with zero time constants).

        Returns
        -------
        spmatrix
            The reduced state matrix
        """
        self.gyx = matrix(gx)
        self.solver.linsolve(gy, self.gyx)

        Tfnz = Tf + np.ones_like(Tf) * np.equal(Tf, 0.0)
        iTf = spdiag((1 / Tfnz).tolist())

        self.fxy = (fx - fy * self.gyx)

        if dense:
            return iTf * self.fxy
        else:
            return sparse(iTf * self.fxy)

    def _fold_zstates(self, fx, fy, gx, gy, Tf):
        """
        Fold zero-Tf states into the algebraic system.

        States with ``Tf=0`` satisfy ``0 = f(x, y)`` — algebraic, not
        differential.  This method moves their equation rows from
        ``(fx, fy)`` into an expanded ``(gx, gy)`` and returns reduced
        Jacobians for the remaining (non-zero-Tf) differential states.

        Parameters
        ----------
        fx, fy, gx, gy : spmatrix
            Original Jacobian sub-matrices (n×n, n×m, m×n, m×m).
        Tf : np.ndarray
            Time-constant vector (length n).

        Returns
        -------
        fx', fy', gx', gy' : spmatrix
            Restructured Jacobians (n'×n', n'×m', m'×n', m'×m')
            where n' = n - nz, m' = m + nz.
        Tf' : np.ndarray
            Time constants for non-zero-Tf states only.
        """
        n = fx.size[0]
        m = gy.size[0]
        zs = self.zstate_idx
        nz = len(zs)

        nz_mask = np.ones(n, dtype=bool)
        nz_mask[zs] = False
        nz_idx = np.where(nz_mask)[0]
        n_nz = len(nz_idx)

        # Selection matrices for non-zero-Tf states (rows/cols of fx)
        Sx, Sx_T = self._selection_matrices(list(nz_idx), n)
        # Selection matrices for zero-Tf states (rows/cols of fx)
        Sz, Sz_T = self._selection_matrices(list(zs), n)

        # Build expanded Jacobians:
        #   fx' = Sx * fx * Sx_T                        (n' x n')
        #   fy' = [Sx * fy | Sx * fx * Sz_T]            (n' x m')
        #   gx' = [gx * Sx_T ; Sz * fx * Sx_T]          (m' x n')
        #   gy' = [[gy, gx * Sz_T], [Sz * fy, Sz * fx * Sz_T]]  (m' x m')

        fx_new = Sx * fx * Sx_T                     # n' x n'

        fy_left = Sx * fy                           # n' x m
        fy_right = Sx * fx * Sz_T                   # n' x nz
        fy_new = sparse([[fy_left], [fy_right]])    # n' x (m + nz)

        gx_top = gx * Sx_T                          # m  x n'
        gx_bot = Sz * fx * Sx_T                     # nz x n'
        gx_new = sparse([gx_top, gx_bot])           # (m + nz) x n'

        gy_new = sparse([[gy,       Sz * fy],       # (m + nz) x (m + nz)
                         [gx * Sz_T, Sz * fx * Sz_T]])

        # Keep only non-zero-Tf state names
        self.x_name = self.x_name[nz_idx]
        self.x_tex_name = self.x_tex_name[nz_idx]

        return fx_new, fy_new, gx_new, gy_new, Tf[nz_idx]

    def calc_eig(self, As=None):
        """
        Calculate eigenvalues and right eigen vectors.

        This function is a wrapper to ``np.linalg.eig``.
        Results are returned but not stored to ``EIG``.

        Returns
        -------
        np.array(dtype=complex)
            eigenvalues
        np.array()
            right eigenvectors
        """
        if As is None:
            As = self.As

        # `mu`: eigenvalues, `N`: right eigenvectors with each column corr. to one eigvalue
        mu, N = np.linalg.eig(As)

        return mu, N

    def _store_stats(self):
        """
        Count and store the number of eigenvalues with positive, zero,
        and negative real parts using ``self.mu``.
        """

        mu_real = self.mu.real

        self.n_positive = np.count_nonzero(mu_real > self.config.tol)
        self.n_zeros = np.count_nonzero(abs(mu_real) <= self.config.tol)
        self.n_negative = np.count_nonzero(mu_real < self.config.tol)

        return True

    def calc_pfactor(self, As=None):
        """
        Compute participation factor of states in eigenvalues.

        Each row in the participation factor correspond to one state,
        and each column correspond to one mode.

        Parameters
        ----------
        As : np.array or None
            State matrix to process. If None, use ``self.As``.

        Returns
        -------
        np.array(dtype=complex)
            eigenvalues
        np.array
            participation factor matrix
        """

        mu, N = self.calc_eig(As)

        n_state = len(mu)

        # --- calculate the left eig vector and store to ``W```
        #   based on orthogonality that `W.T @ N = I`,
        #   left eigenvector is `inv(N)^T`

        Weye = np.eye(n_state)
        WT = solve(N, Weye, overwrite_b=True)
        W = WT.T

        # --- calculate participation factor ---
        pfactor = np.abs(W) * np.abs(N)
        b = np.ones(n_state)
        W_abs = b @ pfactor
        pfactor = pfactor.T

        # --- normalize participation factor ---
        for item in range(n_state):
            pfactor[:, item] /= W_abs[item]
        pfactor = np.round(pfactor, 5)

        return mu, pfactor, N, W

    def summary(self):
        """
        Print out a summary to ``logger.info``.
        """

        out = list()
        out.append('')
        out.append('-> Eigenvalue Analysis:')
        out_str = '\n'.join(out)
        logger.info(out_str)

    def find_zero_states(self):
        """
        Find the indices of states associated with zero time constants in ``x``.
        """

        system = self.system
        self.zstate_idx = np.array([], dtype=int)

        if sum(system.dae.Tf != 0) != len(system.dae.Tf):

            self.zstate_idx = np.where(system.dae.Tf == 0)[0]
            logger.debug("%d states are associated with zero time constants.", len(self.zstate_idx))
            logger.debug([system.dae.x_name[i] for i in self.zstate_idx])


    @staticmethod
    def _sparse_row_norms(mat):
        """Compute L2 row norms of a sparse matrix using triplet data."""

        m = mat.size[0]
        row_norm_sq = np.zeros(m)
        for i, v in zip(mat.I, mat.V):
            row_norm_sq[i] += v * v
        return np.sqrt(row_norm_sq)

    @staticmethod
    def _sparse_col_norms(mat):
        """Compute L2 column norms of a sparse matrix using triplet data."""

        n = mat.size[1]
        col_norm_sq = np.zeros(n)
        for j, v in zip(mat.J, mat.V):
            col_norm_sq[j] += v * v
        return np.sqrt(col_norm_sq)

    @staticmethod
    def _selection_matrices(alive, total):
        """Build rectangular selection matrix ``S`` and its transpose.

        ``S`` is (n_alive x total) and selects only the *alive* rows/cols.
        """

        n = len(alive)
        alive_int = [int(i) for i in alive]
        S = spmatrix([1.0] * n, list(range(n)), alive_int, (n, total))
        S_T = spmatrix([1.0] * n, alive_int, list(range(n)), (total, n))
        return S, S_T

    def find_dead_algebs(self, gy, gx):
        """
        Find algebraic variables with near-zero ``gy`` rows.

        These variables have degenerate equations (only ``diag_eps``
        on the diagonal) and cause ``gy`` to be singular.  They are
        eliminated before computing the state matrix.

        Dead rows with nonzero ``gx`` coupling encode state constraints
        ``0 = C δx``, which are extracted by ``_extract_state_constraints``
        and used to reduce the state dimension in
        ``_apply_state_constraints``.

        Parameters
        ----------
        gy : spmatrix
            The algebraic Jacobian to scan.  After ``_fold_zstates`` this
            includes both original algebraic variables and zero-Tf states.
        gx : spmatrix
            The algebraic-to-state Jacobian, checked for state constraints.
        """

        self.dead_algeb_idx = np.array([], dtype=int)

        dead = np.where(self._sparse_row_norms(gy) < self.config.gy_tol)[0]

        if len(dead) > 0:
            self.dead_algeb_idx = dead
            logger.debug("%d algebraic variables have near-zero gy coupling "
                         "and will be eliminated.", len(dead))

            # Check if any eliminated rows encode state constraints (gx ≠ 0)
            gx_norms = self._sparse_row_norms(gx)
            coupled = dead[gx_norms[dead] >= self.config.gy_tol]
            if len(coupled) > 0:
                logger.debug(
                    "%d of these encode state constraints (nonzero gx) "
                    "and will be used to reduce the state dimension.",
                    len(coupled))

    def _eliminate_algebs(self, fx, fy, gx, gy):
        """
        Remove dead algebraic variables from the Jacobian matrices.

        Builds a rectangular selection matrix ``S`` (m_alive x m) that
        keeps only the alive algebraic rows/cols, then returns the
        reduced Jacobians.

        Returns ``fx, fy_alive, gx_alive, gy_alive``.
        """

        m = gy.size[0]
        alive = sorted(set(range(m)) - set(self.dead_algeb_idx))
        S, S_T = self._selection_matrices(alive, m)

        return fx, fy * S_T, S * gx, S * gy * S_T

    def _regularize_dead_columns(self, gy):
        """
        Regularize algebraic variables with near-zero ``gy`` columns.

        A near-zero column means the variable does not appear in any
        algebraic equation.  Setting its diagonal to 1.0 makes ``gy``
        non-singular and forces the variable's perturbation to zero
        in the Schur complement, which is physically correct for an
        unconstrained variable.

        This handles the complementary case to ``find_dead_algebs``
        (dead rows).  Typical cause: a zero-Tf state whose
        self-coupling vanishes (e.g. ``Lag2ndOrd.x`` with ``T1=0``).
        """
        col_norms = self._sparse_col_norms(gy)
        dead_cols = np.where(col_norms < self.config.gy_tol)[0]

        if len(dead_cols) > 0:
            for j in dead_cols:
                gy[int(j), int(j)] = 1.0
            logger.debug(
                "%d algebraic variables with near-zero gy columns "
                "regularized (diagonal set to 1).", len(dead_cols))

    def _extract_state_constraints(self, gx):
        """
        Extract state constraint rows from dead algebraic variables.

        Dead algebraic rows with ``gy ≈ 0`` but ``gx ≠ 0`` encode
        linear constraints ``0 = C δx`` on the state perturbations.
        This method extracts those rows from ``gx`` as a constraint
        matrix ``C`` before dead rows are eliminated.

        Parameters
        ----------
        gx : spmatrix
            Algebraic-to-state Jacobian (possibly expanded by folding).

        Returns
        -------
        spmatrix or None
            Constraint matrix ``C`` of shape (nc × n), or ``None``
            if no dead rows have nonzero ``gx`` coupling.
        """
        if len(self.dead_algeb_idx) == 0:
            return None

        gx_norms = self._sparse_row_norms(gx)
        coupled_mask = gx_norms[self.dead_algeb_idx] >= self.config.gy_tol
        coupled = self.dead_algeb_idx[coupled_mask]

        if len(coupled) == 0:
            return None

        m = gx.size[0]
        S, _ = self._selection_matrices([int(i) for i in coupled], m)
        return S * gx

    def _apply_state_constraints(self, As, C):
        r"""
        Reduce the state matrix using algebraic constraints on states.

        Dead algebraic rows with ``gy ≈ 0`` but ``gx ≠ 0`` impose
        constraints ``0 = C δx`` on the linearized perturbations.
        Each constraint eliminates one dependent state:

        .. math::

            \delta x_d = -C_d^{-1} C_f \, \delta x_f

        where *d* denotes dependent (pivot) states and *f* the remaining
        free states.  The reduced state matrix is

        .. math::

            A_{\text{red}} = A_{\text{full}}[\text{free}, :] \; R

        where ``R`` is the substitution matrix satisfying ``C R = 0``.

        Parameters
        ----------
        As : matrix or ndarray
            Full state matrix (n × n) from ``_reduce``, already
            includes ``T^{-1}``.
        C : spmatrix
            Constraint matrix (nc × n) from dead algebraic rows.

        Returns
        -------
        ndarray
            Reduced state matrix of size (n - nc) × (n - nc).
        """
        n = As.size[0]   # kvxopt matrix .size is (rows, cols)
        nc = C.size[0]

        # Convert sparse C to dense numpy
        C_np = np.zeros((nc, n))
        for k in range(len(C.V)):
            C_np[C.I[k], C.J[k]] = C.V[k]

        # Identify pivot (dependent) state for each constraint row.
        # Pick the largest unused coefficient as pivot to ensure C_d
        # is well-conditioned.
        dep_list = []
        used = set()
        for i in range(nc):
            row_abs = np.abs(C_np[i, :])
            for pivot in np.argsort(-row_abs):
                if pivot not in used and row_abs[pivot] > 0:
                    dep_list.append(int(pivot))
                    used.add(pivot)
                    break
            else:
                logger.warning("No valid pivot for constraint %d. "
                               "Skipping state constraint reduction.", i)
                return As

        dep_states = np.array(dep_list, dtype=int)
        free_mask = np.ones(n, dtype=bool)
        free_mask[dep_states] = False
        free_states = np.where(free_mask)[0]
        n_f = len(free_states)

        # Build substitution: δx_dep = -C_d^{-1} C_f δx_free
        C_d = C_np[:, dep_states]
        C_f = C_np[:, free_states]
        G = -solve(C_d, C_f)

        # Substitution matrix R: δx = R δx_free
        R = np.zeros((n, n_f))
        R[free_states, np.arange(n_f)] = 1.0
        R[dep_states, :] = G

        # Convert As to numpy and project
        As_np = np.array(As)
        if As_np.ndim == 1:
            As_np = As_np.reshape((n, n), order='F')

        As_red = As_np[free_states, :] @ R

        self.x_name = self.x_name[free_states]
        self.x_tex_name = self.x_tex_name[free_states]

        logger.debug("%d state constraints applied, reducing %d states to %d.",
                     nc, n, n_f)

        return As_red

    def sweep(self, params, idxes, values):
        """
        Parameter sweep for root loci plot.

        Parameters
        ----------
        params : list of NumParam or ConstService
            list of parameters indices to sweep. For example, ``[ss.GENCLS.M]``
            for GENCLS.M. To update ``ss.GENCLS.M`` for two generators,
            ``params`` should be set to ``[ss.GENCLS.M, ss.GENCLS.M]``.
        idxes : list of int or str
            list of indices to sweep. For example, ``["GENCLS_1", "GENCLS_2"]``
            for the indices of GENCLS whose corresponding parameter will be
            updated. The length of ``idxes`` must match that of ``params`` and
            ``values``.
        values: list of lists
            New values of the parameters. Each element in ``values`` is a list
            for the corresponding element in ``params`` and ``idxes``.

        Examples
        --------
        To apply 10 parameters evenly spaced between 1 and 10 to
        ``ss.GENCLS.M`` of ``GENCLS_1``, do

        .. code-block:: python

            ret = ss.EIG.sweep(ss.GENCLS.M, "GENCLS_1", np.linspace(1, 2, 10))

        This is equivalent to the following just for convenience.

        .. code-block:: python

            ret = ss.EIG.sweep([ss.GENCLS.M],
                               ["GENCLS_1"],
                               [np.linspace(1, 2, 10)])

        Returns
        -------
        dict
            A dictionary of the results where the keys are 0-indexed count of
            parameter set, and the values are dictionaries. Each value
            dictionary contains a ``mu`` field for the eigenvalues.

        """

        ret = False
        results = dict()

        if not isinstance(params, Iterable):
            params = (params, )

        if not isinstance(values, Iterable):
            logger.error("values must be a list or tuple.")
            return ret
        elif not isinstance(values[0], Iterable):
            values = (values, )

        if isinstance(idxes, str):
            idxes = (idxes, )
        elif not isinstance(idxes, Iterable):
            idxes = (idxes, )

        if len(params) != len(values):
            logger.error("params and values must have the same length.")
            return ret

        # check if all values are of the same length
        if len(values) > 1:
            len0 = len(values[0])
            idx = 1
            for value in values[1:]:
                len1 = len(value)
                if len1 == len0:
                    len0 = len1
                    idx += 1
                    continue
                logger.error(f"value[{idx}] is of length {len1} =/ previous length {len0}.")
                return ret

        # get position for the parameters
        positions = list()
        param_names = list()
        for param, idx in zip(params, idxes):
            positions.append(param.owner.idx2uid(idx))
            param_names.append(param.name)

        # set parameters and run cases
        for count, val in enumerate(zip(*values)):
            logger.debug(f"Parameter sweep: round={count}")

            for idx, (param, pos) in enumerate(zip(params, positions)):
                param.v[pos] = val[idx]
                logger.debug(f"Set {param.name} = {param.v[pos]}")

            self.system.TDS.init()
            self.system.TDS.itm_step()
            self.calc_As()
            mu, N = self.calc_eig(self.As)

            self.mu, self.N = mu, N  # save to `EIG` for writing if needed

            results[count] = dict(param_values=val, mu=mu,)

        return results

    def plot_root_loci(self, results, eig_indices, ax=None, dpi=None, figsize=None,
                       draw_line=False, arrow_threshold=0.2, **kwargs):
        """
        Plot the root loci.

        Markers increase in size for the first parameter through the last.

        Parameters
        ----------
        results : dict
            Eigenvalue results from parameter sweeping
        eig_indices : Iterable
            A list of eigenvalue indices to plot. The indices are 0-based,
            whereas the indices in the eigenvalue analysis report are 1-based.
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, create a new figure.
        dpi : int or None
            DPI of the figure. If None, use the default DPI.
        figsize : tuple or None
            Figure size. If None, use the default size.
        draw_line : bool, optional, False by default
            If True, draw lines to connect the roots. Note that due to the
            non-fixed ordering of eigenvalues, lines will largely connect
            different modesl
        arrow_threshold : float
            Threshold for plotting arrows. If the begin and end points of a
            locus is shorter than this threshold, no arrow is plotted.

        Examples
        --------
        To plot the root loci of the first two eigenvalues, do

        .. code-block:: python

            fig, ax = ss.EIG.plot_root_loci(ret, [0, 1])

        where ``ret`` is the return of :py:meth:`andes.routines.eig.EIG.sweep`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot.
        matplotlib.axes.Axes
            Axes of the plot.
        """

        if ax is None:
            fig = plt.figure(dpi=dpi, figsize=figsize)
            ax = plt.gca()

        loci_list = list()

        # extract data into an array
        for data in results.values():
            mu = data['mu'][eig_indices]
            if not isinstance(mu, Iterable):
                mu = np.array([mu])

            loci_list.append(mu)

        loci_data = np.array(loci_list)
        npoints, nloci = loci_data.shape

        # plot the eigenvalues - markers increase in size from beginning to the
        # end.

        for i in range(npoints):
            s = 10 + 40 * i / (npoints - 1)
            fig, ax = self.plot(mu=loci_data[i, :], s=s, fig=fig, ax=ax, show=False, **kwargs)

        # Note:
        # We are not able to plot a loci by connecting the eigenvalues because
        # the order of the returned eigenvalues are not and cannot be guaranteed.

        # draw solid lines to connect the roots
        if draw_line:
            ax.plot(loci_data.real, loci_data.imag, color='grey', linewidth=1)

            # draw arrows using the middle points
            if npoints > 1:
                leftp = np.floor(npoints / 2 - 1).astype(int)
                rightp = leftp + 1

                loci_r = loci_data.real
                loci_i = loci_data.imag

                for i in range(nloci):
                    # skip drawing arrows if the distance is too small
                    if np.abs(loci_data[0, i] - loci_data[-1, i]) < arrow_threshold:
                        continue

                    left_r = loci_r[leftp, i]
                    left_i = loci_i[leftp, i]
                    right_r = loci_r[rightp, i]
                    right_i = loci_i[rightp, i]

                    ax.annotate("", xy=(right_r, right_i), xytext=(left_r, left_i),
                                arrowprops=dict(arrowstyle="simple", color='black'))

        return fig, ax

    def _pre_check(self):
        """
        Helper function for pre-computation checks.
        """

        system = self.system
        status = True

        if system.PFlow.converged is False:
            logger.warning('Power flow not solved. Eig analysis will not continue.')
            status = False

        if system.TDS.initialized is False:
            system.TDS.init()
            system.TDS.itm_step()

        elif system.dae.n == 0:
            logger.error('No dynamic model. Eig analysis will not continue.')
            status = False

        return status

    @check_conn_before_init
    def run(self, **kwargs):
        """
        Run small-signal stability analysis.
        """

        succeed = False
        system = self.system

        if not self._pre_check():
            system.exit_code += 1
            return False

        self.summary()
        t1, s = elapsed()

        self.calc_As()
        self.mu, self.pfactors, self.N, self.W = self.calc_pfactor()
        self._store_stats()
        t2, s = elapsed(t1)
        self.exec_time = t2 - t1

        logger.info(self.stats())
        logger.info('Eigenvalue analysis finished in {:s}.'.format(s))

        if not self.system.files.no_output:
            self.report()
            if system.options.get('state_matrix'):
                self.export_mat()

        if self.config.plot:
            self.plot()

        succeed = True

        if not succeed:
            system.exit_code += 1
        return succeed

    def stats(self):
        """
        Return statistics of results in a string.
        """
        out = list()
        out.append('  Positive  %6g' % self.n_positive)
        out.append('  Zeros     %6g' % self.n_zeros)
        out.append('  Negative  %6g' % self.n_negative)

        return '\n'.join(out)

    def plot(self, mu=None, fig=None, ax=None,
             left=-6, right=0.5, ymin=-8, ymax=8, damping=0.05,
             line_width=0.5, s=40, dpi=DPI, figsize=None, base_color='black',
             show=True, latex=True, style='default',
             ):
        """
        Plot utility for eigenvalues in the S domain.

        Parameters
        ----------
        mu : array, optional
            an array of complex eigenvalues
        fig : figure handl, optional
            existing matplotlib figure handle
        ax : axis handle, optional
            existing axis handle
        left : int, optional
            left tick for the x-axis, by default -6
        right : float, optional
            right tick, by default 0.5
        ymin : int, optional
            bottom tick, by default -8
        ymax : int, optional
            top tick, by default 8
        damping : float, optional
            damping value for which the dash plots are drawn
        line_width : float, optional
            default line width, by default 0.5
        s : float or array-like, shape (n, ), optional
            The marker size in points**2
        dpi : int, optional
            figure dpi
        figsize : [type], optional
            default figure size, by default None
        base_color : str, optional
            base color for negative eigenvalues
        show : bool, optional
            True to show figure after plot, by default True
        latex : bool, optional
            True to use latex, by default True

        Returns
        -------
        figure
            matplotlib figure object
        axis
            matplotlib axis object

        """

        set_style(style)

        if mu is None:
            mu = self.mu
        mu_real = mu.real
        mu_imag = mu.imag
        p_mu_real, p_mu_imag = list(), list()
        z_mu_real, z_mu_imag = list(), list()
        n_mu_real, n_mu_imag = list(), list()

        for re, im in zip(mu_real, mu_imag):
            if abs(re) <= self.config.tol:
                z_mu_real.append(re)
                z_mu_imag.append(im)
            elif re > self.config.tol:
                p_mu_real.append(re)
                p_mu_imag.append(im)
            elif re < -self.config.tol:
                n_mu_real.append(re)
                n_mu_imag.append(im)

        if latex:
            set_latex()

        if fig is None or ax is None:
            fig = plt.figure(dpi=dpi, figsize=figsize)
            ax = plt.gca()

        ax.scatter(z_mu_real, z_mu_imag, marker='o', s=s, linewidth=0.5, facecolors='none', edgecolors='green')
        ax.scatter(n_mu_real, n_mu_imag, marker='x', s=s, linewidth=0.5, color=base_color)
        ax.scatter(p_mu_real, p_mu_imag, marker='x', s=s, linewidth=0.5, color='red')

        # axes lines
        ax.axhline(linewidth=0.5, color='grey', linestyle='--')
        ax.axvline(linewidth=0.5, color='grey', linestyle='--')

        # TODO: Improve the damping and range
        # --- plot 5% damping lines ---
        xin = np.arange(left, 0, 0.01)
        yneg = xin / damping
        ypos = - xin / damping

        ax.plot(xin, yneg, color='grey', linewidth=line_width, linestyle='--')
        ax.plot(xin, ypos, color='grey', linewidth=line_width, linestyle='--')
        # --- damping lines end ---

        if latex:
            ax.set_xlabel('Real [$s^{-1}$]')
            ax.set_ylabel('Imaginary [$s^{-1}$]')
        else:
            ax.set_xlabel('Real [s -1]')
            ax.set_ylabel('Imaginary [s -1]')

        ax.set_xlim(left=left, right=right)
        ax.set_ylim(ymin, ymax)

        if show is True:
            plt.show()
        return fig, ax

    def export_mat(self):
        """
        Export state matrix to a ``<CaseName>_As.mat`` file with the variable name ``As``, where
        ``<CaseName>`` is the test case name.

        State variable names are stored in variables ``x_name`` and ``x_tex_name``.

        Returns
        -------
        bool
            True if successful
        """
        system = self.system
        out = {'As': self.As,
               'x_name': np.array(self.x_name, dtype=object),
               'x_tex_name': np.array(self.x_tex_name, dtype=object),
               }

        scipy.io.savemat(system.files.mat, mdict=out)
        logger.info('State matrix saved to "%s"', system.files.mat)
        return True

    def post_process(self):
        """
        Post processing of eigenvalues.
        """

        # --- statistics ---
        n_states = len(self.mu)
        mu_real = self.mu.real

        numeral = [''] * n_states
        for idx, item in enumerate(range(n_states)):
            if abs(mu_real[idx]) <= self.config.tol:
                marker = '*'
            elif mu_real[idx] > self.config.tol:
                marker = '**'
            else:
                marker = ''
            numeral[idx] = '#' + str(idx + 1) + marker

        # compute frequency, un-damped frequency and damping
        freq = np.zeros(n_states)
        ufreq = np.zeros(n_states)
        damping = np.zeros(n_states)

        for idx, item in enumerate(self.mu):
            if item.imag == 0:
                freq[idx] = 0
                ufreq[idx] = 0
                damping[idx] = 0
            else:
                ufreq[idx] = abs(item) / 2 / pi
                freq[idx] = abs(item.imag / 2 / pi)
                damping[idx] = -div(item.real, abs(item)) * 100

        return freq, ufreq, damping, numeral

    def report(self, x_name=None, **kwargs):
        """
        Save eigenvalue analysis reports.

        Returns
        -------
        None
        """
        if x_name is None:
            x_name = self.x_name

        n_states = len(self.mu)
        mu_real = self.mu.real
        mu_imag = self.mu.imag
        freq, ufreq, damping, numeral = self.post_process()

        # obtain most associated variables
        var_assoc = []
        for prow in range(n_states):
            temp_row = self.pfactors[prow, :]
            name_idx = list(temp_row).index(max(temp_row))
            var_assoc.append(x_name[name_idx])

        text, header, rowname, data = list(), list(), list(), list()

        # opening info section
        text.append(report_info(self.system))
        header.append(None)
        rowname.append(None)
        data.append(None)
        text.append('')

        text.append('EIGENVALUE ANALYSIS REPORT')
        header.append([])
        rowname.append([])
        data.append([])

        text.append('STATISTICS\n')
        header.append([''])
        rowname.append(['Positives', 'Zeros', 'Negatives'])
        data.append((self.n_positive, self.n_zeros, self.n_negative))

        text.append('EIGENVALUE DATA\n')
        header.append([
            'Most Associated',
            'Real',
            'Imag.',
            'Damped Freq.',
            'Frequency',
            'Damping [%]'])
        rowname.append(numeral)
        data.append(
            [var_assoc,
             list(mu_real),
             list(mu_imag),
             freq,
             ufreq,
             damping])

        n_cols = 7  # columns per block
        n_block = int(ceil(n_states / n_cols))

        if n_block <= 100:
            for idx in range(n_block):
                start = n_cols * idx
                end = n_cols * (idx + 1)
                text.append('PARTICIPATION FACTORS [{}/{}]\n'.format(
                    idx + 1, n_block))
                header.append(numeral[start:end])
                rowname.append(x_name)
                data.append(self.pfactors[start:end, :])

        dump_data(text, header, rowname, data, self.system.files.eig)
        logger.info('Report saved to "%s".', self.system.files.eig)
