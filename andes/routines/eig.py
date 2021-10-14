"""
Module for eigenvalue analysis.
"""

import logging
import scipy.io
import numpy as np

from math import ceil, pi
from scipy.linalg import solve

from andes.io.txt import dump_data
from andes.utils.misc import elapsed
from andes.routines.base import BaseRoutine
from andes.variables.report import report_info

from andes.plot import set_latex

from andes.shared import matrix, spmatrix, sparse, plt, mpl
from andes.shared import div, spdiag


logger = logging.getLogger(__name__)


class EIG(BaseRoutine):
    """
    Eigenvalue analysis routine
    """

    def __init__(self, system, config):
        super().__init__(system=system, config=config)

        self.config.add(plot=0, tol=1e-6)
        self.config.add_extra("_help",
                              plot="show plot after computation",
                              tol="numerical tolerance to treat eigenvalues as zeros")

        self.config.add_extra("_alt", plot=(0, 1))

        # internal flags and storage
        self.As = None     # state matrix after removing the ones associated with zero T consts
        self.Asc = None    # the original complete As without reordering
        self.mu = None     # eigenvalues
        self.N = None      # right eigenvectors
        self.W = None      # left eigenvectors
        self.pfactors = None

        # --- related to states with zero time constants (zs) ---
        self.zstate_idx = np.array([], dtype=int)
        self.nz_counts = None

        # --- statistics --
        self.n_positive = 0
        self.n_zeros = 0
        self.n_negative = 0

        self.x_name = []

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

        Returns
        -------
        kvxopt.matrix
            state matrix
        """
        dae = self.system.dae

        self.find_zero_states()
        self.x_name = np.array(dae.x_name)

        self.As = self._reduce(dae.fx, dae.fy,
                               dae.gx, dae.gy, dae.Tf,
                               dense=dense)

        if len(self.zstate_idx) > 0:
            self.Asc = self.As
            self.As = self._reduce(*self._reorder())

        return self.As

    def _reduce(self, fx, fy, gx, gy, Tf, dense=True):
        """
        Reduce algebraic equations (or states associated with zero time constants).

        Returns
        -------
        spmatrix
            The reduced state matrix
        """
        gyx = matrix(gx)
        self.solver.linsolve(gy, gyx)

        Tfnz = Tf + np.ones_like(Tf) * np.equal(Tf, 0.0)
        iTf = spdiag((1 / Tfnz).tolist())

        if dense:
            return iTf * (fx - fy * gyx)
        else:
            return sparse(iTf * (fx - fy * gyx))

    def _reorder(self):
        """
        reorder As by moving rows and cols associated with zero time constants to the end.

        Returns `fx`, `fy`, `gx`, `gy`, `Tf`.
        """
        dae = self.system.dae
        rows = np.arange(dae.n, dtype=int)
        cols = np.arange(dae.n, dtype=int)
        vals = np.ones(dae.n, dtype=float)

        swaps = []
        bidx = self.nz_counts
        for ii in range(dae.n - self.nz_counts):
            if ii in self.zstate_idx:
                while (bidx in self.zstate_idx):
                    bidx += 1
                cols[ii] = bidx
                rows[bidx] = ii
                swaps.append((ii, bidx))

        # swap the variable names
        for fr, bk in swaps:
            bk_name = self.x_name[bk]
            self.x_name[fr] = bk_name
        self.x_name = self.x_name[:self.nz_counts]

        # compute the permutation matrix for `As` containing non-states
        perm = spmatrix(matrix(vals), matrix(rows), matrix(cols))
        As_perm = perm * sparse(self.As) * perm
        self.As_perm = As_perm

        nfx = As_perm[:self.nz_counts, :self.nz_counts]
        nfy = As_perm[:self.nz_counts, self.nz_counts:]
        ngx = As_perm[self.nz_counts:, :self.nz_counts]
        ngy = As_perm[self.nz_counts:, self.nz_counts:]
        nTf = np.delete(self.system.dae.Tf, self.zstate_idx)

        return nfx, nfy, ngx, ngy, nTf

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
            logger.info("%d states are associated with zero time constants. ", len(self.zstate_idx))
            logger.debug([system.dae.x_name[i] for i in self.zstate_idx])

        self.nz_counts = system.dae.n - len(self.zstate_idx)

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
        _, s = elapsed(t1)

        logger.info('  Positive  %6g', self.n_positive)
        logger.info('  Zeros     %6g', self.n_zeros)
        logger.info('  Negative  %6g', self.n_negative)

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

    def plot(self, mu=None, fig=None, ax=None,
             left=-6, right=0.5, ymin=-8, ymax=8, damping=0.05,
             line_width=0.5, dpi=100, figsize=None, base_color='black',
             show=True, latex=True):
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
        dpi : int, optional
            figure dpi, by default 100
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

        mpl.rc('font', family='Times New Roman', size=12)

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

        ax.scatter(z_mu_real, z_mu_imag, marker='o', s=40, linewidth=0.5, facecolors='none', edgecolors='green')
        ax.scatter(n_mu_real, n_mu_imag, marker='x', s=40, linewidth=0.5, color=base_color)
        ax.scatter(p_mu_real, p_mu_imag, marker='x', s=40, linewidth=0.5, color='red')

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
               'Asc': self.Asc,
               'x_name': np.array(system.dae.x_name, dtype=object),
               'x_tex_name': np.array(system.dae.x_tex_name, dtype=object),
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
            if mu_real[idx] == 0:
                marker = '*'
            elif mu_real[idx] > 0:
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
