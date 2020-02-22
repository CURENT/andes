import logging
from math import ceil, pi

from cvxopt import mul, div
from cvxopt.lapack import gesv

from andes.io.txt import dump_data
from andes.utils.misc import elapsed
from andes.routines.base import BaseRoutine
from andes.shared import np, matrix, spmatrix, plt, mpl

logger = logging.getLogger(__name__)
__cli__ = 'eig'


class EIG(BaseRoutine):
    """
    Eigenvalue analysis routine
    """
    def __init__(self, system, config):
        super().__init__(system=system, config=config)

        self.config.add(plot=0)

        # internal flags and storages
        self.As = None
        self.eigs = None
        self.mu = None
        self.part_fact = None

    def calc_state_matrix(self):
        """
        Return state matrix and store to ``self.As``

        Returns
        -------
        matrix
            state matrix
        """
        system = self.system

        gyx = matrix(system.dae.gx)
        self.solver.linsolve(system.dae.gy, gyx)

        self.As = matrix(system.dae.fx - system.dae.fy * gyx)

        # ------------------------------------------------------
        # TODO: use scipy eigs
        # self.As = sparse(self.As)
        # I = np.array(self.As.I).reshape((-1,))
        # J = np.array(self.As.J).reshape((-1,))
        # V = np.array(self.As.V).reshape((-1,))
        # self.As = csr_matrix((V, (I, J)), shape=self.As.size)
        # ------------------------------------------------------
        return self.As

    def calc_eigvals(self):
        """
        Solve eigenvalues of the state matrix ``self.As``

        Returns
        -------
        None
        """
        self.eigs = np.linalg.eigvals(self.As)
        # TODO: use scipy.sparse.linalg.eigs(self.As)

        return self.eigs

    def calc_part_factor(self):
        """
        Compute participation factor of states in eigenvalues

        Returns
        -------

        """
        mu, N = np.linalg.eig(self.As)
        # TODO: use scipy.sparse.linalg.eigs(self.As)

        N = matrix(N)
        n = len(mu)
        idx = range(n)

        mu_complex = np.array([0] * n, dtype=complex)
        W = matrix(spmatrix(1.0, idx, idx, (n, n), N.typecode))
        gesv(N, W)

        partfact = mul(abs(W.T), abs(N))

        b = matrix(1.0, (1, n))
        WN = b * partfact
        partfact = partfact.T

        for item in idx:
            mu_real = float(mu[item].real)
            mu_imag = float(mu[item].imag)
            mu_complex[item] = complex(round(mu_real, 4), round(mu_imag, 4))
            partfact[item, :] /= WN[item]

        # participation factor
        self.mu = matrix(mu_complex)
        self.part_fact = matrix(partfact)

        return self.mu, self.part_fact

    def run(self, **kwargs):
        ret = False
        system = self.system

        if system.PFlow.converged is False:
            logger.warning('Power flow not solved. Eig analysis will not continue.')
            return ret
        else:
            if system.TDS.initialized is False:
                system.TDS._initialize()
                system.TDS._implicit_step()

        if system.dae.n == 0:
            logger.warning('No dynamic model. Eig analysis will not continue.')
            return ret

        t1, s = elapsed()
        logger.info('-> Eigenvalue Analysis:')

        self.calc_state_matrix()
        self.calc_part_factor()

        self.dump_results()

        if self.config.plot:
            self.plot()
        ret = True

        _, s = elapsed(t1)
        logger.info('Eigenvalue analysis finished in {:s}.'.format(s))

        return ret

    def plot(self, left=-6, right=0.5, ymin=-8, ymax=8, damping=0.05,
             linewidth=0.5, dpi=150):
        mpl.rc('font', family='Times New Roman', size=12)

        mu_real = self.mu.real()
        mu_imag = self.mu.imag()
        p_mu_real, p_mu_imag = list(), list()
        z_mu_real, z_mu_imag = list(), list()
        n_mu_real, n_mu_imag = list(), list()

        for re, im in zip(mu_real, mu_imag):
            if re == 0:
                z_mu_real.append(re)
                z_mu_imag.append(im)
            elif re > 0:
                p_mu_real.append(re)
                p_mu_imag.append(im)
            elif re < 0:
                n_mu_real.append(re)
                n_mu_imag.append(im)

        if len(p_mu_real) > 0:
            logger.warning(
                'System is not stable due to {} positive eigenvalues.'.format(
                    len(p_mu_real)))
        else:
            logger.info(
                'System is small-signal stable in the initial neighbourhood.')

        mpl.rc('text', usetex=True)
        fig, ax = plt.subplots(dpi=dpi)
        ax.scatter(n_mu_real, n_mu_imag, marker='x', s=40, linewidth=0.5, color='black')
        ax.scatter(z_mu_real, z_mu_imag, marker='o', s=40, linewidth=0.5, facecolors='none', edgecolors='black')
        ax.scatter(p_mu_real, p_mu_imag, marker='x', s=40, linewidth=0.5, color='black')
        ax.axhline(linewidth=0.5, color='grey', linestyle='--')
        ax.axvline(linewidth=0.5, color='grey', linestyle='--')

        # plot 5% damping lines
        xin = np.arange(left, 0, 0.01)
        yneg = xin / damping
        ypos = - xin / damping

        ax.plot(xin, yneg, color='grey', linewidth=linewidth, linestyle='--')
        ax.plot(xin, ypos, color='grey', linewidth=linewidth, linestyle='--')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_xlim(left=left, right=right)
        ax.set_ylim(ymin, ymax)

        plt.show()

        return fig, ax

    def dump_results(self):
        """
        Save eigenvalue analysis reports

        Returns
        -------
        None
        """
        system = self.system
        mu = self.mu
        partfact = self.part_fact

        if system.files.no_output:
            return

        text = []
        header = []
        rowname = []
        data = []

        neig = len(mu)
        mu_real = mu.real()
        mu_imag = mu.imag()
        npositive = sum(1 for x in mu_real if x > 0)
        nzero = sum(1 for x in mu_real if x == 0)
        nnegative = sum(1 for x in mu_real if x < 0)

        numeral = []
        for idx, item in enumerate(range(neig)):
            if mu_real[idx] == 0:
                marker = '*'
            elif mu_real[idx] > 0:
                marker = '**'
            else:
                marker = ''
            numeral.append('#' + str(idx + 1) + marker)

        # compute frequency, undamped frequency and damping
        freq = [0] * neig
        ufreq = [0] * neig
        damping = [0] * neig
        for idx, item in enumerate(mu):
            if item.imag == 0:
                freq[idx] = 0
                ufreq[idx] = 0
                damping[idx] = 0
            else:
                freq[idx] = abs(item) / 2 / pi
                ufreq[idx] = abs(item.imag / 2 / pi)
                damping[idx] = -div(item.real, abs(item)) * 100

        # obtain most associated variables
        var_assoc = []
        for prow in range(neig):
            temp_row = partfact[prow, :]
            name_idx = list(temp_row).index(max(temp_row))
            var_assoc.append(system.dae.x_name[name_idx])

        pf = []
        for prow in range(neig):
            temp_row = []
            for pcol in range(neig):
                temp_row.append(round(partfact[prow, pcol], 5))
            pf.append(temp_row)

        text.append('')
        header.append([''])
        rowname.append(['EIGENVALUE ANALYSIS REPORT'])
        data.append('')

        text.append('STATISTICS\n')
        header.append([''])
        rowname.append(['Positives', 'Zeros', 'Negatives'])
        data.append([npositive, nzero, nnegative])

        text.append('EIGENVALUE DATA\n')
        header.append([
            'Most Associated', 'Real', 'Imag', 'Damped Freq.', 'Frequency',
            'Damping [%]'
        ])
        rowname.append(numeral)
        data.append(
            [var_assoc,
             list(mu_real),
             list(mu_imag), ufreq, freq, damping])

        cpb = 7  # columns per block
        nblock = int(ceil(neig / cpb))

        if nblock <= 100:
            for idx in range(nblock):
                start = cpb * idx
                end = cpb * (idx + 1)
                text.append('PARTICIPATION FACTORS [{}/{}]\n'.format(
                    idx + 1, nblock))
                header.append(numeral[start:end])
                rowname.append(system.dae.x_name)
                data.append(pf[start:end])

        dump_data(text, header, rowname, data, system.files.eig)
        logger.info(f'Report saved to <{system.files.eig}>.')
