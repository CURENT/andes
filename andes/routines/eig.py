import logging
from math import ceil

import numpy.linalg
from cvxopt import matrix, spmatrix, mul, div
from cvxopt.lapack import gesv
from matplotlib import pyplot as plt

from .base import RoutineBase
from andes.config.eig import Eig
from andes.consts import pi
from andes.formats.txt import dump_data
from andes.utils.solver import Solver
from andes.utils import elapsed

logger = logging.getLogger(__name__)
__cli__ = 'eig'


class EIG(RoutineBase):
    """
    Eigenvalue analysis routine
    """
    def __init__(self, system, rc=None):
        self.system = system
        self.solver = Solver(system.config.sparselib)
        self.config = Eig(rc=rc)

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

        Gyx = matrix(system.dae.Gx)
        self.solver.linsolve(system.dae.Gy, Gyx)

        self.As = matrix(system.dae.Fx - system.dae.Fy * Gyx)
        return self.As

    def calc_eigvals(self):
        """
        Solve eigenvalues of the state matrix ``self.As``

        Returns
        -------
        None
        """
        self.eigs = numpy.linalg.eigvals(self.As)

        return self.eigs

    def calc_part_factor(self):
        """
        Compute participation factor of states in eigenvalues

        Returns
        -------

        """
        mu, N = numpy.linalg.eig(self.As)
        N = matrix(N)
        n = len(mu)
        idx = range(n)

        W = matrix(spmatrix(1.0, idx, idx, (n, n), N.typecode))
        gesv(N, W)

        partfact = mul(abs(W.T), abs(N))

        b = matrix(1.0, (1, n))
        WN = b * partfact
        partfact = partfact.T

        for item in idx:
            mu_real = mu[item].real
            mu_imag = mu[item].imag
            mu[item] = complex(round(mu_real, 4), round(mu_imag, 4))
            partfact[item, :] /= WN[item]

        # participation factor
        self.mu = matrix(mu)
        self.part_fact = matrix(partfact)

        return self.mu, self.part_fact

    def run(self):
        ret = False
        system = self.system

        if system.pflow.solved is False:
            logger.warning(
                'Power flow not solved. Eigenvalue analysis will not continue.')
            return ret
        elif system.dae.n == 0:
            logger.warning('No dynamic model. Eivgenvalue analysis will not continue.')
            return ret

        t1, s = elapsed()
        logger.info('-> Eigenvalue Analysis:')

        system.dae.factorize = True
        exec(system.call.int)

        self.calc_state_matrix()
        self.calc_part_factor()

        self.dump_results()
        self.plot_results()
        ret = True

        t2, s = elapsed(t1)
        logger.info('Eigenvalue analysis finished in {:s}.'.format(s))

        return ret

    def plot_results(self):

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

        if self.config.plot:
            fig, ax = plt.subplots()
            ax.scatter(n_mu_real, n_mu_imag, marker='x', s=26, color='green')
            ax.scatter(z_mu_real, z_mu_imag, marker='o', s=26, color='orange')
            ax.scatter(p_mu_real, p_mu_imag, marker='x', s=26, color='red')
            plt.show()

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
            var_assoc.append(system.varname.unamex[name_idx])

        pf = []
        for prow in range(neig):
            temp_row = []
            for pcol in range(neig):
                temp_row.append(round(partfact[prow, pcol], 5))
            pf.append(temp_row)

        text.append(system.report.info)
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
                rowname.append(system.varname.unamex)
                data.append(pf[start:end])

        dump_data(text, header, rowname, data, system.files.eig)
        logger.info('report saved.')
