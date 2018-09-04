import numpy as np

from cvxopt import matrix
import logging

logger = logging.getLogger(__name__)


class VarOut(object):
    """
    Output variable value recorder

    TODO: merge in tds.py
    """

    def __init__(self, system):
        """Constructor of empty Varout object"""
        self.system = system
        self.t = []
        self.k = []
        self.vars = []
        self.dat = None
        self._mode = 'w'

    def store(self, t, step):
        """
        Record the state/algeb values at time t to self.vars
        """
        max_cache = int(self.system.tds.config.max_cache)
        if len(self.vars) >= max_cache > 0:
            self.dump()
            self.vars = list()
            self.t = list()
            self.k = list()
            logger.debug(
                'varout cache cleared at simulation t = {:g}.'.format(
                    self.system.dae.t))
            self._mode = 'a'

        self.t.append(t)
        self.k.append(step)
        self.vars.append(matrix([self.system.dae.x, self.system.dae.y]))

        if self.system.tds.config.compute_flows:
            self.system.dae.y = self.system.dae.y[:self.system.dae.m]

    def show(self):
        """
        The representation of an Varout object

        :return: the full result matrix (for use with PyCharm viewer)
        :rtype: np.array
        """
        out = []

        for item in self.vars:
            out.append(list(item))

        return np.array(out)

    def concat_t_vars(self):
        """
        Concatenate ``self.t`` with ``self.vars`` and output a single matrix
        for data dump

        :return matrix: concatenated matrix with ``self.t`` as the 0-th column
        """
        out = np.array([])

        if len(self.t) == 0:
            return out

        out = np.ndarray(shape=(0, self.vars[0].size[0] + 1))

        for t, var in zip(self.t, self.vars):
            line = [[t]]
            line[0].extend(list(var))
            out = np.append(out, line, axis=0)

        return out

    def get_xy(self, yidx, xidx=0):
        """
        Return stored data for the given indices for plot

        :param yidx: the indices of the y-axis variables(1-indexing)
        :param xidx: the index of the x-axis variables
        :return: None
        """
        assert isinstance(xidx, int)
        if isinstance(yidx, int):
            yidx = [yidx]

        t_vars = self.concat_t_vars()

        xdata = t_vars[:, xidx]
        ydata = t_vars[:, yidx]

        return xdata.tolist(), ydata.transpose().tolist()

    def dump(self):
        """
        Dump the TDS results to the output `dat` file

        :param lst: dump ``.lst`` file

        :return: succeed flag
        """
        ret = False

        if self.system.files.no_output:
            # return ``True`` because it did not fail
            return True

        if self.write_lst() and self.write_dat():
            ret = True

        return ret

    def write_dat(self):
        """
        Write ``system.Varout.vars`` to a ``.dat`` file
        :return:
        """
        ret = False
        system = self.system

        # compute the total number of columns, excluding time
        if not system.Recorder.n:
            n_vars = system.dae.m + system.dae.n
            if system.tds.config.compute_flows:
                n_vars += 2 * system.Bus.n + 4 * system.Line.n
            idx = list(range(n_vars))

        else:
            n_vars = len(system.Recorder.varout_idx)
            idx = system.Recorder.varout_idx

        template = ['{:<8g}'] + ['{:0.10f}'] * n_vars
        template = ' '.join(template)

        # format the output in a string
        out = ''
        for t, line in zip(self.t, self.vars):
            values = [t] + list(line[idx])
            out += template.format(*values) + '\n'

        try:
            with open(system.files.dat, self._mode) as f:
                f.write(out)
            ret = True

        except IOError:
            logger.error('I/O Error while writing the dat file.')

        return ret

    def write_lst(self):
        """
        Dump the variable name lst file

        :return: succeed flag
        """

        ret = False
        out = ''
        system = self.system
        dae = self.system.dae
        varname = self.system.varname
        template = '{:>6g}, {:>25s}, {:>25s}\n'

        # header line
        out += template.format(0, 'Time [s]', '$Time\ [s]$')

        # include line flow variables in algebraic variables
        nflows = 0
        if self.system.tds.config.compute_flows:
            nflows = 2 * self.system.Bus.n + \
                     4 * self.system.Line.n + \
                     2 * self.system.Area.n_combination

        # output variable indices
        if system.Recorder.n == 0:
            state_idx = list(range(dae.n))
            algeb_idx = list(range(dae.n, dae.m + nflows))
            idx = state_idx + algeb_idx
        else:
            idx = system.Recorder.varout_idx

        # variable names concatenated
        uname = varname.unamex + varname.unamey
        fname = varname.fnamex + varname.fnamey

        for e, i in enumerate(idx):
            out += template.format(e + 1, uname[i], fname[i])

        try:
            with open(self.system.files.lst, 'w') as f:
                f.write(out)
            ret = True
        except IOError:
            logger.error('I/O Error while writing the lst file.')

        return ret
