import numpy as np
import os
import time
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
        self.epoch_t = []
        self.t = []
        self.k = []
        self.vars = []
        self.vars_array = None

        self._np_block_rows = 600
        self._np_block_shape = (0, 0)
        self.np_vars = np.ndarray(shape=(0, 0))
        self.np_t = np.ndarray(shape=(0,))
        self.np_epoch_t = np.ndarray(shape=(0,))
        self.np_k = np.ndarray(shape=(0,))

        self.np_nrows = 0
        self.np_ncols = 0

        self.dat = None
        self._mode = 'w'

        self._last_t = 0
        self._last_epoch_t = 0
        self._last_vars = []

    def store(self, t, step):
        """
        Record the state/algeb values at time t to self.vars
        """
        epoch_time = time.time()
        max_cache = int(self.system.tds.config.max_cache)
        if len(self.vars) >= max_cache > 0:
            self.dump()
            self.vars = list()
            self.t = list()
            self.epoch_t = list()
            self.k = list()
            logger.debug(
                'varout cache cleared at simulation t = {:g}.'.format(
                    self.system.dae.t))
            self._mode = 'a'

        var_data = matrix([self.system.dae.x, self.system.dae.y])

        # ===== This code block is deprecated =====
        self.t.append(t)
        self.epoch_t.append(epoch_time)
        self.k.append(step)
        self.vars.append(var_data)
        # =========================================

        # temporary storage
        self._last_t = t
        self._last_epoch_t = epoch_time
        self._last_vars = list(var_data)

        #

        # clear data cache if written to disk
        if self.np_nrows >= max_cache > 0:
            self.dump_np_vars()
            self.np_vars = np.zeros(self._np_block_shape)
            self.np_nrows = 0
            self.np_t = np.zeros((self._np_block_rows,))
            self.np_epoch_t = np.zeros((self._np_block_rows,))
            self.np_k = np.zeros((self._np_block_rows,))
            logger.debug(
                'np_vars cache cleared at simulation t = {:g}.'.format(
                    self.system.dae.t))
            self._mode = 'a'

        # initialize before first-time adding data
        if self.np_nrows == 0:
            self.np_ncols = len(var_data)
            self._np_block_shape = (self._np_block_rows, self.np_ncols)
            self.np_vars = np.zeros(self._np_block_shape)
            self.np_t = np.zeros((self._np_block_rows,))
            self.np_epoch_t = np.zeros((self._np_block_rows,))
            self.np_k = np.zeros((self._np_block_rows,))

        # adding data to the matrix
        # self.np_vars[self.np_nrows, 0] = t
        self.np_t[self.np_nrows] = t
        self.np_epoch_t[self.np_nrows] = epoch_time
        self.np_k[self.np_nrows] = step
        self.np_vars[self.np_nrows, :] = np.array(var_data).reshape((-1))
        self.np_nrows += 1

        # check if matrix extension is needed
        if self.np_nrows >= self.np_vars.shape[0]:
            self.np_vars = np.concatenate([self.np_vars, np.zeros(self._np_block_shape)], axis=0)
            self.np_t = np.concatenate([self.np_t, np.zeros((self._np_block_rows,))], axis=0)
            self.np_epoch_t = np.concatenate([self.np_epoch_t, np.zeros((self._np_block_rows,))], axis=0)
            self.np_k = np.concatenate([self.np_k, np.zeros((self._np_block_rows,))], axis=0)

        # remove the post-computed variables from the variable list
        if self.system.tds.config.compute_flows:
            self.system.dae.y = self.system.dae.y[:self.system.dae.m]

    def get_latest_data(self):
        """
        Get the latest data and simulation time

        Returns
        -------
        dict : a dictionary containing `t` and `vars`
        """
        return {'epoch_time': float(self._last_epoch_t), 't': float(self._last_t), 'var': self._last_vars}

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
        logger.warning('This function is deprecated and replaced by `concat_t_vars_np`.')
        out = np.array([])

        if len(self.t) == 0:
            return out

        out = np.ndarray(shape=(0, self.vars[0].size[0] + 1))

        for t, var in zip(self.t, self.vars):
            line = [[t]]
            line[0].extend(list(var))
            out = np.append(out, line, axis=0)

        return out

    def concat_t_vars_np(self, vars_idx=None):
        """
        Concatenate `self.np_t` with `self.np_vars` and return a single matrix.
        The first column corresponds to time, and the rest of the matrix is the variables.

        Returns
        -------
        np.array : concatenated matrix

        """
        selected_np_vars = self.np_vars
        if vars_idx is not None:
            selected_np_vars = self.np_vars[:, vars_idx]

        return np.concatenate([self.np_t[:self.np_nrows].reshape((-1, 1)),
                               selected_np_vars[:self.np_nrows, :]], axis=1)

    def get_xy(self, yidx, xidx=0):
        """
        Return stored data for the given indices for plot

        :param yidx: the indices of the y-axis variables(1-indexing)
        :param xidx: the index of the x-axis variables
        :return: None
        """
        if not isinstance(xidx, int):
            logger.error('Argument xidx must be an integer. get_xy() cannot continue')

        if isinstance(yidx, int):
            yidx = [yidx]

        t_vars = self.concat_t_vars()

        xdata = t_vars[:, xidx]
        ydata = t_vars[:, yidx]

        return xdata.tolist(), ydata.transpose().tolist()

    def dump_np_vars(self, store_format='csv', delimiter=','):
        """
        Dump the TDS simulation data to files by calling subroutines `write_lst` and
        `write_np_dat`.

        Parameters
        -----------

        store_format : str
            dump format in `('csv', 'txt', 'hdf5')`

        delimiter : str
            delimiter for the `csv` and `txt` format

        Returns
        -------
        bool: success flag
        """

        ret = False

        if self.system.files.no_output is True:
            logger.debug('no_output is True, thus no TDS dump saved ')
            return True

        if self.write_lst() and self.write_np_dat(store_format=store_format, delimiter=delimiter):
            ret = True

        return ret

    def dump(self):
        """
        Dump the TDS results to the output `dat` file

        :return: succeed flag
        """
        logger.warning('This function is deprecated and replaced by `dump_np_vars`.')

        ret = False

        if self.system.files.no_output:
            # return ``True`` because it did not fail
            return True

        if self.write_lst() and self.write_dat():
            ret = True

        return ret

    def write_np_dat(self, store_format='csv', delimiter=',', fmt='%.18e'):
        """
        Write TDS data stored in `self.np_vars` to the output file

        Parameters
        ----------
        store_format : str
            dump format in ('csv', 'txt', 'hdf5')

        delimiter : str
            delimiter for the `csv` and `txt` format

        fmt : str
            output formatting template

        Returns
        -------
        bool : success flag

        """
        ret = False
        system = self.system

        # compute the total number of columns, excluding time
        if not system.Recorder.n:
            n_vars = system.dae.m + system.dae.n
            # post-computed power flows include:
            #   bus   - (Pi, Qi)
            #   line  - (Pij, Pji, Qij, Qji, Iij_Real, Iij_Imag, Iji_real, Iji_Imag)
            if system.tds.config.compute_flows:
                n_vars += 2 * system.Bus.n + 8 * system.Line.n + 2 * system.Area.n_combination
            idx = list(range(n_vars))

        else:
            n_vars = len(system.Recorder.varout_idx)
            idx = system.Recorder.varout_idx

        # prepare data
        t_vars_concatenated = self.concat_t_vars_np(vars_idx=idx)

        try:
            os.makedirs(os.path.abspath(os.path.dirname(system.files.dat)), exist_ok=True)
            with open(system.files.dat, self._mode) as f:
                if store_format in ('csv', 'txt'):
                    np.savetxt(f, t_vars_concatenated, fmt=fmt, delimiter=delimiter)
                elif store_format == 'hdf5':
                    pass
                ret = True
                logger.info('TDS data dumped to <{}>'.format(system.files.dat))

        except IOError:
            logger.error('I/O Error while writing the dat file.')

        return ret

    def write_dat(self):
        """
        Write ``system.Varout.vars`` to a ``.dat`` file
        :return:
        """
        logger.warn('This function is deprecated and replaced by `write_np_dat`.')

        ret = False
        system = self.system

        # compute the total number of columns, excluding time
        if not system.Recorder.n:
            n_vars = system.dae.m + system.dae.n
            # post-computed power flows include:
            #   bus   - (Pi, Qi)
            #   line  - (Pij, Pji, Qij, Qji, Iij_Real, Iij_Imag, Iji_real, Iji_Imag)
            if system.tds.config.compute_flows:
                n_vars += 2 * system.Bus.n + 8 * system.Line.n + 2 * system.Area.n_combination
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
            os.makedirs(os.path.abspath(os.path.dirname(system.files.dat)), exist_ok=True)
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
        template = '{:>6g}, {:>25s}, {:>35s}\n'

        # header line
        out += template.format(0, 'Time [s]', '$Time\\ [s]$')

        # include line flow variables in algebraic variables
        nflows = 0
        if self.system.tds.config.compute_flows:
            nflows = 2 * self.system.Bus.n + \
                     8 * self.system.Line.n + \
                     2 * self.system.Area.n_combination

        # output variable indices
        if system.Recorder.n == 0:
            state_idx = list(range(dae.n))
            algeb_idx = list(range(dae.n, dae.n + dae.m + nflows))
            idx = state_idx + algeb_idx
        else:
            idx = system.Recorder.varout_idx

        # variable names concatenated
        uname = varname.unamex + varname.unamey
        fname = varname.fnamex + varname.fnamey

        for e, i in enumerate(idx):
            # `idx` in the lst file is always consecutive
            out += template.format(e + 1, uname[i], fname[i])

        try:
            with open(self.system.files.lst, 'w') as f:
                f.write(out)
            ret = True
        except IOError:
            logger.error('I/O Error while writing the lst file.')

        return ret

    def vars_to_array(self):
        """
        Convert `self.vars` to a numpy array

        Returns
        -------
        numpy.array
        """

        logger.warn('This function is deprecated. You can inspect `self.np_vars` directly as NumPy arrays '
                    'without conversion.')
        if not self.vars:
            return None

        vars_matrix = matrix(self.vars, size=(self.vars[0].size[0],
                                              len(self.vars))).trans()

        self.vars_array = np.array(vars_matrix)

        return self.vars_array
