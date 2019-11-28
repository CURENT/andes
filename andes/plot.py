#!/usr/bin/env python3

# ANDES, a power system simulation tool for research.
#
# Copyright 2015-2019 Hantao Cui
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Andes plotting tool
"""

import logging
import os
import re
import sys
from argparse import ArgumentParser
from distutils.spawn import find_executable

import numpy as np

logger = logging.getLogger(__name__)

try:
    from matplotlib import rc
    from matplotlib import pyplot as plt
except ImportError:
    logger.critical('Package <matplotlib> not found')
    sys.exit(1)

lfile = []
dfile = []


class TDSData(object):
    """
    A time-domain simulation data container for loading, extracing and
    plotting data
    """

    def __init__(self, file_name_full, path=None):
        # paths and file names
        self._path = path if path else os.getcwd()

        self.file_name, _ = os.path.splitext(file_name_full)

        self._dat_file = os.path.join(self._path, self.file_name + '.dat')
        self._lst_file = os.path.join(self._path, self.file_name + '.lst')

        # data members for raw data
        self._idx = []  # indices of variables
        self._uname = []  # unformatted variable names
        self._fname = []  # formatted variable names
        self._data = []  # data loaded from dat file

        # auxillary data members for fast query
        self.t = []
        self.nvars = 0  # total number of variables including `t`

        # TODO: consider moving the loading calls outside __init__
        self.load_lst()
        self.load_dat()

    def load_lst(self):
        """
        Load the lst file into internal data structures `_idx`, `_fname`, `_uname`, and counts the number of
        variables to `nvars`

        Returns
        -------
        None

        """
        with open(self._lst_file, 'r') as fd:
            lines = fd.readlines()

        idx, uname, fname = list(), list(), list()

        for line in lines:
            values = line.split(',')
            values = [x.strip() for x in values]

            # preserve the idx ordering here in case variables are not
            # ordered by idx
            idx.append(int(values[0]))  # convert to integer
            uname.append(values[1])
            fname.append(values[2])

        self._idx = idx
        self._fname = fname
        self._uname = uname
        self.nvars = len(uname)

    def find_var_idx(self, query, exclude=None, formatted=False):
        """
        Return variable names and indices matching `query`

        Parameters
        ----------
        query : str
            The string for querying variables
        exclude  : str, optional
            A string pattern to be excluded
        formatted : bool, optional
            True to return formatted names, False otherwise

        Returns
        -------
        (list, list)
            (List of found indices, list of found names)
        """

        # load the variable list to search in
        names = self._uname if formatted is False else self._fname

        found_idx, found_names = list(), list()

        for idx, name in zip(self._idx, names):
            if re.search(query, name):
                if exclude and re.search(exclude, name):
                    continue

                found_idx.append(idx)
                found_names.append(name)

        return found_idx, found_names

    def load_dat(self, delimiter=','):
        """
        Load the dat file into internal data structures `self._data`

        Parameters
        ----------
        delimiter : str, optional
            The delimiter for the case file. Default to comma.

        Returns
        -------
        None
        """
        try:
            data = np.loadtxt(self._dat_file, delimiter=',')
        except ValueError:
            data = np.loadtxt(self._dat_file)

        self._data = data

    def get_values(self, idx):
        """
        Return the variable values at the given indices

        Parameters
        ----------
        idx : list
            The indicex of the variables to retrieve. `idx=0` is for Time. Variable indices start at 1.

        Returns
        -------
        np.ndarray
            Variable data
        """
        return self._data[:, idx]

    def get_header(self, idx, formatted=False):
        """
        Return a list of the variable names at the given indices

        Parameters
        ----------
        idx : list or int
            The indices of the variables to retrieve
        formatted : bool
            True to retrieve latex-formatted names, False for unformatted names

        Returns
        -------
        list
            A list of variable names (headers)

        """

        if isinstance(idx, int):
            idx = [idx]
        header = self._uname if not formatted else self._fname
        return [header[x] for x in idx]

    def export_csv(self, path, idx=None, header=None, formatted=False,
                   sort_idx=True, fmt='%.18e'):
        """
        Export to a csv file

        Parameters
        ----------
        path : str
            path of the csv file to save
        idx : None or array-like, optional
            the indices of the variables to export. Export all by default
        header : None or array-like, optional
            customized header if not `None`. Use the names from the lst file
            by default
        formatted : bool, optional
            Use LaTeX-formatted header. Does not apply when using customized
            header
        sort_idx : bool, optional
            Sort by idx or not, # TODO: implement sort
        fmt : str
            cell formatter
        """
        if not idx:
            idx = self._idx
        if not header:
            header = self.get_header(idx, formatted=formatted)

        if len(idx) != len(header):
            raise ValueError("Idx length does not match header length")

        body = self.get_values(idx)

        with open(path, 'w') as fd:
            fd.write(','.join(header) + '\n')
            np.savetxt(fd, body, fmt=fmt, delimiter=',')

    def plot(self, yidx, xidx=(0,), y_calc=None, left=None, right=None, ymin=None, ymax=None, ytimes=None,
             xlabel=None, ylabel=None, legend=True, grid=False,
             latex=True, dpi=200, save=None, show=True, **kwargs):
        """
        Entery function for plot scripting. This function retrieves the x and y values based
        on the `xidx` and `yidx` inputs and then calls `plot_data()` to do the actual plotting.

        Note that `ytimes` and `y_calc` are applied sequentially if apply.

        Refer to `plot_data()` for the definition of arguments.

        Parameters
        ----------
        xidx : list or int
            The index for the x-axis variable

        yidx : list or int
            The indices for the y-axis variables

        Returns
        -------
        (fig, ax)
            Figure and axis handles
        """
        x_value = self.get_values(xidx)
        y_value = self.get_values(yidx)

        x_header = self.get_header(xidx, formatted=latex)
        y_header = self.get_header(yidx, formatted=latex)

        ytimes = float(ytimes) if ytimes is not None else ytimes

        if ytimes and (ytimes != 1):
            y_scale_func = scale_func(ytimes)
        else:
            y_scale_func = None

        # apply `ytimes` first
        if y_scale_func:
            y_value = y_scale_func(y_value)

        # `y_calc` is a callback function for manipulating data
        if y_calc is not None:
            y_value = y_calc(y_value)

        return self.plot_data(xdata=x_value, ydata=y_value, xheader=x_header, yheader=y_header,
                              left=left, right=right, ymin=ymin, ymax=ymax,
                              xlabel=xlabel, ylabel=ylabel, legend=legend, grid=grid,
                              latex=latex, dpi=dpi, save=save, show=show, **kwargs)

    def data_to_df(self):
        """Convert to pandas.DataFrame"""
        pass

    def guess_event_time(self):
        """Guess the event starting time from the input data by checking
        when the values start to change
        """
        pass

    def plot_data(self, xdata, ydata, xheader=None, yheader=None, xlabel=None, ylabel=None,
                  left=None, right=None, ymin=None, ymax=None, legend=True, grid=False, fig=None, ax=None,
                  latex=True, dpi=150, greyscale=False, save=None, show=True, **kwargs):
        """
        Plot lines for the supplied data and options. This functions takes `xdata` and `ydata` values. If
        you provide variable indices instead of values, use `plot()`.

        Parameters
        ----------
        xdata : array-like
            An array-like object containing the values for the x-axis variable

        ydata : array
            An array containing the values of each variables for the y-axis variable. The row
            of `ydata` must match the row of `xdata`. Each column correspondings to a variable.

        xheader : list
            A list containing the variable names for the x-axis variable

        yheader : list
            A list containing the variable names for the y-axis variable

        xlabel : str
            A label for the x axis

        ylabel : str
            A label for the y axis

        left : float
            The starting value of the x axis

        right : float
            The ending value of the x axis

        ymin : float
            The minimum value of the y axis
        ymax : float
            The maximum value of the y axis

        legend : bool
            True to show legend and False otherwise
        grid : bool
            True to show grid and False otherwise
        fig
            Matplotlib fig object to draw the axis on
        ax
            Matplotlib axis object to draw the lines on
        latex : bool
            True to enable latex and False to disable

        dpi : int
            Dots per inch for screen print or save
        greyscale : bool
            True to use greyscale, False otherwise
        save : bool
            True to save to png file
        show : bool
            True to show the image

        kwargs
            Optional kwargs

        Returns
        -------
        (fig, ax)
            The figure and axis handles
        """

        if not isinstance(ydata, np.ndarray):
            TypeError("ydata must be numpy array. Retrieve with get_values().")

        if ydata.ndim == 1:
            ydata = ydata.reshape((-1, 1))

        n_lines = ydata.shape[1]

        rc('font', family='Arial', size=12)

        using_latex = set_latex(latex)

        # set default x min based on simulation time
        if not left:
            left = xdata[0] - 1e-6
        if not right:
            right = xdata[-1] + 1e-6

        linestyles = ['-', '--', '-.', ':'] * len(ydata)

        if not (fig and ax):
            fig = plt.figure(dpi=dpi)
            ax = plt.gca()

        for i in range(n_lines):
            ax.plot(xdata, ydata[:, i],
                    ls=linestyles[i],
                    label=(yheader[i] if yheader else None),
                    linewidth=1,
                    )

        # for line, label in zip(line_objects, yheader):
        #     plt.legend(line, label)

        if xlabel:
            if using_latex:
                ax.set_xlabel(label_texify(xlabel))
        else:
            ax.set_xlabel(xheader[0])

        if ylabel:
            if using_latex:
                ax.set_ylabel(label_texify(ylabel))
            else:
                ax.set_ylabel(ylabel)

        ax.ticklabel_format(useOffset=False)

        ax.set_xlim(left=left, right=right)
        ax.set_ylim(ymin=ymin, ymax=ymax)

        if grid:
            ax.grid(b=True, linestyle='--')

        if legend:
            if yheader:
                ax.legend()

        plt.draw()

        if save:
            count = 1

            while True:
                outfile = self.file_name + '_' + str(count) + '.png'
                if not os.path.isfile(outfile):
                    break
                count += 1

            try:
                fig.savefig(outfile, dpi=dpi)
                logger.info('Figure saved to file {}'.format(outfile))
            except IOError:
                logger.error('* Error occurred. Try disabling LaTex with "-d".')
                return

        if show:
            plt.show()

        return fig, ax


def tdsplot_parse():
    """
    command line input parser for tdsplot

    Returns
    -------
    dict
        A dict of the command line arguments
    """
    parser = ArgumentParser(prog='tdsplot')
    parser.add_argument('datfile', nargs=1, default=[], help='dat file name.')
    parser.add_argument('x', nargs=1, type=int, help='x axis variable index')
    parser.add_argument('y', nargs='*', help='y axis variable index')
    parser.add_argument('--xmin', type=float, help='x axis minimum value', dest='left')
    parser.add_argument('--xmax', type=float, help='x axis maximum value', dest='right')
    parser.add_argument('--ymax', type=float, help='y axis maximum value')
    parser.add_argument('--ymin', type=float, help='y axis minimum value')

    parser.add_argument('--checkinit', action='store_true', help='check initialization value')

    parser.add_argument('-x', '--xlabel', type=str, help='manual x-axis text label')
    parser.add_argument('-y', '--ylabel', type=str, help='y-axis text label')

    parser.add_argument('-s', '--save', action='store_true', help='save to file')
    parser.add_argument('-g', '--grid', action='store_true', help='grid on')
    parser.add_argument('-d', '--no_latex', action='store_false', dest='latex', help='disable LaTex formatting')

    parser.add_argument('-n', '--no_show', action='store_false', dest='show', help='do not show the plot window')

    parser.add_argument('--ytimes', type=str, help='y switch_times')
    parser.add_argument('--dpi', type=int, help='image resolution in dot per inch (DPI)')

    args = parser.parse_args()
    return vars(args)


def parse_y(y, upper, lower=0):
    """
    Parse command-line input for Y indices and return a list of indices

    Parameters
    ----------
    y : Union[List, Set, Tuple]
        Input for Y indices. Could be single item (with or without colon), or
         multiple items

    upper : int
        Upper limit. In the return list y, y[i] <= uppwer.

    lower : int
        Lower limit. In the return list y, y[i] >= lower.

    Returns
    -------

    """
    if len(y) == 1:
        if y[0].count(':') >= 3:
            logger.error('Index format not acceptable. Must not contain more than three colons.')
            return []

        elif y[0].count(':') == 0:
            if isint(y[0]):
                y[0] = int(y[0])
                return y
        elif y[0].count(':') == 1:
            if y[0].endswith(':'):
                y[0] += str(upper)
            if y[0].startswith(':'):
                y[0] = str(lower) + y[0]

        elif y[0].count(':') == 2:
            if y[0].endswith(':'):
                y[0] += str(1)
            if y[0].startswith(':'):
                y[0] = str(lower) + y[0]

            if y[0].count('::') == 1:
                y[0] = y[0].replace('::', ':{}:'.format(upper))
                print(y)

        y = y[0].split(':')

        for idx, item in enumerate(y[:]):
            try:
                y[idx] = int(item)
            except ValueError:
                logger.warning('y contains non-numerical values <{}>. Parsing could not proceed.'.format(item))
                return []

        y_from_range = list(range(*y))

        y_in_range = []

        for item in y_from_range:
            if lower <= item < upper:
                y_in_range.append(item)

        return y_in_range

    else:
        y_to_int = []
        for idx, val in enumerate(y):
            try:
                y_to_int.append(int(val))
            except ValueError:
                logger.warning('Y indices contains non-numerical values. Skipped <{}>.'.format(val))

        y_in_range = [item for item in y_to_int if lower <= item < upper]
        return list(y_in_range)


def add_plot(x, y, xl, yl, fig, ax, LATEX=False, linestyle=None, **kwargs):
    """Add plots to an existing plot"""
    if LATEX:
        # xl_data = xl[1]  # NOQA
        yl_data = yl[1]
    else:
        # xl_data = xl[0]  # NOQA
        yl_data = yl[0]

    for idx, y_val in enumerate(y):
        ax.plot(x, y_val, label=yl_data[idx], linestyle=linestyle)

    ax.legend(loc='upper right')
    ax.set_ylim(auto=True)


def eig_plot(name, args):
    fullpath = os.path.join(name, '.txt')
    raw_data = []
    started = 0
    fid = open(fullpath)
    for line in fid.readline():
        if '#1' in line:
            started = 1
        elif 'PARTICIPATION FACTORS' in line:
            started = -1

        if started == 1:
            raw_data.append(line)
        elif started == -1:
            break
    fid.close()

    for line in raw_data:
        # data = line.split()
        # TODO: complete this function
        pass


def tdsplot(datfile, y, x=(0,), **kwargs):
    """
    TDS plot main function based on the new TDSData class

    Parameters
    ----------
    datfile : str
        Path to the andes tds output data file (.dat or .lst file)
    x : list or int, optional
        The index for the x-axis variable. x=0 by default for time
    y : list or int
        The indices for the y-axis variable

    Returns
    -------
    TDSData object
    """

    # single data file
    if len(datfile) == 1:
        tds_data = TDSData(datfile[0])
        y_num = parse_y(y, lower=0, upper=tds_data.nvars)
        tds_data.plot(xidx=x, yidx=y_num, **kwargs)
        return tds_data
    else:
        raise NotImplementedError("Plotting multiple data files are not supported yet")


def tdsplot_main():
    """
    Entry function for tds plot. Parses command line arguments and calls `tdsplog`

    Returns
    -------
    None

    """
    args = tdsplot_parse()
    tdsplot(**args)


def check_init(yval, yl):
    """"Check initialization by comparing t=0 and t=end values"""
    suspect = []
    for var, label in zip(yval, yl):
        if abs(var[0] - var[-1]) >= 1e-6:
            suspect.append(label)
    if suspect:
        logger.error('Initialization failure:')
        logger.error(', '.join(suspect))
    else:
        logger.error('Initialization is correct.')


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def scale_func(k):
    """
    Return a lambda function that scales its input by k

    Parameters
    ----------
    k : float
        The scaling factor of the returned lambda function
    Returns
    -------
    Lambda function

    """
    return lambda y_values_input: k * y_values_input


def label_texify(label):
    """
    Convert a label to latex format by appending surrounding $ and escaping spaces

    Parameters
    ----------
    label : str
        The label string to be converted to latex expression

    Returns
    -------
    str
        A string with $ surrounding
    """
    return '$' + label.replace(' ', r'\ ') + '$'


def set_latex(with_latex=True):
    """
    Enables latex for matplotlib based on the `with_latex` option and `dvipng` availability

    Parameters
    ----------
    with_latex : bool, optional
        True for latex on and False for off

    Returns
    -------
    bool
        True for latex on and False for off
    """

    has_dvipng = find_executable('dvipng')

    if has_dvipng and with_latex:
        rc('text', usetex=True)
        return True
    else:
        rc('text', usetex=False)
        return False


def main(cli=True, **args):
    logger.warning('andesplot is deprecated and will be remove in future versions. '
                   'Use "tdsplot" for TDS data or "eigplot" for EIG data.')
    tdsplot_main()


if __name__ == "__main__":
    main(cli=True)
