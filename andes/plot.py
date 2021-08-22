"""
The Andes plotting tool.
"""

import os
import re
import logging
import math

import numpy as np

from andes.shared import mpl, plt
from andes.shared import set_latex

from andes.core.var import BaseVar

logger = logging.getLogger(__name__)
DPI = 100


class TDSData:
    """
    A data container for loading and plotting results from Andes time-domain simulation.
    """

    def __init__(self, full_name=None, mode='file', dae=None, path=None):
        # paths and file names
        self._mode = mode
        self.full_name = full_name
        self.dae = dae
        self._path = path if path else os.getcwd()
        self.file_name = None
        self.file_ext = None
        self._npy_file = None
        self._lst_file = None

        # data members for raw data
        self._idx = []    # indices of variables
        self._uname = []  # unformatted variable names
        self._fname = []  # formatted variable names
        self._data = []   # data loaded from file

        # auxillary data members for fast query
        self.t = []
        self.nvars = 0  # total number of variables including `t`

        if self._mode == 'file':
            self._process_names()
            self.load_lst()
            self.load_npy_or_csv()
        elif self._mode == 'memory':
            self.load_dae()
            self._process_names()
        else:
            raise NotImplementedError(f'Unknown mode {self._mode}.')

        self._process_names()

    def _process_names(self):
        if self.full_name is None:
            logger.info("Input file name not detected. Using `Untitled`.")
            self.full_name = 'Untitled'
            self.file_name = 'Untitled'
            self.file_ext = ''
        else:
            self.file_name, self.file_ext = os.path.splitext(self.full_name)

        self._npy_file = os.path.join(self._path, self.file_name + '.npy')

        npz_path = os.path.join(self._path, self.file_name + '.npz')
        if os.path.isfile(npz_path):
            self._npy_file = npz_path

        self._lst_file = os.path.join(self._path, self.file_name + '.lst')
        self._csv_file = os.path.join(self._path, self.file_name + '.csv')

    def load_dae(self):
        """Load from DAE time series"""
        dae = self.dae
        self.t = dae.ts.t
        self.nvars = dae.n + dae.m + dae.o + 1

        self._idx = list(range(self.nvars))
        self._uname = ['Time [s]'] + dae.x_name + dae.y_name + dae.z_name
        self._fname = ['Time [s]'] + dae.x_tex_name + dae.y_tex_name + dae.z_tex_name
        self._data = dae.ts.txyz

        if dae.system.files.lst is not None:
            self.full_name = dae.system.files.lst
        else:
            self.full_name = dae.system.files.case

    def load_lst(self):
        """
        Load the lst file into internal data structures `_idx`, `_fname`, `_uname`, and counts the number of
        variables to `nvars`.

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

    def find(self, query, exclude=None, formatted=False, idx_only=False):
        """
        Return variable names and indices matching `query`.

        Parameters
        ----------
        query : str
            The string for querying variables. Multiple conditions can be separated by comma without space.
        exclude  : str, optional
            A string pattern to be excluded
        formatted : bool, optional
            True to return formatted names, False otherwise
        idx_only : bool, optional
            True if only return indices

        Returns
        -------
        (list, list)
            (List of found indices, list of found names)
        """

        # load the variable list to search in
        names = self._uname if formatted is False else self._fname

        found_idx, found_names = list(), list()

        query_list = query.split(',')
        for idx, name in zip(self._idx, names):
            for q in query_list:
                if re.search(q, name):
                    if exclude and re.search(exclude, name):
                        continue

                    found_idx.append(idx)
                    found_names.append(name)

        if idx_only:
            return found_idx
        else:
            return found_idx, found_names

    def load_npy_or_csv(self, delimiter=','):
        """
        Load the npy, zpy or (the legacy) csv file into the
        internal data structure `self._xy`.

        Parameters
        ----------
        delimiter : str, optional
            The delimiter for the case file. Default to comma.

        Returns
        -------
        None
        """
        try:
            data = np.load(self._npy_file)

            if self._npy_file.endswith('npz'):
                data = data['data']

        except FileNotFoundError:
            data = np.loadtxt(self._csv_file, delimiter=delimiter, skiprows=1)

        self._data = data

    def get_values(self, idx):
        """
        Return the variable values at the given indices.

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
        Return a list of the variable names at the given indices.

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

        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        header = self._uname if not formatted else self._fname
        return [header[x] for x in idx]

    def export_csv(self, path=None, idx=None, header=None, formatted=False,
                   sort_idx=True, fmt='%.18e'):
        """
        Export to a csv file.

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
        if not path:
            path = self._csv_file
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

        logger.info(f'CSV data saved to "{path}".')

    def plot(self, yidx, xidx=(0,), *, a=None, ytimes=None, ycalc=None,
             left=None, right=None, ymin=None, ymax=None,
             xlabel=None, ylabel=None, xheader=None, yheader=None,
             legend=None, grid=False, greyscale=False, latex=True,
             dpi=DPI, line_width=1.0, font_size=12, savefig=None, save_format=None, show=True,
             title=None, linestyles=None, use_bqplot=False,
             hline1=None, hline2=None, vline1=None, vline2=None,
             fig=None, ax=None, backend=None,
             set_xlim=True, set_ylim=True, autoscale=False,
             legend_bbox=None, legend_loc=None, legend_ncol=1,
             figsize=None,
             **kwargs):
        """
        Entry function for plotting.

        This function retrieves the x and y values based on the `xidx` and
        `yidx` inputs, applies scaling functions `ytimes` and `ycalc` sequentially,
        and delegates the plotting to the backend.

        Parameters
        ----------
        yidx : list or int
            The indices for the y-axis variables
        xidx : tuple or int, optional
            The index for the x-axis variable
        a : tuple or list, optional
            The 0-indexed sub-indices into `yidx` to plot.
        ytimes : float, optional
            A scaling factor to apply to all y values.
        left : float
            The starting value of the x axis
        right : float
            The ending value of the x axis
        ymin : float
            The minimum value of the y axis
        ymax : float
            The maximum value of the y axis
        ylabel : str
            Text label for the y axis
        yheader : list
            A list containing the variable names for the y-axis variable
        title : str
            Title string to be shown at the top
        fig
            Existing figure object to draw the axis on.
        ax
            Existing axis object to draw the lines on.

        Other Parameters
        ----------------
        ycalc: callable, optional
            A callable to apply to all y values after scaling with `ytimes`.
        xlabel : str
            Text label for the x axis
        xheader : list
            A list containing the variable names for the x-axis variable
        legend : bool
            True to show legend and False otherwise
        legend_ncol : int
            Number of columns in legend
        legend_bbox : tuple of two floats
            legend box to anchor
        grid : bool
            True to show grid and False otherwise
        latex : bool
            True to enable latex and False to disable
        greyscale : bool
            True to use greyscale, False otherwise
        savefig : bool or str
            True to save to png figure file.
            str is treated as the output file name.
        save_format : str
            File extension string (pdf, png or jpg) for the savefig format
        dpi : int
            Dots per inch for screen print or save.
            `savefig` uses a minimum of 200 dpi
        line_width : float
            Plot line width
        font_size : float
            Text font size (labels and legends)
        figsize : tuple
            Figure size passed when creating new figure
        show : bool
            True to show the image
        backend : str or None
            `bqplot` to use the bqplot backend in notebook.
            None for matplotlib.
        hline1: float, optional
            Dashed horizontal line 1
        hline2: float, optional
            Dashed horizontal line 2
        vline1: float, optional
            Dashed horizontal line 1
        vline2: float, optional
            Dashed vertical line 2

        Returns
        -------
        (fig, ax)
            Figure and axis handles for matplotlib backend.
        fig
            Figure object for bqplot backend.

        """
        if self._mode == 'memory':
            if isinstance(yidx, BaseVar):
                if yidx.n == 0:
                    logger.error(f"Variable <{yidx.name}> contains no values.")
                    return
                offs = 1
                if yidx.v_code == 'y':
                    offs += self.dae.n

                yidx = yidx.a + offs

        if a is not None:
            yidx = np.take(yidx, a)

        xvalue = self.get_values(xidx)
        yvalue = self.get_values(yidx)

        # header: names for variables
        # axis labels: the texts next to axes
        if not xheader:
            xheader = self.get_header(xidx, formatted=latex)
        if not yheader:
            yheader = self.get_header(yidx, formatted=latex)

        # process `ytimes`
        if ytimes is not None:
            ytimes = float(ytimes)
            if ytimes != 1.0:
                yvalue = scale_func(ytimes)(yvalue)

        # call `ycalc` on `yvalue`
        if ycalc is not None:
            yvalue = ycalc(yvalue)

        plot_call = self.get_call(backend)

        return plot_call(xdata=xvalue, ydata=yvalue,
                         left=left, right=right, ymin=ymin, ymax=ymax,
                         xheader=xheader, yheader=yheader, xlabel=xlabel, ylabel=ylabel,
                         legend=legend, grid=grid, greyscale=greyscale, latex=latex,
                         dpi=dpi, line_width=line_width, font_size=font_size,
                         savefig=savefig, save_format=save_format, show=show, title=title,
                         hline1=hline1, hline2=hline2, vline1=vline1, vline2=vline2,
                         fig=fig, ax=ax, linestyles=linestyles,
                         set_xlim=set_xlim, set_ylim=set_ylim, autoscale=autoscale,
                         legend_bbox=legend_bbox, legend_loc=legend_loc, legend_ncol=legend_ncol,
                         figsize=figsize,
                         **kwargs)

    def get_call(self, backend=None):
        """
        Get the internal `plot_data` function for the specified backend.
        """
        if backend == 'bqplot':
            return self.bqplot_data

        return self.plot_data

    def data_to_df(self):
        """Convert to pandas.DataFrame"""
        pass

    def guess_event_time(self):
        """
        Guess the event starting time from the input data by checking
        when the values start to change
        """
        pass

    def bqplot_data(self, xdata, ydata, *, xheader=None, yheader=None, xlabel=None, ylabel=None,
                    left=None, right=None, ymin=None, ymax=None, legend=True, grid=False, fig=None,
                    dpi=DPI, line_width=1.0, greyscale=False, savefig=None, save_format=None,
                    title=None,
                    **kwargs):
        """
        Plot with ``bqplot``. Experimental and incomplete.
        """

        from bqplot import pyplot as plt
        if not isinstance(ydata, np.ndarray):
            raise TypeError("ydata must be numpy array. Retrieve with `get_values()`.")

        if ydata.ndim == 1:
            ydata = ydata.reshape((-1, 1))

        if fig is None:
            fig = plt.figure(dpi=dpi)
        plt.plot(xdata, ydata.transpose(),
                 linewidth=line_width,
                 figure=fig,
                 )

        if yheader:
            plt.label(yheader)
        if title:
            plt.title(title)
        plt.show()

        return fig

    def plot_data(self, xdata, ydata, *, xheader=None, yheader=None, xlabel=None, ylabel=None, linestyles=None,
                  left=None, right=None, ymin=None, ymax=None, legend=None, grid=False, fig=None, ax=None,
                  latex=True, dpi=DPI, line_width=1.0, font_size=12, greyscale=False, savefig=None,
                  save_format=None, show=True, title=None, hline1=None, hline2=None, vline1=None,
                  vline2=None, set_xlim=True, set_ylim=True, autoscale=False, figsize=None,
                  legend_bbox=None, legend_loc=None, legend_ncol=1,
                  mask=True,
                  **kwargs):
        """
        Plot lines for the supplied data and options.

        This functions takes `xdata` and `ydata` values.
        If you provide variable indices instead of values, use `plot()`.

        See the argument lists of `plot()` for more.

        Parameters
        ----------
        xdata : array-like
            An array-like object containing the values for the x-axis variable
        ydata : array
            An array containing the values of each variables for the y-axis variable. The row
            of `ydata` must match the row of `xdata`. Each column correspondings to a variable.
        mask : bool
            If enabled (1), when specifying axis limits, only data in the limits will be
            used for plotting to optimize for autoscaling.
            It is done through an index mask.

        Returns
        -------
        (fig, ax)
            The figure and axis handles

        Examples
        --------
        To plot the results of arithmetic calculation of variables, retrieve the values,
        do the calculation, and plot with `plot_data`.

        >>> v = ss.dae.ts.y[:, ss.PVD1.v.a]
        >>> Ipcmd = ss.dae.ts.y[:, ss.PVD1.Ipcmd_y.a]
        >>> t = ss.dae.ts.t

        >>> ss.TDS.plt.plot_data(t, v * Ipcmd,
        >>>                      xlabel='Time [s]',
        >>>                      ylabel='Ipcmd [pu]')

        """
        mpl.rc('font', family='serif', size=font_size)

        if not isinstance(ydata, np.ndarray):
            raise TypeError("ydata must be a numpy array. Retrieve with get_values().")

        if ydata.ndim == 1:
            ydata = ydata.reshape((-1, 1))

        n_lines = ydata.shape[1]

        if latex:
            set_latex()

        # set default x min based on simulation time
        if not left:
            left = xdata[0] - 1e-6
        if not right:
            right = xdata[-1] + 1e-6

        if not linestyles:
            linestyles = ['-', '--', '-.', ':']

        linestyles = linestyles * int(n_lines / len(linestyles) + 1)

        if fig is None or ax is None:
            fig = plt.figure(dpi=dpi, figsize=figsize)
            ax = plt.gca()

        if greyscale:
            plt.gray()

        if mask is True:
            mask = (xdata >= (left - 0.1)) & (xdata <= (right + 0.1))
            xdata = xdata[mask]
            ydata = ydata[mask.reshape(-1, )]

        for i in range(n_lines):
            ax.plot(xdata,
                    ydata[:, i],
                    ls=linestyles[i],
                    label=yheader[i] if yheader else None,
                    linewidth=line_width,
                    color='0.2' if greyscale else None,
                    )

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif xheader is not None and len(xheader) > 0:
            ax.set_xlabel(xheader[0])

        if ylabel:
            ax.set_ylabel(ylabel)

        ax.ticklabel_format(useOffset=False)

        if set_xlim is True:
            ax.set_xlim(left=left, right=right)
        if set_ylim is True:
            ax.set_ylim(bottom=ymin, top=ymax)
        if autoscale is True:
            ax.autoscale(axis='y')

        if grid:
            ax.grid(b=True, linestyle='--')

        if yheader is None:
            legend = False
        elif legend is None:
            if len(yheader) <= 8:
                legend = True

        if legend:
            ax.legend(bbox_to_anchor=legend_bbox,
                      loc=legend_loc,
                      ncol=legend_ncol)

        if title:
            ax.set_title(title)

        if hline1:
            ax.axhline(y=hline1, linewidth=1, ls=':', color='grey')
        if hline2:
            ax.axhline(y=hline2, linewidth=1, ls=':', color='grey')
        if vline1:
            ax.axvline(x=vline1, linewidth=1, ls=':', color='grey')
        if vline2:
            ax.axvline(x=vline2, linewidth=1, ls=':', color='grey')

        plt.draw()

        if savefig is not None:
            if save_format is None:
                save_format = 'png'

            if dpi is None:
                dpi = 200
            else:
                dpi = max(dpi, 200)

            # use supplied file name
            if isinstance(savefig, str):
                outfile = savefig + '.' + save_format
            # or generate a new name
            else:
                count = 1
                while True:
                    outfile = f'{self.file_name}_{count}.{save_format}'
                    if not os.path.isfile(outfile):
                        break
                    count += 1

            fig.savefig(outfile, dpi=dpi)
            logger.info('Figure saved to "%s".', outfile)

        if show:
            plt.show()

        return fig, ax

    def plotn(self, nrows: int, ncols: int, yidxes, xidxes=None, *, dpi=DPI, titles=None,
              a=None, figsize=None, xlabel=None, ylabel=None, sharex=None, sharey=None, show=True,
              xlabel_offs=(0.5, 0.01), ylabel_offs=(0.05, 0.5), hspace=0.2, wspace=0.2,
              **kwargs):
        """
        Plot multiple subfigures in one figure.

        Parameters ``xidxes``, ``a``, ``xlabels`` and ``ylabels``, if provided,
        must have the same length as ``yidxes``.

        Parameters
        ----------
        nrows : int
            number of rows
        ncols : int
            number of cols
        yidx
            A list of `BaseVar` or index lists.
        """

        nyidxes = len(yidxes)
        if nyidxes > nrows * ncols:
            raise ValueError("yidxes with length %d does not fit nrows=%d and ncols=%d",
                             nyidxes, nrows, ncols)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        xidx = (0,)
        aidx = None
        title = ''

        sharex = True if (sharex is None and ncols == 1) else False
        sharey = True if (sharey is None and nrows == 1) else False

        axes = fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=False)

        ii = 0
        for jj in range(nrows):
            for kk in range(ncols):
                if ii >= nyidxes:
                    break

                yidx = yidxes[ii]
                ax = axes[jj, kk]
                if xidxes is not None:
                    xidx = xidxes[ii]

                if a is not None:
                    aidx = a[ii]

                if titles is not None:
                    title = titles[ii]

                fig, ax = self.plot(yidx, xidx, a=aidx, fig=fig, ax=ax,
                                    xlabel='', ylabel='', title=title,
                                    show=False, **kwargs)
                ii += 1
        if xlabel:
            fig.text(*xlabel_offs, xlabel, ha='center', va='center')
        if ylabel:
            fig.text(*ylabel_offs, ylabel, ha='center', va='center', rotation='vertical')

        fig.subplots_adjust(hspace=hspace, wspace=wspace,)

        if show:
            plt.show()

        return fig, axes

    def panoview(self, mdl, *, ncols=3, vars=None, idx=None, a=None, figsize=None, **kwargs):
        """
        Panoramic view of variables of a given model instance.

        Select variables through ``vars``. Select devices through ``idx`` or ``a``,
        which has a higher priority.

        This function also takes other arguments recognizable by ``self.plot``.

        Parameters
        ----------
        mdl : ModelBase
            Model instance
        ncol : int
            Number of columns
        var : list of str
            A list of variable names to display
        idx : list
            A list of device idx-es for showing
        a : list of int
            A list of device 0-based positions for showing
        figsize : tuple
            Figure size for plotting

        Examples
        --------
        To plot ``omega`` and ``delta`` of GENROUs ``GENROU_1`` and ``GENROU_2``:

        .. code-block :: python

            system.TDS.plt.plot(system.GENROU,
                                vars=['omega', 'delta'],
                                idx=['GENROU_1', 'GENROU_2'])

        """
        # `a` takes precedece over `idx`
        if a is None:
            a = mdl.idx2uid(idx)

        # compute the number of rows and cols
        states = list()
        algebs = list()

        if vars is None:
            states = mdl.states.values()
            algebs = mdl.algebs.values()
        else:
            for item in vars:
                if item in mdl.states:
                    states.append(mdl.states[item])
                elif item in mdl.algebs:
                    algebs.append(mdl.algebs[item])
                else:
                    logger.warning("Variable <%s> does not exist in model <%s>",
                                   item, mdl.class_name)
        nstates = len(states)
        nalgebs = len(algebs)

        nrows_states = math.ceil(nstates / ncols)
        nrows_algebs = math.ceil(nalgebs / ncols)

        # build canvas
        if figsize is None:
            figsize = (3 * ncols, 2 * (nrows_states + nrows_algebs))

        fig, axes = plt.subplots(nrows_states + nrows_algebs, ncols,
                                 figsize=figsize, dpi=DPI, squeeze=False,
                                 )
        fig.tight_layout()

        # turn off unused axes
        if nstates % ncols != 0:
            for i in range(nstates % ncols, ncols):
                axes[nrows_states-1, i].axis('off')

        if nalgebs % ncols != 0:
            for i in range(nalgebs % ncols, ncols):
                axes[-1, i].axis('off')

        # plot states
        for ii, item in enumerate(states):
            row_no = math.floor(ii / ncols)
            col_no = ii % ncols
            self.plot(item, a=a,
                      title=f'${item.tex_name}$',
                      xlabel='',
                      fig=fig, ax=axes[row_no, col_no], show=False, **kwargs)

        # plot algebs
        for ii, item in enumerate(algebs):
            row_no = math.floor(ii / ncols) + nrows_states
            col_no = ii % ncols
            self.plot(item, a=a,
                      title=f'${item.tex_name}$',
                      xlabel='',
                      fig=fig, ax=axes[row_no, col_no], show=False, **kwargs)

        return fig, axes


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
                logger.warning(f'y contains non-numerical values <{item}>. Parsing cannot proceed.')
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


def tdsplot(filename, y, x=(0,),
            to_csv=False,
            find=None,
            xargs=None,
            exclude=None,
            **kwargs):
    """
    TDS plot main function based on the new TDSData class.

    Parameters
    ----------
    filename : str
        Path to the ANDES TDS output data file. Works without extension.
    x : list or int, optional
        The index for the x-axis variable. x=0 by default for time
    y : list or int
        The indices for the y-axis variable
    to_csv : bool
        True if need to export to a csv file
    find : str, optional
        if not none, specify the variable name to find
    xargs : str, optional
        similar to find, but return the result indices with file name, x idx name for xargs
    exclude : str, optional
        variable name pattern to exclude

    Returns
    -------
    TDSData object
    """

    # single data file
    if len(filename) == 1:
        tds_data = TDSData(filename[0])
        if to_csv is True:
            tds_data.export_csv()
            return
        if find is not None:
            out = tds_data.find(query=find, exclude=exclude)
            print(out)
            return
        if xargs is not None:
            out = tds_data.find(query=xargs, exclude=exclude, idx_only=True)
            out = [str(i) for i in out]
            print(filename[0] + ' 0 ' + ' '.join(out))
            return
        if len(y) == 0:
            logger.error('Must specify Y indices to plot.')
            return
        y_num = parse_y(y, lower=0, upper=tds_data.nvars)
        tds_data.plot(xidx=x, yidx=y_num, **kwargs)
        return tds_data
    else:
        raise NotImplementedError("Plotting multiple data files are not supported yet")


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


def label_latexify(label):
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
