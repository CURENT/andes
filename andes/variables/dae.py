import logging
from distutils.spawn import find_executable
from typing import Optional, Union, Callable

import numpy as np
from cvxopt.base import spmatrix

import matplotlib as mpl
from matplotlib import pyplot as plt

from andes.common.config import Config
from andes.core.var import BaseVar

logger = logging.getLogger(__name__)


class DAETimeSeries(object):
    def __init__(self):
        self.t_y = None
        self.t_x = None
        self.x = None
        self.y = None
        self.c = None


class DAE(object):
    """
    Numerical DAE class
    """
    jac_name = ('fx', 'fy', 'gx', 'gy', 'rx', 'tx')
    jac_type = ('c', '')

    def __init__(self, config):

        self.t = 0
        self.ts = DAETimeSeries()

        self.m, self.n, self.k = 0, 0, 0

        self.x, self.y, self.c = None, None, None
        self.f, self.g = None, None

        self.fx = None
        self.fy = None
        self.gx = None
        self.gy = None
        self.tx = None
        self.rx = None

        self.fx_pattern = None
        self.fy_pattern = None
        self.gx_pattern = None
        self.gy_pattern = None
        self.tx_pattern = None
        self.rx_pattern = None

        self.x_name, self.x_tex_name = [], []
        self.y_name, self.y_tex_name = [], []

        # ----- indices of sparse matrices -----
        self.ifx, self.jfx, self.vfx = list(), list(), list()
        self.ify, self.jfy, self.vfy = list(), list(), list()
        self.igx, self.jgx, self.vgx = list(), list(), list()
        self.igy, self.jgy, self.vgy = list(), list(), list()
        self.itx, self.jtx, self.vtx = list(), list(), list()
        self.irx, self.jrx, self.vrx = list(), list(), list()

        self.config = Config(self.__class__.__name__)
        self.config.add({'latex': 1,
                         'dpi': 150,
                         })

    def clear_ts(self):
        self.ts = DAETimeSeries()

    def clear_array(self):
        """
        Reset equation and variable arrays to empty.
        """
        self.clear_fg()
        self.clear_xy()

    def clear_fg(self):
        """Resets equation arrays to empty
        """
        self.f = np.zeros(self.n)
        self.g = np.zeros(self.m)

    def clear_xy(self):
        """Reset variable arrays to empty
        """
        self.x = np.zeros(self.n)
        self.y = np.zeros(self.m)
        self.c = np.zeros(self.k)

    def clear_ijv(self):
        self.ifx, self.jfx, self.vfx = list(), list(), list()
        self.ify, self.jfy, self.vfy = list(), list(), list()
        self.igx, self.jgx, self.vgx = list(), list(), list()
        self.igy, self.jgy, self.vgy = list(), list(), list()
        self.itx, self.jtx, self.vtx = list(), list(), list()
        self.irx, self.jrx, self.vrx = list(), list(), list()

    def restore_sparse(self):
        """
        Restore all sparse arrays to shape with non-zero constants
        """
        for name in self.jac_name:
            self.build_pattern(name)

    def reset(self):
        self.m = 0
        self.n = 0
        self.clear_fg()
        self.clear_xy()
        self.clear_ijv()
        self.clear_ts()

    def get_size(self, name):
        """
        Get the size of an array or sparse matrix based on name.

        Parameters
        ----------
        name : str (f, g, fx, gy, etc.)
            array/sparse name

        Returns
        -------
        tuple
            sizes of each element in a tuple
        """
        ret = []
        for char in name:
            if char in ('f', 'x', 't'):
                ret.append(self.n)
            elif char in ('g', 'y', 'r'):
                ret.append(self.m)
        return tuple(ret)

    def store_sparse_ijv(self, name, row, col, val):
        """Store the sparse pattern triplets.

        This function is to be called after building the sparse pattern
        in System.

        Parameters
        ----------
        name : str
            sparse matrix name
        row : np.ndarray
            all row indices
        col : np.ndarray
            all col indices
        val : np.ndarray
            all values
        """
        self.__dict__[f'i{name}'] = row
        self.__dict__[f'j{name}'] = col
        self.__dict__[f'v{name}'] = val

    def row_of(self, name):
        """Get the row indices of the named sparse matrix.

        Parameters
        ----------
        name : str
            name of the sparse matrix

        Returns
        -------
        np.ndarray
            row indices
        """
        return self.__dict__[f'i{name}']

    def col_of(self, name):
        """
        Get the col indices of the named sparse matrix.

        Parameters
        ----------
        name : str
            name of the sparse matrix

        Returns
        -------
        np.ndarray
            col indices
        """
        return self.__dict__[f'j{name}']

    def val_of(self, name):
        """
        Get the values of the named sparse matrix.

        Parameters
        ----------
        name : str
            name of the sparse matrix

        Returns
        -------
        np.ndarray
            values
        """
        return self.__dict__[f'v{name}']

    def build_pattern(self, name):
        """
        Build sparse matrices with stored patterns.

        Call to `store_row_col_idx` should be made before this function.

        Parameters
        ----------
        name : name
            jac name
        """
        self.__dict__[name] = spmatrix(self.val_of(name),
                                       self.row_of(name),
                                       self.col_of(name),
                                       self.get_size(name), 'd')

    def resize_array(self):
        """
        Resize arrays to the new `m` and `n`

        Returns
        -------

        """
        pass
        self.x = np.append(self.x, np.zeros(self.n - len(self.x)))
        self.y = np.append(self.y, np.zeros(self.m - len(self.y)))

        self.f = np.append(self.f, np.zeros(self.n - len(self.f)))
        self.g = np.append(self.g, np.zeros(self.m - len(self.g)))

    @property
    def xy(self):
        return np.hstack((self.x, self.y))

    @property
    def fg(self):
        return np.hstack((self.f, self.g))

    @property
    def xy_name(self):
        return self.x_name + self.y_name

    @property
    def xy_tex_name(self):
        return self.x_tex_name + self.y_tex_name

    def get_name(self, arr):
        mapping = {'f': 'x', 'g': 'y', 'x': 'x', 'y': 'y'}
        return self.__dict__[mapping[arr] + '_name']

    def print_array(self, name, value=None):
        if value is None:
            value = self.__dict__[name]
        res = "\n".join("{:15s} {:<10.4g}".format(x, y) for x, y in zip(self.get_name(name), value))
        logger.info(res)

    def store_yt_single(self):
        """
        Store algebraic variable value and the corresponding t

        Returns
        -------

        """
        # store t
        if self.ts.t_y is None:
            self.ts.t_y = np.array([self.t])
        else:
            self.ts.t_y = np.hstack((self.ts.t_y, self.t))

        # store y
        if self.ts.y is None:
            self.ts.y = np.array(self.y)
        else:
            self.ts.y = np.vstack((self.ts.y, self.y))

    def store_x_single(self):
        """
        Store differential variable value

        Returns
        -------

        """
        # store x
        if self.ts.x is None:
            self.ts.x = np.array(self.x)
        else:
            self.ts.x = np.vstack((self.ts.x, self.x))

    def store_c_single(self):
        """
        Store differential variable value

        Returns
        -------

        """
        raise NotImplementedError
        # if self.ts.c is None:
        #     self.ts.c = np.array(self.c)
        # else:
        #     self.ts.c = np.vstack((self.ts.c, self.c))

    def store_xt_array(self, x, t):
        self.ts.x = x
        self.ts.t_x = t

    def plot(self,
             var,
             idx: Optional[Union[BaseVar, list, np.ndarray]] = None,
             legend: Optional[bool] = False,
             grid: Optional[bool] = True,
             left: Optional[Union[int, float]] = None,
             right: Optional[Union[int, float]] = None,
             fun: Optional[Callable] = None,
             fig=None,
             ax=None):
        if var not in ('x', 'y', 'c'):
            raise ValueError('Only x, y or c is allowed for var')

        # set latex option
        self.set_latex(self.config.latex)

        if isinstance(idx, BaseVar):
            idx = idx.a
        elif isinstance(idx, (int, np.int64)):
            idx = [idx]

        if idx is None or len(idx) == 0:
            value_array = self.ts.__dict__[var]
        else:
            # slice values
            value_array = self.ts.__dict__[var][:, idx]

        # apply callable function `fun`
        if fun is not None:
            value_array = fun(value_array)

        legend_list = self._get_legend(var, idx)

        # get the correct time array
        if self.ts.__dict__['t_' + var] is not None:
            t_array = self.ts.__dict__['t_' + var]
        else:
            t_array = self.ts.t_y  # fallback

        if not left:
            left = t_array[0] - 1e-6
        if not right:
            right = t_array[-1] + 1e-6

        # set latex
        self.set_latex(self.config.latex)

        if fig is None:
            fig = plt.figure(dpi=self.config.dpi)
            ax = plt.gca()

        ls_list = ['-', '--', '-.', ':'] * (int(value_array.shape[1] / 4) + 1)

        for i in range(value_array.shape[1]):
            ax.plot(t_array,
                    value_array[:, i],
                    linestyle=ls_list[i],
                    )

        ax.set_xlim(left=left, right=right)
        ax.ticklabel_format(useOffset=False)

        if grid:
            ax.grid(b=True, linestyle='--')

        if legend is True:
            ax.legend(legend_list)

        return fig, ax

    def set_latex(self, enable=1):
        """
        Enables latex for matplotlib based on the `with_latex` option and `dvipng` availability

        Parameters
        ----------
        enable: bool, optional
            True for latex on and False for off

        Returns
        -------
        bool
            True for latex on and False for off
        """

        if enable == 1:
            if find_executable('dvipng'):
                mpl.rc('text', usetex=True)
                self.config.latex = 1
                return True

        mpl.rc('text', usetex=False)
        self.config.latex = 0
        return False

    def _get_legend(self, var, idx=None):
        attr_name = f'{var}_name'
        attr_tex_name = f'{var}_tex_name'

        if self.config.latex == 1:
            out = self.__dict__[attr_tex_name]
        else:
            out = self.__dict__[attr_name]

        if (idx is not None) and len(idx) > 0:
            out = [out[i] for i in idx]

        return out
