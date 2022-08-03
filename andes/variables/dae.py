import logging
from collections import OrderedDict
from typing import List, Union

import numpy as np

from andes.core import JacTriplet
from andes.core.var import BaseVar, ExtVar
from andes.shared import jac_names, pd, spmatrix

logger = logging.getLogger(__name__)


# TODO: Separate array data and Triplets

class DAETimeSeries:
    """
    DAE time series data.
    """

    def __init__(self, dae=None):
        self.dae = dae

        # accessible attributes
        self._public = ['t', 'x', 'y', 'z', 'xy', 'txyz',
                        'df_x', 'df_y', 'df_z', 'df_xy', 'df_xyz']

        # internal dict storage
        self._xs = OrderedDict()
        self._ys = OrderedDict()
        self._zs = OrderedDict()
        self._fs = OrderedDict()
        self._hs = OrderedDict()
        self._is = OrderedDict()

        self.idx_ptr = 0  # index pointer to the beginning of data that should be written

    def unpack_np(self, attr, warn_empty=True):
        """
        Unpack dict data into numpy arrays.
        """

        n_steps = len(self._ys)

        if attr is None or 't' in attr:
            self.t = np.array(list(self._ys.keys()))

        def _dict2array(src, dest):
            """
            Helper function to convert data stord in a dict to an array.
            """
            if len(self.__dict__[src]):
                nx = len(self.__dict__[src][self.t[0]])
            else:
                nx = 0

            self.__dict__[dest] = np.zeros((n_steps, nx))

            if len(self.__dict__[src]) > 0:
                for ii, val in enumerate(self.__dict__[src].values()):
                    self.__dict__[dest][ii, :] = val

        pairs = (('_xs', 'x'), ('_ys', 'y'), ('_zs', 'z'),
                 ('_fs', 'f'), ('_hs', 'h'), ('_is', 'i'))

        for a, b in pairs:
            if attr is None or b in attr:
                _dict2array(a, b)

        if attr is None or attr == 'xy':
            self.xy = np.hstack((self.x, self.y))
        if attr is None or attr == 'txy':
            self.txy = np.hstack((self.t.reshape((-1, 1)), self.x, self.y))
        if attr is None or attr == 'txyz':
            self.txyz = np.hstack((self.t.reshape((-1, 1)), self.x, self.y, self.z))

        if n_steps == 0:
            if warn_empty:
                logger.warning("TimeSeries does not contain any time stamp.")
            return False

        return True

    def unpack(self, df=False, attr=None, warn_empty=True):
        """
        Unpack dict-stored data into arrays and/or dataframes.

        Parameters
        ----------
        df : bool
            True to construct DataFrames `self.df` and `self.df_z`
            (time-consuming).
        attr : str, optional
            Attribute name to unpack. If None, unpack all.

        Returns
        -------
        True when done.
        """

        self.unpack_np(attr=attr, warn_empty=warn_empty)
        if df is True:
            self.unpack_df(attr=attr)

        return True

    def unpack_df(self, attr):
        """
        Construct pandas dataframes.
        """

        uxname = self.dae.x_name_output
        uyname = self.dae.y_name_output
        uzname = self.dae.z_name

        if attr is None or 'x' in attr:
            self.df_x = pd.DataFrame.from_dict(self._xs, orient='index',
                                               columns=uxname)

        if attr is None or 'y' in attr:
            self.df_y = pd.DataFrame.from_dict(self._ys, orient='index',
                                               columns=uyname)

        if attr is None or 'z' in attr:
            self.df_z = pd.DataFrame.from_dict(self._zs, orient='index',
                                               columns=uzname)

        if attr is None or attr == 'df_xy':
            self.df_xy = pd.concat((self.df_x, self.df_y), axis=1)

        if attr is None or attr == 'df_xyz':
            self.df_xyz = pd.concat((self.df_x, self.df_y, self.df_z), axis=1)

        return True

    def get_data(self, base_vars: Union[BaseVar, List[BaseVar]], *,
                 a=None, rhs: bool = False,):
        """
        Get time-series data, either for a variable or for the equation
        associated with the variable.

        Parameters
        ----------
        base_var : BaseVar or a sequence of BaseVar(s)
            The variable types and internal addresses are used for looking up
            the data.
        a : an array/list of int or None
            Sub-indices into the address of `base_var`. Applied to each
            variable.

        Returns
        -------
        np.ndarray

            A two-dimensional array. Each row corresponds to one time step. Each
            column corresponds to a different different variable.
        """

        out = np.zeros((len(self.t), 0))
        if isinstance(base_vars, BaseVar):
            base_vars = (base_vars, )

        for base_var in base_vars:
            if base_var.n == 0:
                logger.info("Variable <%s.%s> does not contain any element.",
                            base_var.owner.class_name, base_var.name)
                continue

            if (rhs is True) and (base_var.e_code == 'g') and \
                    not isinstance(base_var, ExtVar):
                logger.warning("RHS of the internal Algeb var <%s.%s> is always zero.",
                               base_var.owner.class_name, base_var.name)
                continue

            if rhs is False:
                indices = base_var.a
                array_code = base_var.v_code

                if self.dae.system.Output.n > 0:
                    indices = self.dae.system.Output.to_output_addr(base_var, check=True)
                    if len(indices) == 0:
                        continue

            else:
                if isinstance(base_var, ExtVar):
                    # external algebraic variables
                    indices = base_var.r
                    array_code = base_var.r_code
                else:
                    # internal differential variables
                    indices = base_var.a
                    array_code = base_var.e_code

            indices = indices[a] if a is not None else indices
            out = np.hstack((out, self._access_array(array_code, indices)))

        return out

    def _access_array(self, array_name, indices=None):
        """
        Helper function to access an existing array in TimeSeries.

        The function checks for empty arrays and shows warnings.
        """

        if np.count_nonzero(self.__dict__[array_name]) == 0:
            logger.error("TimeSeries matrix <%s> contains no element. Check if `[TDS] store_%s = 1`",
                         array_name, array_name)

            return None

        if indices is None:
            return self.__dict__[array_name][:, :]
        else:
            return self.__dict__[array_name][:, indices]

    @property
    def df(self):
        """
        Short-hand for the xy dataframe.
        """

        return self.df_xy

    def __getattr__(self, attr):
        if attr in super().__getattribute__('_public'):
            df = True if attr.startswith("df") else False
            self.unpack(df=df, attr=attr)

        return super().__getattribute__(attr)

    def __getstate__(self):
        return self.__dict__

    def reset(self):
        """
        Reset the internal storage and erase all data.
        """
        self._xs = OrderedDict()
        self._ys = OrderedDict()
        self._zs = OrderedDict()
        self._fs = OrderedDict()
        self._hs = OrderedDict()
        self._is = OrderedDict()

        self.unpack_np(attr=None, warn_empty=False)
        self.unpack_df(attr=None)

        self.idx_ptr = 0

        logger.debug("TimeSeries storage is cleared.")


class DAE:
    r"""
    Class for storing numerical values of the DAE system, including variables, equations and first order
    derivatives (Jacobian matrices).

    Variable values and equation values are stored as :py:class:`numpy.ndarray`, while Jacobians are stored as
    :py:class:`kvxopt.spmatrix`. The defined arrays and descriptions are as follows:

    +-----------+---------------------------------------------+
    | DAE Array |                 Description                 |
    +===========+=============================================+
    |  x        | Array for state variable values             |
    +-----------+---------------------------------------------+
    |  y        | Array for algebraic variable values         |
    +-----------+---------------------------------------------+
    |  z        | Array for 0/1 limiter states (if enabled)   |
    +-----------+---------------------------------------------+
    |  f        | Array for differential equation derivatives |
    +-----------+---------------------------------------------+
    |  Tf       | Left-hand side time constant array for f    |
    +-----------+---------------------------------------------+
    |  g        | Array for algebraic equation mismatches     |
    +-----------+---------------------------------------------+

    The defined scalar member attributes to store array sizes are

    +-----------+---------------------------------------------+
    | Scalar    |                 Description                 |
    +===========+=============================================+
    |  m        | The number of algebraic variables/equations |
    +-----------+---------------------------------------------+
    |  n        | The number of algebraic variables/equations |
    +-----------+---------------------------------------------+
    |  o        | The number of limiter state flags           |
    +-----------+---------------------------------------------+

    The derivatives of `f` and `g` with respect to `x` and `y` are stored in four :py:mod:`kvxopt.spmatrix`
    sparse matrices:
    **fx**, **fy**, **gx**, and **gy**,
    where the first letter is the equation name, and the second letter is the variable name.

    Notes
    -----
    DAE in ANDES is defined in the form of

    .. math ::
        T \dot{x} = f(x, y) \\
        0 = g(x, y)

    DAE does not keep track of the association of variable and address.
    Only a variable instance keeps track of its addresses.
    """

    def __init__(self, system):
        self.system = system
        self.t = np.array(0.0, dtype=float)
        self.ts = DAETimeSeries(self)
        self.kcount = 0  # time step count

        self._array_and_counter = {
            'f': 'n',  # differential equation RHS
            'x': 'n',  # differential variables
            'g': 'm',  # algebraic equation residual
            'y': 'm',  # algebraic variables
            'z': 'o',  # limiter flags
            'h': 'p',  # RHS of external states
            'i': 'q',  # RHS of external algebraic variables
        }

        self.m, self.n, self.o, self.p, self.q = 0, 0, 0, 0, 0

        self.x, self.y, self.z = np.array([]), np.array([]), np.array([])
        self.f, self.g = np.array([]), np.array([])  # RHS of equations
        self.h, self.i = np.array([]), np.array([])  # RHS of external equations

        # `self.Tf` is the time-constant array for differential equations
        self.Tf = np.array([])

        self.fx, self.fy = None, None
        self.gx, self.gy = None, None
        self.rx, self.tx = None, None

        self.x_name, self.x_tex_name = [], []
        self.y_name, self.y_tex_name = [], []
        self.z_name, self.z_tex_name = [], []
        self.h_name, self.h_tex_name = [], []
        self.i_name, self.i_tex_name = [], []

        self.triplets = JacTriplet()

        self.tpl = dict()  # sparsity templates with constants

        self._write_append = False  # True if data should be appended when writing to output
        self._lst_written = False

    def request_address(self, array_name: str, ndevice, nvar, collate=False):
        """
        Interface for requesting addresses for a model.

        Parameters
        ----------
        array_name : str
            array name in 'x' and 'y'
        ndevice : int
            number of devices
        nvar : int
            number of variables
        collate : bool, optional
            False if the same variable for different devices are contiguous.
            True if variables for the same devices should collate. Note: setting
            ``collate`` to True will degrade the performance.

        Returns
        -------
        list
            A list of arrays for each variable.
        """

        out = []
        counter_name = self._array_and_counter[array_name]

        idx_begin = self.__dict__[counter_name]
        idx_end = idx_begin + ndevice * nvar

        if not collate:
            for idx in range(nvar):
                out.append(np.arange(idx_begin + idx * ndevice, idx_begin + (idx + 1) * ndevice))
        else:
            for idx in range(nvar):
                out.append(np.arange(idx_begin + idx, idx_end, nvar))

        self.__dict__[counter_name] = idx_end

        return out

    def clear_ts(self):
        """
        Drop the TimeSeries data and create a new one.
        """

        self.ts = DAETimeSeries(self)

    def clear_arrays(self):
        """
        Reset equation and variable arrays to empty.
        """

        self.clear_fg()
        self.clear_xy()
        self.clear_z()

    def clear_fg(self):
        """
        Resets equation arrays to empty.
        """

        self.f[:] = 0
        self.g[:] = 0

    def clear_xy(self):
        """
        Reset variable arrays to empty.
        """

        self.x[:] = 0
        self.y[:] = 0

    def clear_z(self):
        """
        Reset status arrays to empty
        """

        self.z[:] = 0

    def clear_ijv(self):
        """
        Clear stored triplets.
        """

        self.triplets.clear_ijv()

    def restore_sparse(self, names=None):
        """
        Restore all sparse matrices to the sparsity pattern
        filled with zeros (for variable Jacobian elements)
        and non-zero constants.

        Parameters
        ----------
        names : None or list
            List of Jacobian names to restore sparsity pattern
        """

        if names is None:
            names = jac_names
        elif isinstance(names, str):
            names = [names]

        for name in names:
            self.__dict__[name] = spmatrix(self.tpl[name].V,
                                           self.tpl[name].I,
                                           self.tpl[name].J,
                                           self.tpl[name].size, 'd')

    def reset(self):
        """
        Reset array sizes to zero and clear all arrays.
        """

        self.set_t(0.0)
        self.m = 0
        self.n = 0
        self.o = 0
        self.resize_arrays()
        self.clear_ijv()
        self.clear_ts()

    def set_t(self, t):
        """
        Helper function for setting time in-place.
        """

        self.t.itemset(t)

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
            if char in ('f', 'x'):
                ret.append(self.n)
            elif char in ('g', 'y'):
                ret.append(self.m)
            elif char == 'z':
                ret.append(self.o)
        return tuple(ret)

    def store_sparse_ijv(self, name, row, col, val):
        """
        Store the sparse pattern triplets.

        This function is to be called by System after building the complete sparsity pattern for each Jacobian
        matrix.

        Parameters
        ----------
        name : str
            sparse Jacobian matrix name
        row : np.ndarray
            all row indices
        col : np.ndarray
            all col indices
        val : np.ndarray
            all values
        """
        self.triplets.ijac[name] = row
        self.triplets.jjac[name] = col
        self.triplets.vjac[name] = val

    def build_pattern(self, name):
        """
        Build sparse matrices with stored patterns.

        Call to `store_row_col_idx` should be made before this function.

        Parameters
        ----------
        name : name
            jac name
        """
        try:
            self.tpl[name] = spmatrix(self.triplets.vjac[name],
                                      self.triplets.ijac[name],
                                      self.triplets.jjac[name],
                                      self.get_size(name), 'd')
        except TypeError as e:
            logger.error("Your new model might have accessed an Algeb using ExtState, or vice versa.")
            raise e

        self.restore_sparse(name)

    def _compare_pattern(self, name):
        """
        Compare the sparsity pattern for the given Jacobian name.

        This function is for debugging the symbolic factorization error / sparsity pattern change.
        To use, add the following line in `System.j_update` for each `j_name` at the end:

            self.dae._compare_pattern(j_name)
        """
        self.__dict__[f'{name}_tpl'] = spmatrix(self.triplets.vjac[name],
                                                self.triplets.ijac[name],
                                                self.triplets.jjac[name],
                                                self.get_size(name), 'd')
        m_before = self.__dict__[f'{name}_tpl']
        m_after = self.__dict__[name]

        for i in range(len(m_after)):
            if m_after.I[i] != m_before.I[i] or m_after.J[i] != m_before.J[i]:
                raise KeyError

    def store(self):
        """
        Store values for the current time step to the TimeSeries storage. Values
        include variables, equation RHS and discrete states.
        """

        system = self.system
        tds = self.system.TDS
        ts = self.ts
        t = self.t.tolist()

        if system.Output.n > 0:
            # select variables based on `Output`
            ts._xs[t] = self.x[system.Output.xidx]
            ts._ys[t] = self.y[system.Output.yidx]
        else:
            ts._xs[t] = np.array(self.x)
            ts._ys[t] = np.array(self.y)

        if tds.config.store_z:
            z_vals = system.get_z(system.exist.pflow_tds)
            ts._zs[t] = np.array(z_vals)

        if tds.config.store_f:
            ts._fs[t] = np.array(self.f)
        if tds.config.store_h:
            ts._hs[t] = np.array(self.h)
        if tds.config.store_i:
            ts._is[t] = np.array(self.i)

    def resize_arrays(self):
        """
        Resize arrays to the new sizes `m` and `n`, and `o`.

        If ``m > len(self.y)`` or ``n > len(self.x``, arrays will be extended.
        Otherwise, new empty arrays will be sliced, starting from 0 to the given
        size.

        Warnings
        --------
        This function should not be called directly. Instead, it is called in
        ``System.set_address`` which re-points variables used in power flow to
        the new array for dynamic analyses.
        """
        self.x = self._extend_or_slice(self.x, self.n)
        self.y = self._extend_or_slice(self.y, self.m)
        self.z = self._extend_or_slice(self.z, self.o)

        self.f = self._extend_or_slice(self.f, self.n)
        self.g = self._extend_or_slice(self.g, self.m)
        self.h = self._extend_or_slice(self.h, self.p)
        self.i = self._extend_or_slice(self.i, self.q)

        self.Tf = self._extend_or_slice(self.Tf, self.n, fill_func=np.ones)

    def _extend_or_slice(self, array, new_size, fill_func=np.zeros):
        """
        Helper function for ``self.resize_arrays`` to grow or shrink arrays.
        """
        if new_size > len(array):
            array = np.append(array, fill_func(new_size - len(array)))
        else:
            array = array[0:new_size]
        return array

    def alloc_or_extend_names(self):
        """
        Allocate empty lists for names for the given size.
        """
        specs = {'x_name': self.n,
                 'y_name': self.m,
                 'h_name': self.p,
                 'i_name': self.q,
                 'x_tex_name': self.n,
                 'y_tex_name': self.m,
                 'h_tex_name': self.p,
                 'i_tex_name': self.q,
                 }

        for name, size in specs.items():
            length = len(self.__dict__[name])
            if length == 0:
                self.__dict__[name] = [''] * size
            elif 0 < length <= size:
                self.__dict__[name].extend([''] * (size - length))
            else:
                raise NotImplementedError("Does not know how to shrink arrays")

    @property
    def xy(self):
        """
        Return a concatenated array of [x, y].
        """

        return np.hstack((self.x, self.y))

    @property
    def xyz(self):
        """
        Return a concatenated array of [x, y].
        """

        return np.hstack((self.x, self.y, self.z))

    @property
    def fg(self):
        """
        Return a concatenated array of [f, g].
        """

        return np.hstack((self.f, self.g))

    @property
    def x_name_output(self):
        """
        Return a list of state var names selected by Output.
        """
        if self.system.Output.n == 0:
            return self.x_name
        else:
            return [self.x_name[i] for i in self.system.Output.xidx]

    @property
    def y_name_output(self):
        """
        Return a list of algeb var names selected by Output.
        """

        if self.system.Output.n == 0:
            return self.y_name
        else:
            return [self.y_name[i] for i in self.system.Output.yidx]

    @property
    def x_tex_name_output(self):
        """
        Return a list of state var LaTeX names selected by Output.
        """

        if self.system.Output.n == 0:
            return self.x_tex_name
        else:
            return [self.x_tex_name[i] for i in self.system.Output.xidx]

    @property
    def y_tex_name_output(self):
        """
        Return a list of algeb var LaTeX names selected by Output.
        """

        if self.system.Output.n == 0:
            return self.y_tex_name
        else:
            return [self.y_tex_name[i] for i in self.system.Output.yidx]

    @property
    def xy_name(self):
        """
        Return a concatenated list of all variable names without format.
        """

        return self.x_name + self.y_name

    @property
    def xyz_name(self):
        """
        Return a concatenated list of all variable names without format.
        """

        return self.x_name + self.y_name + self.z_name

    @property
    def xy_tex_name(self):
        """
        Return a concatenated list of all variable names in LaTeX format.
        """

        return self.x_tex_name + self.y_tex_name

    @property
    def xyz_tex_name(self):
        """
        Return a concatenated list of all variable names in LaTeX format.
        """

        return self.x_tex_name + self.y_tex_name + self.z_tex_name

    def get_name(self, arr):
        """
        Helper function for geting the list of variable names based on the
        array name.

        Parameters
        ----------
        arr : str
            Array name in 'f', 'g', 'x', 'y', 'z'.
        """

        mapping = {'f': 'x', 'g': 'y', 'x': 'x', 'y': 'y', 'z': 'z'}
        return self.__dict__[mapping[arr] + '_name']

    def print_array(self, name, values=None, tol=None):
        """
        Debug helper to print array values and names.

        Parameters
        ----------
        name : str
            array name in 'f', 'g', 'x', 'y'
        values : array-like, optional
            substitute array values to use
        tol : float, optional
            tolerance value to use. Values below `tol` will not be displayed
        """

        if values is None:
            values = self.__dict__[name]

        indices = list(range(len(values)))
        if tol is not None:
            indices = np.where(abs(values) >= tol)
            values = values[indices]

        name_list = np.array(self.get_name(name))[indices]

        if not len(name_list):
            return
        logger.info(f"Debug Print at {self.t:.4f}")
        res = "\n".join("{:15s} {:<10.4g}".format(x, y) for x, y in zip(name_list, values))
        logger.info(res)

    def write_lst(self, lst_path):
        """
        Dump the variable name lst file.

        Parameters
        ----------
        lst_path
            Path to the lst file.

        Returns
        -------
        bool
            succeed flag
        """

        if self._lst_written is True:
            return

        system = self.system

        out = ''
        template = '{:>6g}, {:>25s}, {:>35s}\n'

        # header line
        out += template.format(0, 'Time [s]', 'Time [s]')

        if system.Output.n == 0:
            # output variable indices
            idx = list(range(self.m + self.n + self.o))

            # variable names concatenated
            uname = self.xyz_name
            fname = self.xyz_tex_name
        else:
            idx = list(range(len(system.Output.xidx) + len(system.Output.yidx) + self.o))
            uname = self.x_name_output + self.y_name_output + self.z_name
            fname = self.x_tex_name_output + self.y_tex_name_output + self.z_tex_name

        for e, i in enumerate(idx):
            # `idx` in the lst file is always consecutive
            out += template.format(e + 1, uname[i], fname[i])

        with open(lst_path, 'w') as f:
            f.write(out)

        self._lst_written = True

        return True

    def write_npy(self, file_path):
        """
        Write TDS data into NumPy uncompressed format.
        """

        txyz_data = self.ts.txyz
        np.save(file_path, txyz_data)

    def write_npz(self, file_path):
        """
        Write TDS data into NumPy compressed format.

        The function supports writing out all values at once or writing them out
        incrementally.
        """

        tds = self.system.TDS
        ts = self.ts

        if not tds.config.limit_store:
            # write the whole TimeSeries in one step
            txyz_data = self.ts.txyz
            np.savez_compressed(file_path, data=txyz_data)

        else:
            # create a new npz file and write for the first time
            if self._write_append is False:
                txyz_data = self.ts.txyz[ts.idx_ptr:, :]
                np.savez_compressed(file_path, data=txyz_data)
                self._write_append = True
                ts.idx_ptr = len(self.ts.t)

            # write and append to an existing npz file
            else:
                self.ts.unpack()
                txyz_data = self.ts.txyz[ts.idx_ptr:, :]

                # skip if no new data
                if len(txyz_data) == 0:
                    logger.debug("No new data to write to file. Skipped.")
                    return

                data = np.load(file_path)['data']

                # in most cases, append new data to the existing
                if len(data) > 0:
                    data = np.vstack((data, txyz_data))
                    logger.debug("Appended new data to output file.")

                # in case the previous step stopped at tf=0
                else:
                    data = txyz_data

                np.savez_compressed(file_path, data=data)
                ts.idx_ptr = len(self.ts.t)
