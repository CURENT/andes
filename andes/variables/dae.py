import logging
import numpy as np

from typing import List, Union
from collections import OrderedDict

from andes.core import JacTriplet
from andes.core.var import BaseVar
from andes.shared import pd, spmatrix, jac_names

logger = logging.getLogger(__name__)


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

    def store(self, t, x, y, *, z=None, f=None, h=None, i=None,):
        """
        Store t, x, y, and z in internal storage, respectively.

        Parameters
        ----------
        t : float
            simulation time
        x, y : array-like
            array data for states and algebraic variables
        z : array-like or None
            discrete flags data
        """
        self._xs[t] = np.array(x)
        self._ys[t] = np.array(y)
        if z is not None:
            self._zs[t] = np.array(z)
        if f is not None:
            self._fs[t] = np.array(f)
        if h is not None:
            self._hs[t] = np.array(h)
        if i is not None:
            self._is[t] = np.array(i)

    def unpack_np(self):
        """
        Unpack dict data into numpy arrays.
        """
        n_steps = len(self._ys)

        self.t = np.array(list(self._ys.keys()))

        def _dict2array(src, dest):
            nx = len(self.__dict__[src][0]) if len(self.__dict__[src]) else 0
            self.__dict__[dest] = np.zeros((n_steps, nx))

            if len(self.__dict__[src]) > 0:
                for ii, val in enumerate(self.__dict__[src].values()):
                    self.__dict__[dest][ii, :] = val

        pairs = (('_xs', 'x'), ('_ys', 'y'), ('_zs', 'z'),
                 ('_fs', 'f'), ('_hs', 'h'), ('_is', 'i'))

        for a, b in pairs:
            _dict2array(a, b)

        self.xy = np.hstack((self.x, self.y))
        self.txy = np.hstack((self.t.reshape((-1, 1)), self.xy))
        self.txyz = np.hstack((self.t.reshape((-1, 1)), self.xy, self.z))

        if n_steps == 0:
            logger.warning("TimeSeries does not contain any time stamp.")
            return False
        return True

    def unpack(self, df=False):
        """
        Unpack dict-stored data into arrays and/or dataframes.

        Parameters
        ----------
        df : bool
            True to construct DataFrames `self.df` and `self.df_z` (time-consuming).

        Returns
        -------
        True when done.
        """

        self.unpack_np()
        if df is True:
            self.unpack_df()

        return True

    def unpack_df(self):
        """
        Construct pandas dataframes.
        """

        self.df_x = pd.DataFrame.from_dict(self._xs, orient='index',
                                           columns=self.dae.x_name)
        self.df_y = pd.DataFrame.from_dict(self._ys, orient='index',
                                           columns=self.dae.y_name)
        self.df_z = pd.DataFrame.from_dict(self._zs, orient='index',
                                           columns=self.dae.z_name)

        self.df_xy = pd.concat((self.df_x, self.df_y), axis=1)
        self.df_xyz = pd.concat((self.df_xy, self.df_z), axis=1)

        return True

    def get_data(self, base_vars: Union[BaseVar, List[BaseVar]], a=None):
        """
        Get time-series data for a variable instance.

        Values for different variables will be stacked horizontally.

        Parameters
        ----------
        base_var : BaseVar or a sequence of BaseVar(s)
            The variable types and internal addresses
            are used for looking up the data.
        a : an array-like of int or None
            Sub-indices into the address of `base_var`.
            Applied to each variable.

        """
        out = np.zeros((len(self.t), 0))
        if isinstance(base_vars, BaseVar):
            base_vars = (base_vars, )

        for base_var in base_vars:
            if base_var.n == 0:
                logger.info("Variable <%s.%s> does not contain any element.",
                            base_var.owner.class_name, base_var.name)
                continue

            indices = base_var.a
            if a is not None:
                indices = indices[a]

            out = np.hstack((out, self.__dict__[base_var.v_code][:, indices]))

        return out

    @property
    def df(self):
        return self.df_xy

    def __getattr__(self, attr):
        if attr in super().__getattribute__('_public'):
            df = True if attr.startswith("df") else False
            self.unpack(df=df)

        return super().__getattribute__(attr)


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
        self.t = np.array(0)
        self.ts = DAETimeSeries(self)

        self.m, self.n, self.o = 0, 0, 0
        self.p, self.q = 0, 0

        self.x, self.y, self.z = np.array([]), np.array([]), np.array([])
        self.f, self.g = np.array([]), np.array([])
        self.h, self.i = np.array([]), np.array([])

        # `self.Tf` is the time-constant array for differential equations
        self.Tf = np.array([])

        self.fx, self.fy = None, None
        self.gx, self.gy = None, None
        self.rx, self.tx = None, None

        self.h_name, self.h_tex_name = [], []
        self.i_name, self.i_tex_name = [], []
        self.x_name, self.x_tex_name = [], []
        self.y_name, self.y_tex_name = [], []
        self.z_name, self.z_tex_name = [], []

        self.triplets = JacTriplet()

        self.tpl = dict()  # sparsity templates with constants

    def clear_ts(self):
        self.ts = DAETimeSeries(self)

    def clear_arrays(self):
        """
        Reset equation and variable arrays to empty.
        """
        self.clear_fg()
        self.clear_xy()
        self.clear_z()

    def clear_fg(self):
        """Resets equation arrays to empty
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
        """Helper function for setting time in-place"""
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
            if char in ('f', 'x', 't'):
                ret.append(self.n)
            elif char in ('g', 'y', 'r'):
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

    def resize_arrays(self):
        """
        Resize arrays to the new sizes `m` and `n`, and `o`.

        If ``m > len(self.y)`` or ``n > len(self.x``, arrays will be extended.
        Otherwise, new empty arrays will be sliced, starting from 0 to the given size.

        Warnings
        --------
        This function should not be called directly. Instead, it is called in
        ``System.set_address`` which re-points variables used in power flow
        to the new array for dynamic analyses.
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

    @property
    def xy(self):
        """Return a concatenated array of [x, y]."""
        return np.hstack((self.x, self.y))

    @property
    def xyz(self):
        """Return a concatenated array of [x, y]."""
        return np.hstack((self.x, self.y, self.z))

    @property
    def fg(self):
        """Return a concatenated array of [f, g]."""
        return np.hstack((self.f, self.g))

    @property
    def xy_name(self):
        """Return a concatenated list of all variable names without format."""
        return self.x_name + self.y_name

    @property
    def xyz_name(self):
        """Return a concatenated list of all variable names without format."""
        return self.x_name + self.y_name + self.z_name

    @property
    def xy_tex_name(self):
        """Return a concatenated list of all variable names in LaTeX format."""
        return self.x_tex_name + self.y_tex_name

    @property
    def xyz_tex_name(self):
        """Return a concatenated list of all variable names in LaTeX format."""
        return self.x_tex_name + self.y_tex_name + self.z_tex_name

    def get_name(self, arr):
        mapping = {'f': 'x', 'g': 'y', 'x': 'x', 'y': 'y', 'z': 'z'}
        return self.__dict__[mapping[arr] + '_name']

    def print_array(self, name, values=None, tol=None):
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

        out = ''
        template = '{:>6g}, {:>25s}, {:>35s}\n'

        # header line
        out += template.format(0, 'Time [s]', 'Time [s]')

        # output variable indices
        idx = list(range(self.m + self.n + self.o))

        # variable names concatenated
        uname = self.xyz_name
        fname = self.xyz_tex_name

        for e, i in enumerate(idx):
            # `idx` in the lst file is always consecutive
            out += template.format(e + 1, uname[i], fname[i])

        with open(lst_path, 'w') as f:
            f.write(out)
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
        """
        txyz_data = self.ts.txyz
        np.savez_compressed(file_path, data=txyz_data)
