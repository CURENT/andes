import logging
from collections import OrderedDict
from andes.core import JacTriplet
from andes.shared import pd, np, spmatrix, jac_names

logger = logging.getLogger(__name__)


class DAETimeSeries(object):
    """
    DAE time series data
    """
    def __init__(self, dae=None):
        self.dae = dae

        # internal dict storage
        self._xy = OrderedDict()
        self._z = OrderedDict()

        self.t = np.array([])
        self.xy = np.array([]).reshape((-1, 1))
        self.z = np.array([]).reshape((-1, 1))

        # data frame members
        self.df = None
        self.df_z = None

    @property
    def x(self):
        return self.xy[:, 0:self.dae.n]

    @property
    def y(self):
        return self.xy[:, self.dae.n:self.dae.n + self.dae.m]

    @property
    def txyz(self):
        """
        Return the values of [t, x, y, z] in an array.
        """
        if len(self._xy) != len(self.t):
            self.unpack()

        if len(self._z):
            return np.hstack((self.t.reshape((-1, 1)), self.xy, self.z))
        else:
            return np.hstack((self.t.reshape((-1, 1)), self.xy))

    def unpack(self, df=False):
        """
        Unpack stored data in `_xy` and `_z` into arrays `t`, `xy`, and `z`.

        Parameters
        ----------
        df : bool
            True to construct DataFrames `self.df` and `self.df_z` (time-consuming).
        """
        if df is True:
            self.df = pd.DataFrame.from_dict(self._xy, orient='index', columns=self.dae.xy_name)
            self.t = self.df.index.to_numpy()
            self.xy = self.df.to_numpy()

            self.df_z = pd.DataFrame.from_dict(self._z, orient='index', columns=self.dae.z_name)
            self.z = self.df_z.to_numpy()
        else:
            n_steps = len(self._xy)
            self.t = np.array(list(self._xy.keys()))
            self.xy = np.zeros((n_steps, self.dae.m + self.dae.n))
            self.z = np.zeros((n_steps, self.dae.o))

            for idx, xy in enumerate(self._xy.values()):
                self.xy[idx, :] = xy

            for idx, z in enumerate(self._z.values()):
                self.z[idx, :] = z

    def store_txyz(self, t, xy, z=None):
        """
        Store t, xy, and z in internal storage, respectively.

        Parameters
        ----------
        t : float
            simulation time
        xy : array-like
            array data for states and algebraic variables
        z : array-like or None
            discrete flags data
        """
        self._xy[t] = xy
        if z is not None:
            self._z[t] = z


class DAE(object):
    r"""
    Class for storing numerical values of the DAE system, including variables, equations and first order
    derivatives (Jacobian matrices).

    Variable values and equation values are stored as :py:class:`numpy.ndarray`, while Jacobians are stored as
    :py:class:`cvxopt.spmatrix`. The defined arrays and descriptions are as follows:

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

    The derivatives of `f` and `g` with respect to `x` and `y` are stored in four :py:mod:`cvxopt.spmatrix`
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
        self.x, self.y, self.z = np.array([]), np.array([]), np.array([])
        self.f, self.g = np.array([]), np.array([])
        # `self.Tf` is the time-constant array for differential equations
        self.Tf = np.array([])

        self.fx, self.fy = None, None
        self.gx, self.gy = None, None
        self.rx, self.tx = None, None

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
