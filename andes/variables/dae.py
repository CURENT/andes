import logging
from collections import OrderedDict
from andes.shared import pd, np, spmatrix

logger = logging.getLogger(__name__)


class DAETimeSeries(object):
    def __init__(self, dae=None):
        self.dae = dae
        self._data = OrderedDict()
        self._z = OrderedDict()

        self.t = np.array([])
        self.xy = np.array([])
        self.z = np.array([])
        self.df = None
        self.df_z = None

    @property
    def txyz(self):
        """
        Return the values of [t, x, y, z] in an array.
        """
        self.df = pd.DataFrame.from_dict(self._data, orient='index', columns=self.dae.xy_name)
        self.t = self.df.index.to_numpy()
        self.xy = self.df.to_numpy()

        if len(self._z):
            self.df_z = pd.DataFrame.from_dict(self._z, orient='index', columns=self.dae.z_name)
            self.z = self.df_z.to_numpy()
            return np.hstack((self.t.reshape((-1, 1)), self.xy, self.z))
        else:
            return np.hstack((self.t.reshape((-1, 1)), self.xy))

    def store_txyz(self, t, xy, z=None):
        self._data[t] = xy
        if z is not None:
            self._z[t] = z


class DAE(object):
    """
    The numerical DAE class.
    """
    jac_name = ('fx', 'fy', 'gx', 'gy', 'rx', 'tx')
    jac_type = ('c', '')

    def __init__(self, system):
        self.system = system
        self.t = 0
        self.ts = DAETimeSeries(self)

        self.m, self.n, self.o = 0, 0, 0
        self.x, self.y, self.z = np.array([]), np.array([]), np.array([])
        self.f, self.g = np.array([]), np.array([])

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
        self.z_name, self.z_tex_name = [], []

        # ----- indices of sparse matrices -----
        self.ifx, self.jfx, self.vfx = list(), list(), list()
        self.ify, self.jfy, self.vfy = list(), list(), list()
        self.igx, self.jgx, self.vgx = list(), list(), list()
        self.igy, self.jgy, self.vgy = list(), list(), list()
        self.itx, self.jtx, self.vtx = list(), list(), list()
        self.irx, self.jrx, self.vrx = list(), list(), list()

    def clear_ts(self):
        self.ts = DAETimeSeries(self)

    def clear_array(self):
        """
        Reset equation and variable arrays to empty.
        """
        self.clear_fg()
        self.clear_xy()
        self.clear_z()

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

    def clear_z(self):
        """
        Reset status arrays to empty
        """
        self.z = np.zeros(self.n)

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
        self.clear_z()
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
            elif char == 'z':
                ret.append(self.o)
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

    def _compare_pattern(self, name):
        """
        Compare the sparsity pattern for the given Jacobian name.

        This function is for debugging the symbolic factorization error / sparsity pattern change.
        To use, add the following line in `System.j_update` for each `j_name` at the end:

            self.dae._compare_pattern(j_name)
        """
        self.__dict__[f'{name}_tpl'] = spmatrix(self.val_of(name),
                                                self.row_of(name),
                                                self.col_of(name),
                                                self.get_size(name), 'd')
        m_before = self.__dict__[f'{name}_tpl']
        m_after = self.__dict__[name]

        for i in range(len(m_after)):
            if m_after.I[i] != m_before.I[i] or m_after.J[i] != m_before.J[i]:
                raise KeyError

    def resize_array(self):
        """
        Resize arrays to the new `m` and `n`

        Returns
        -------

        """
        self.x = np.append(self.x, np.zeros(self.n - len(self.x)))
        self.y = np.append(self.y, np.zeros(self.m - len(self.y)))
        self.z = np.append(self.z, np.zeros(self.o - len(self.z)))

        self.f = np.append(self.f, np.zeros(self.n - len(self.f)))
        self.g = np.append(self.g, np.zeros(self.m - len(self.g)))

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

    def print_array(self, name, value=None):
        if value is None:
            value = self.__dict__[name]
        res = "\n".join("{:15s} {:<10.4g}".format(x, y) for x, y in zip(self.get_name(name), value))
        logger.info(res)

    def write_lst(self, lst_path):
        """
        Dump the variable name lst file
        :return: succeed flag
        """

        out = ''
        template = '{:>6g}, {:>25s}, {:>35s}\n'

        # header line
        out += template.format(0, 'Time [s]', '$Time\\ [s]$')

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

    def write_npy(self, npy_path):
        np.save(npy_path, self.ts.txyz)
