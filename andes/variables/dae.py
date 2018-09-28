from cvxopt import matrix, spmatrix, sparse, spdiag
from ..utils.math import zeros, ones
from ..utils.math import ageb, aleb, aandb, agtb  # NOQA
from ..utils.math import index, altb  # NOQA
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DAE(object):
    """Class for numerical Differential Algebraic Equations (dae)"""

    def __init__(self, system):
        self.system = system
        self._data = dict(
            x=[],
            y=[],
            f=[],
            g=[],
            Fx=[],
            Fy=[],
            Gx=[],
            Gy=[],
            Fx0=[],
            Fy0=[],
            Gx0=[],
            Gy0=[],
            Ac=[],
            tn=[])

        self._scalars = dict(
            m=0,
            n=0,
            lamda=0,
            npf=0,
            kg=0,
            t=-1,
            factorize=True,
            rebuild=True,
            ac_reset=True)

        self._flags = dict(
            zxmax=[], zxmin=[], zymax=[], zymin=[], ux=[], uy=[])

        self.__dict__.update(self._data)
        self.__dict__.update(self._scalars)
        self.__dict__.update(self._flags)

        # self._temp = {'Fx': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                      'J': np.ndarray((0, 1), dtype=np.int32),
        #                      'V': np.ndarray((0, 1))},
        #               'Fy': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                      'J': np.ndarray((0, 1), dtype=np.int32),
        #                      'V': np.ndarray((0, 1))},
        #               'Gx': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                      'J': np.ndarray((0, 1), dtype=np.int32),
        #                      'V': np.ndarray((0, 1))},
        #               'Gy': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                      'J': np.ndarray((0, 1), dtype=np.int32),
        #                      'V': np.ndarray((0, 1))},
        #               'Fx0': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                       'J': np.ndarray((0, 1), dtype=np.int32),
        #                       'V': np.ndarray((0, 1))},
        #               'Fy0': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                       'J': np.ndarray((0, 1), dtype=np.int32),
        #                       'V': np.ndarray((0, 1))},
        #               'Gx0': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                       'J': np.ndarray((0, 1), dtype=np.int32),
        #                       'V': np.ndarray((0, 1))},
        #               'Gy0': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                       'J': np.ndarray((0, 1), dtype=np.int32),
        #                       'V': np.ndarray((0, 1))},
        #               }
        #
        # self._set = {'Fx': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                     'J': np.ndarray((0, 1), dtype=np.int32),
        #                     'V': np.ndarray((0, 1))},
        #              'Fy': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                     'J': np.ndarray((0, 1), dtype=np.int32),
        #                     'V': np.ndarray((0, 1))},
        #              'Gx': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                     'J': np.ndarray((0, 1), dtype=np.int32),
        #                     'V': np.ndarray((0, 1))},
        #              'Gy': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                     'J': np.ndarray((0, 1), dtype=np.int32),
        #                     'V': np.ndarray((0, 1))},
        #              'Fx0': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                      'J': np.ndarray((0, 1), dtype=np.int32),
        #                      'V': np.ndarray((0, 1))},
        #              'Fy0': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                      'J': np.ndarray((0, 1), dtype=np.int32),
        #                      'V': np.ndarray((0, 1))},
        #              'Gx0': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                      'J': np.ndarray((0, 1), dtype=np.int32),
        #                      'V': np.ndarray((0, 1))},
        #              'Gy0': {'I': np.ndarray((0, 1), dtype=np.int32),
        #                      'J': np.ndarray((0, 1), dtype=np.int32),
        #                      'V': np.ndarray((0, 1))},
        #              }

        self._temp = {
            'Fx': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fy': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gx': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gy': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fx0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fy0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gx0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gy0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        }

        self._set = {
            'Fx': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fy': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gx': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gy': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fx0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fy0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gx0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gy0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        }

    def init_xy(self):
        self.init_x()
        self.init_y()

    def init_x(self):
        self.x = zeros(self.n, 1)
        self.zxmax = ones(self.n, 1)
        self.zxmin = ones(self.n, 1)
        self.ux = ones(self.n, 1)

    def init_y(self):
        self.y = zeros(self.m, 1)
        self.zymax = ones(self.m, 1)
        self.zymin = ones(self.m, 1)
        self.uy = ones(self.m, 1)

    def init_fg(self, resetz=False):
        self.init_f(resetz)
        self.init_g(resetz)

    def init_f(self, resetz=True):
        self.zxmax = ones(self.n, 1)
        self.zxmin = ones(self.n, 1)
        self.f = zeros(self.n, 1)

    def init_g(self, resetz=True):
        self.zymax = ones(self.m, 1)
        self.zymin = ones(self.m, 1)
        self.g = zeros(self.m, 1)

    def setup_Gy(self):
        self.Gy = sparse(self.Gy0)
        self._temp.update({
            'Gy': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        })

        self._set.update({
            'Gy': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        })

    def setup_Fx(self):
        # self.Fx = sparse(self.Fx0)
        # self.Fy = sparse(self.Fy0)
        # self.Gx = sparse(self.Gx0)

        self._temp.update({
            'Fx': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fy': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gx': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        })

        self._set.update({
            'Fx': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fy': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gx': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        })

    def setup_FxGy(self):
        self.setup_Fx()
        self.setup_Gy()

    def setup(self):
        self.init_xy()
        self.init_fg()
        self.init_jac0()
        self.setup_Gy()
        self.setup_Fx()

    def init_Gy0(self):
        self.Gy0 = spmatrix([], [], [], (self.m, self.m), 'd')

        self._temp.update({
            'Gy0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        })

        self._set.update({
            'Gy0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        })

    def init_Fx0(self):
        self.Gx0 = spmatrix([], [], [], (self.m, self.n), 'd')
        self.Fy0 = spmatrix([], [], [], (self.n, self.m), 'd')
        self.Fx0 = spmatrix([], [], [], (self.n, self.n), 'd')

        self._temp.update({
            'Fx0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fy0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gx0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        })

        self._set.update({
            'Fx0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Fy0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
            'Gx0': {
                'I': matrix([]),
                'J': matrix([]),
                'V': matrix([])
            },
        })

    def init_jac0(self):
        self.init_Gy0()
        self.init_Fx0()

    def init1(self):
        self.resize()
        self.init_jac0()

    def resize(self):
        """Resize dae and and extend for init1 variables
        """
        yext = self.m - len(self.y)
        xext = self.n - len(self.x)
        if yext > 0:
            yzeros = zeros(yext, 1)
            yones = ones(yext, 1)
            self.y = matrix([self.y, yzeros], (self.m, 1), 'd')
            self.g = matrix([self.g, yzeros], (self.m, 1), 'd')
            self.uy = matrix([self.uy, yones], (self.m, 1), 'd')
            self.zymin = matrix([self.zymin, yones], (self.m, 1), 'd')
            self.zymax = matrix([self.zymax, yones], (self.m, 1), 'd')
        if xext > 0:
            xzeros = zeros(xext, 1)
            xones = ones(xext, 1)
            self.x = matrix([self.x, xzeros], (self.n, 1), 'd')
            self.f = matrix([self.f, xzeros], (self.n, 1), 'd')
            self.ux = matrix([self.ux, xones], (self.n, 1), 'd')
            self.zxmin = matrix([self.zxmin, xones], (self.n, 1), 'd')
            self.zxmax = matrix([self.zxmax, xones], (self.n, 1), 'd')

    def hard_limit(self, yidx, ymin, ymax, min_set=None, max_set=None):
        """Set hard limits for algebraic variables and reset the equation mismatches

        :param yidx: algebraic variable indices
        :param ymin: lower limit to check for
        :param ymax: upper limit to check for
        :param min_set: optional lower limit to set (``ymin`` as default)
        :param max_set: optional upper limit to set (``ymax`` as default)

        :type yidx: list, matrix
        :type ymin: matrix, int, float, list
        :type ymax: matrix, int, float, list
        :type min_set: matrix
        :type max_set: matrix

        :return: None

        """
        yidx = matrix(yidx)
        yval = self.y[yidx]
        ny = len(yidx)

        if isinstance(ymin, (int, float, list)):
            ymin = matrix(ymin, (ny, 1), 'd')
        if isinstance(ymax, (int, float, list)):
            ymax = matrix(ymax, (ny, 1), 'd')

        if not min_set:
            min_set = ymin
        elif isinstance(min_set, (int, float, list)):
            min_set = matrix(min_set, (ny, 1), 'd')

        if not max_set:
            max_set = ymax
        elif isinstance(max_set, (int, float, list)):
            max_set = matrix(max_set, (ny, 1), 'd')

        above = ageb(yval, ymax)
        below = aleb(yval, ymin)

        above_idx = index(above, 1.0)
        below_idx = index(below, 1.0)

        above_yidx = yidx[above_idx]
        below_yidx = yidx[below_idx]
        idx = list(above_idx) + list(below_idx)

        if len(above_yidx) > 0:
            self.y[above_yidx] = max_set[above_idx]
            self.zymax[above_yidx] = 0

        if len(below_yidx) > 0:
            self.y[below_yidx] = min_set[below_idx]
            self.zymin[below_yidx] = 0

        if len(idx):
            self.g[yidx[idx]] = 0
            self.ac_reset = True

    def hard_limit_remote(self,
                          yidx,
                          ridx,
                          rtype='y',
                          rmin=None,
                          rmax=None,
                          min_yset=0,
                          max_yset=0):
        """Limit the output of yidx if the remote y is not within the limits

        This function needs to be modernized.
        """
        ny = len(yidx)
        assert ny == len(
            ridx), "Length of output vars and remote vars does not match"
        assert rtype in ('x',
                         'y'), "ridx must be either y (algeb) or x (state)"

        if isinstance(min_yset, (int, float)):
            min_yset = matrix(min_yset, (ny, 1), 'd')
        if isinstance(max_yset, (int, float)):
            max_yset = matrix(max_yset, (ny, 1), 'd')

        above_idx, below_idx = list(), list()
        yidx = matrix(yidx)

        if rmax:
            # find the over-limit remote idx
            above = ageb(self.__dict__[rtype][ridx], rmax)
            above_idx = index(above, 1.0)
            # reset the y values based on the remote limit violations
            self.y[yidx[above_idx]] = max_yset[above_idx]
            self.zymax[yidx[above_idx]] = 0

        if rmin:
            below = aleb(self.__dict__[rtype][ridx], rmin)
            below_idx = index(below, 1.0)
            self.y[yidx[below_idx]] = min_yset[below_idx]
            self.zymin[yidx[below_idx]] = 0

        idx = above_idx + below_idx
        self.g[yidx[idx]] = 0

        if len(idx) > 0:
            self.factorize = True

    def anti_windup(self, xidx, xmin, xmax):
        """
        Anti-windup limiter for state variables.

        Resets the limited variables and differential equations.

        :param xidx: state variable indices
        :param xmin: lower limit
        :param xmax: upper limit

        :type xidx: matrix, list
        :type xmin: matrix, float, int, list
        :type xmax: matrix, float, int, list

        """
        xidx = matrix(xidx)
        xval = self.x[xidx]
        fval = self.f[xidx]

        if isinstance(xmin, (float, int, list)):
            xmin = matrix(xmin, xidx.size, 'd')

        if isinstance(xmax, (float, int, list)):
            xmax = matrix(xmax, xidx.size, 'd')

        x_above = ageb(xval, xmax)
        f_above = ageb(fval, 0.0)

        x_below = aleb(xval, xmin)
        f_below = aleb(fval, 0.0)

        above = aandb(x_above, f_above)
        above_idx = index(above, 1.0)
        if len(above_idx) > 0:
            above_xidx = xidx[above_idx]
            self.x[above_xidx] = xmax[above_idx]
            self.zxmax[above_xidx] = 0

        below = aandb(x_below, f_below)
        below_idx = index(below, 1.0)
        if len(below_idx) > 0:
            below_xidx = xidx[below_idx]
            self.x[below_xidx] = xmin[below_idx]
            self.zxmin[below_xidx] = 0

        idx = list(above_idx) + list(below_idx)
        if len(idx) > 0:
            self.f[xidx[idx]] = 0
            self.ac_reset = True

    def reset_Ac(self):
        """
        Reset ``dae.Ac`` sparse matrix for disabled equations
        due to hard_limit and anti_windup limiters.

        :return: None
        """
        if self.ac_reset is False and not self.rebuild:
            return

        mn = self.m + self.n

        x = index(aandb(self.zxmin, self.zxmax), 0.)
        y = [i + self.n for i in index(aandb(self.zymin, self.zymax), 0.)]
        xy = list(x) + y

        eye = spdiag([1.0] * mn)
        H = spmatrix(1.0, xy, xy, (mn, mn), 'd')

        # Modifying ``eye`` is more efficient than ``eye = eye - H``.
        # CVXOPT modifies eye in place because all the accessed elements exist.

        for idx in xy:
            eye[idx, idx] = 0

        if len(xy) > 0:
            self.Ac = eye * (self.Ac * eye) - H
            self.q[x] = 0

        self.ac_reset = False
        self.factorize = True

    def get_size(self, m):
        """
        Return the 2-D size of a Jacobian matrix in tuple
        """
        nrow, ncol = 0, 0
        if m[0] == 'F':
            nrow = self.n
        elif m[0] == 'G':
            nrow = self.m

        if m[1] == 'x':
            ncol = self.n
        elif m[1] == 'y':
            ncol = self.m

        return nrow, ncol

    def add_jac(self, m, val, row, col):
        """Add tuples (val, row, col) to the Jacobian matrix ``m``

        Implemented in numpy.arrays for temporary storage.
        """
        assert m in ('Fx', 'Fy', 'Gx', 'Gy', 'Fx0', 'Fy0', 'Gx0', 'Gy0'), \
            'Wrong Jacobian matrix name <{0}>'.format(m)

        if isinstance(val, (int, float)):
            val = val * ones(len(row), 1)

        self._temp[m]['I'] = matrix([self._temp[m]['I'], matrix(row)])
        self._temp[m]['J'] = matrix([self._temp[m]['J'], matrix(col)])
        self._temp[m]['V'] = matrix([self._temp[m]['V'], matrix(val)])

    def temp_to_spmatrix(self, ty):
        """
        Convert Jacobian tuples to matrices

        :param ty: name of the matrices to convert in ``('jac0','jac')``

        :return: None
        """
        assert ty in ('jac0', 'jac')

        jac0s = ['Fx0', 'Fy0', 'Gx0', 'Gy0']
        jacs = ['Fx', 'Fy', 'Gx', 'Gy']

        if ty == 'jac0':
            todo = jac0s
        elif ty == 'jac':
            todo = jacs

        for m in todo:
            self.__dict__[m] = spmatrix(self._temp[m]['V'],
                                        self._temp[m]['I'], self._temp[m]['J'],
                                        self.get_size(m), 'd')
            if ty == 'jac':
                self.__dict__[m] += self.__dict__[m + '0']

        self.apply_set(ty)

    def set_jac(self, m, val, row, col):
        """
        Set the values at (row, col) to val in Jacobian m

        :param m: Jacobian name
        :param val: values to set
        :param row: row indices
        :param col: col indices
        :return: None
        """
        assert m in ('Fx', 'Fy', 'Gx', 'Gy', 'Fx0', 'Fy0', 'Gx0', 'Gy0'), \
            'Wrong Jacobian matrix name <{0}>'.format(m)

        if isinstance(val,
                      (int, float)) and isinstance(row,
                                                   (np.ndarray, matrix, list)):
            val = val * ones(len(row), 1)

        self._set[m]['I'] = matrix([self._set[m]['I'], matrix(row)])
        self._set[m]['J'] = matrix([self._set[m]['J'], matrix(col)])
        self._set[m]['V'] = matrix([self._set[m]['V'], matrix(val)])

    def apply_set(self, ty):
        """
        Apply Jacobian set values to matrices

        :param ty: Jacobian type in ``('jac0', 'jac')``
        :return:
        """
        assert ty in ('jac0', 'jac')

        if ty == 'jac0':
            todo = ['Fx0', 'Fy0', 'Gx0', 'Gy0']
        else:
            todo = ['Fx', 'Fy', 'Gx', 'Gy']

        for m in todo:
            for idx in range(len(self._set[m]['I'])):
                i = self._set[m]['I'][idx]
                j = self._set[m]['J'][idx]
                v = self._set[m]['V'][idx]
                self.__dict__[m][i, j] = v

    def show(self, eq, value=None):
        """Show equation or variable array along with the names"""
        if eq in ['f', 'x']:
            key = 'unamex'
        elif eq in ['g', 'y']:
            key = 'unamey'

        if value:
            value = list(value)
        else:
            value = list(self.__dict__[eq])

        out = ''
        for name, val, idx in zip(self.system.varname.__dict__[key], value,
                                  range(len(value))):
            out += '{:20s} [{:>12.4f}] {:g}\n'.format(name, val, idx)
        return out

    def find_val(self, eq, val):
        """Return the name of the equation having the given value"""
        if eq not in ('f', 'g', 'q'):
            return
        elif eq in ('f', 'q'):
            key = 'unamex'
        elif eq == 'g':
            key = 'unamey'
        idx = 0
        for m, n in zip(self.system.varname.__dict__[key], self.__dict__[eq]):
            if n == val:
                return m, idx
            idx += 1
        return

    def reset_small(self, eq):
        """Reset numbers smaller than 1e-12 in f and g equations"""
        assert eq in ('f', 'g')
        for idx, var in enumerate(self.__dict__[eq]):
            if abs(var) <= 1e-12:
                self.__dict__[eq][idx] = 0

    def reset_small_f(self):
        pass

    def reset_small_g(self):
        pass

    def check_diag(self, jac, name):
        """
        Check matrix ``jac`` for diagonal elements that equals 0
        """
        system = self.system
        pos = []
        names = []
        pairs = ''
        size = jac.size
        diag = jac[0:size[0] ** 2:size[0] + 1]

        for idx in range(size[0]):
            if abs(diag[idx]) <= 1e-8:
                pos.append(idx)

        for idx in pos:
            names.append(system.varname.__dict__[name][idx])

        if len(names) > 0:
            for i, j in zip(pos, names):
                pairs += '{0}: {1}\n'.format(i, j)
            logger.debug('Jacobian diagonal check:')
            logger.debug(pairs)

    # def add_jac(self, m, val, row, col):
    #     """Add values (val, row, col) to Jacobian m
    #
    #     This implementation construct the Jacobians incrementally.
    #       It is less efficient than storing (I,J,V) and create altogether.
    #     """
    #     if m not in ['Fx', 'Fy', 'Gx', 'Gy', 'Fx0', 'Fy0', 'Gx0', 'Gy0']:
    #         raise NameError('Wrong Jacobian matrix name <{0}>'.format(m))
    #
    #     size = self.__dict__[m].size
    #     self.__dict__[m] += spmatrix(val, row, col, size, 'd')
    #
    # def set_jac(self, m, val, row, col):
    #     """Add values (val, row, col) to Jacobian m
    #
    #     This implementation is very inefficient as it involves two
    #       __getitem__ and two __setitem__ ops.
    #     """
    #     if m not in ['Fx', 'Fy', 'Gx', 'Gy', 'Fx0', 'Fy0', 'Gx0', 'Gy0']:
    #         raise NameError('Wrong Jacobian matrix name <{0}>'.format(m))
    #
    #     old_val = []
    #     if type(row) is int:
    #         row = [row]
    #     if type(col) is int:
    #         col = [col]
    #     if type(row) is range:
    #         row = list(row)
    #     if type(col) is range:
    #         col = list(col)
    #     for i, j in zip(row, col):
    #         old_val.append(self.system.dae.__dict__[m][i, j])
    #     size = self.__dict__[m].size
    #     self.system.dae.__dict__[m] -= spmatrix(old_val, row, col, size, 'd')
    #     self.system.dae.__dict__[m] += spmatrix(val, row, col, size, 'd')
    #
    #
    # def hard_limit(self, yidx, ymin, ymax, min_set=None, max_set=None):
    #     """Limit algebraic variables and set the Jacobians.
    #
    #     This method is slower than the current version because,
    #       comparing with modifying matrix elements directly,
    #     matrix multiplication and re-assignment involves more ops.
    #     """
    #     yidx = matrix(yidx)
    #
    #     if not min_set:
    #         min_set = ymin
    #     if not max_set:
    #         max_set = ymax
    #
    #     if isinstance(min_set, (int, float)):
    #         min_set = matrix(min_set, (len(yidx), 1), 'd')
    #     if isinstance(max_set, (int, float)):
    #         max_set = matrix(max_set, (len(yidx), 1), 'd')
    #
    #     yvals = self.y[yidx]
    #
    #     above = ageb(yvals, ymax)
    #     below = aleb(yvals, ymin)
    #
    #     self.y[yidx] = mul(self.y[yidx], nota(above)) + mul(max_set, above)
    #     self.y[yidx] = mul(self.y[yidx], nota(below)) + mul(min_set, below)
    #
    #     self.zymax[yidx] = mul(self.zymax[yidx], nota(above))
    #     self.zymin[yidx] = mul(self.zymin[yidx], nota(below))
    #
    #     idx = aorb(above, below)
    #     self.g[yidx] = mul(self.g[yidx], nota(idx))
    #
    #     if sum(idx) > 0:
    #         self.ac_reset = True
    #
    #
    # def anti_windup(self, xidx, xmin, xmax):
    #     """State variable anti-windup limiter.
    #
    #     This version is slow for the same reason as hard_limit().
    #     """
    #     xidx = matrix(xidx)
    #     xval = self.x[xidx]
    #
    #     if type(xmin) in (float, int):
    #         xmin = matrix(xmin, xidx.size)
    #     if type(xmax) in (float, int):
    #         xmax = matrix(xmax, xidx.size)
    #
    #     x_above = ageb(xval, xmax)
    #     x_below = aleb(xval, xmin)
    #
    #     f_above = ageb(xval, 0.0)
    #     f_below = aleb(xval, 0.0)
    #
    #     above = aandb(x_above, f_above)
    #     below = aandb(x_below, f_below)
    #
    #     self.x[xidx] = mul(self.x[xidx], nota(above)) + mul(xmax, above)
    #     self.zxmax[xidx] = mul(self.zxmax[xidx], nota(above))
    #
    #     self.x[xidx] = mul(self.x[xidx], nota(below)) + mul(xmin, below)
    #     self.zxmin[xidx] = mul(self.zxmin[xidx], nota(below))
    #
    #     idx = aorb(above, below)
    #     self.f[xidx] = mul(self.f[xidx], nota(idx))
    #
    #     if sum(idx) > 0:
    #         self.factorize = True
