from cvxopt import matrix, spmatrix, sparse, spdiag
from ..utils.math import *


class DAE(object):
    """Class for numerical Differential Algebraic Equations (DAE)"""
    def __init__(self, system):
        self.system = system
        self._data = {'x': [],
                      'y': [],
                      'f': [],
                      'g': [],
                      'Fx': [],
                      'Fy': [],
                      'Gx': [],
                      'Gy': [],
                      'Fx0': [],
                      'Fy0': [],
                      'Gx0': [],
                      'Gy0': [],
                      'Ac': [],
                      'tn': [],
                      'xu': [],
                      'yu': [],
                      'xz': [],
                      'yz': [],
                      }

        self._scalars = {'m': 0,
                         'n': 0,
                         'lamda': 0,
                         'npf': 0,
                         'kg': 0,
                         't': -1,
                         'factorize': True,
                         }

        self.__dict__.update(self._data)
        self.__dict__.update(self._scalars)

    def init_xy(self):
        self.init_x()
        self.init_y()

    def init_x(self):
        self.x = zeros(self.n, 1)
        self.xz = zeros(self.n, 1)
        self.xu = ones(self.n, 1)


    def init_y(self):
        self.y = zeros(self.m, 1)
        self.yz = zeros(self.m, 1)
        self.yu = ones(self.m, 1)

    def init_fg(self):
        self.init_f()
        self.init_g()

    def init_f(self):
        self.f = matrix(0.0, (self.n, 1), 'd')

    def init_g(self):
        self.g = matrix(0.0, (self.m, 1), 'd')

    def setup_Gy(self):
        self.Gy = sparse(self.Gy0)

    def setup_Fx(self):
        self.Fx = sparse(self.Fx0)
        self.Fy = sparse(self.Fy0)
        self.Gx = sparse(self.Gx0)

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

    def init_Fx0(self):
        self.Gx0 = spmatrix([], [], [], (self.m, self.n), 'd')
        self.Fy0 = spmatrix([], [], [], (self.n, self.m), 'd')
        self.Fx0 = spmatrix([], [], [], (self.n, self.n), 'd')

    def init_jac0(self):
        self.init_Gy0()
        self.init_Fx0()

    def init1(self):
        self.resize()
        self.init_jac0()

    def resize(self):
        """Resize DAE and and extend for init1 variables"""
        yext = self.m - len(self.y)
        xext = self.n - len(self.x)
        if yext:
            yzeros = matrix(0.0, (yext, 1), 'd')
            yones = matrix(1.0, (yext, 1), 'd')
            self.y = matrix([self.y, yzeros], (self.m, 1), 'd')
            self.g = matrix([self.g, yzeros], (self.m, 1), 'd')
            self.yu = matrix([self.yu, yones], (self.m, 1), 'd')
            self.yz = matrix([self.yz, yzeros], (self.m, 1), 'd')
        if xext:
            xzeros = matrix(0.0, (xext, 1), 'd')
            xones = matrix(1.0, (xext, 1), 'd')
            self.x = matrix([self.x, xzeros], (self.n, 1), 'd')
            self.f = matrix([self.f, xzeros], (self.n, 1), 'd')
            self.xu = matrix([self.xu, xones], (self.n, 1), 'd')
            self.xz = matrix([self.xz, xzeros], (self.n, 1), 'd')

    def algeb_windup(self, idx):
        """Reset Jacobian elements related to windup algebs"""
        H = spmatrix(1.0, idx, idx, (self.m, self.m))
        I = spdiag([1.0] * self.m) - H
        self.Gy = I * (self.Gy * I) + H

    def ylimiter(self, yidx, ymin, ymax):
        """Limit algebraic variables and set the Jacobians"""
        yidx = matrix(yidx)
        above = agtb(self.y[yidx], ymax)
        above_idx = findeq(above, 1)
        self.y[yidx[above_idx]] = ymax[above_idx]

        below = altb(self.y[yidx], ymin)
        below_idx = findeq(below, 1)
        self.y[yidx[below_idx]] = ymin[below_idx]

        idx = list(above_idx) + list(below_idx)
        self.g[yidx[idx]] = 0
        self.yz[yidx[idx]] = 1

        if len(idx) > 0:
            self.factorize = True

    def xwindup(self, xidx, xmin, xmax):
        """State variable windup limiter"""
        xidx = matrix(xidx)
        above = agtb(self.x[xidx], xmax)
        above_idx = findeq(above, 1.0)
        self.x[xidx[above_idx]] = xmax[above_idx]

        below = altb(self.x[xidx], xmin)
        below_idx = findeq(below, 1.0)
        self.x[xidx[below_idx]] = xmin[below_idx]

        idx = list(above_idx) + list(below_idx)
        self.f[xidx[idx]] = 0
        self.xz[xidx[idx]] = 1

        if len(idx) > 0:
            self.factorize = True

    def set_Ac(self):
        """Reset Jacobian elements for limited algebraic variables"""
        # Todo: replace algeb_windup()
        idx = matrix(findeq(aandb(self.yu, self.yz), 1.0))
        H = spmatrix(1.0, idx, idx, (self.m, self.m))
        I = spdiag([1.0] * self.m) - H
        self.Gy = I * (self.Gy * I) + H
        self.Fy = self.Fy * I
        self.Gx = I * self.Gx

    def add_jac(self, m, val, row, col):
        """Add values (val, row, col) to Jacobian m"""
        if m not in ['Fx', 'Fy', 'Gx', 'Gy', 'Fx0', 'Fy0', 'Gx0', 'Gy0']:
            raise NameError('Wrong Jacobian matrix name <{0}>'.format(m))

        size = self.system.DAE.__dict__[m].size
        self.system.DAE.__dict__[m] += spmatrix(val, row, col, size, 'd')

    def set_jac(self, m, val, row, col):
        """Add values (val, row, col) to Jacobian m """
        if m not in ['Fx', 'Fy', 'Gx', 'Gy', 'Fx0', 'Fy0', 'Gx0', 'Gy0']:
            raise NameError('Wrong Jacobian matrix name <{0}>'.format(m))

        size = self.system.DAE.__dict__[m].size
        oldval = []
        if type(row) is int:
            row = [row]
        if type(col) is int:
            col = [col]
        if type(row) is range:
            row = list(row)
        if type(col) is range:
            col = list(col)
        for i, j in zip(row, col):
            oldval.append(self.system.DAE.__dict__[m][i, j])
        self.system.DAE.__dict__[m] -= spmatrix(oldval, row, col, size, 'd')
        self.system.DAE.__dict__[m] += spmatrix(val, row, col, size, 'd')

    def show(self, eq):
        """Show equation or variable array along with the names"""
        str = ''
        if eq in ['f', 'x']:
            key = 'unamex'
        elif eq in ['g', 'y']:
            key = 'unamey'

        value = list(self.__dict__[eq])
        for name, val in zip(self.system.VarName.__dict__[key], value):
            str += '{:12s} [{:>12.4f}]\n'.format(name, val)
        return str