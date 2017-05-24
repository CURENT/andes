from cvxopt import matrix, spmatrix, sparse, spdiag
from ..utils.math import *


class DAE(object):
    """Class for numerical Differential Algebraic Equations (DAE)"""
    def __init__(self, system):
        self.system = system
        self._data = dict(x=[], y=[], f=[], g=[], Fx=[], Fy=[], Gx=[], Gy=[], Fx0=[], Fy0=[], Gx0=[], Gy0=[], Ac=[],
                          tn=[])

        self._scalars = dict(m=0, n=0, lamda=0, npf=0, kg=0, t=-1, factorize=True, rebuild=True)

        self._flags = dict(zxmax=[], zxmin=[], zymax=[], zymin=[], ux=[], uy=[])

        self.__dict__.update(self._data)
        self.__dict__.update(self._scalars)
        self.__dict__.update(self._flags)

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
            yzeros = zeros(yext, 1)
            yones = ones(yext, 1)
            self.y = matrix([self.y, yzeros], (self.m, 1), 'd')
            self.g = matrix([self.g, yzeros], (self.m, 1), 'd')
            self.uy = matrix([self.uy, yones], (self.m, 1), 'd')
            self.zymin = matrix([self.zymin, yones], (self.m, 1), 'd')
            self.zymax = matrix([self.zymax, yones], (self.m, 1), 'd')
        if xext:
            xzeros = zeros(xext, 1)
            xones = ones(xext, 1)
            self.x = matrix([self.x, xzeros], (self.n, 1), 'd')
            self.f = matrix([self.f, xzeros], (self.n, 1), 'd')
            self.ux = matrix([self.ux, xones], (self.n, 1), 'd')
            self.zxmin = matrix([self.zxmin, xones], (self.n, 1), 'd')
            self.zxmax = matrix([self.zxmax, xones], (self.n, 1), 'd')

    # def algeb_windup(self, idx):
    #     """Reset Jacobian elements related to windup algebs"""
    #     H = spmatrix(1.0, idx, idx, (self.m, self.m))
    #     I = spdiag([1.0] * self.m) - H
    #     self.Gy = I * (self.Gy * I) + H

    def hard_limit(self, yidx, ymin, ymax):
        """Limit algebraic variables and set the Jacobians"""
        self.zymax = ones(self.m, 1)
        self.zymin = ones(self.m, 1)

        yidx = matrix(yidx)

        above = ageb(self.y[yidx], ymax)
        above_idx = findeq(above, 1.0)
        self.y[yidx[above_idx]] = ymax[above_idx]
        self.zymax[yidx[above_idx]] = 0

        below = aleb(self.y[yidx], ymin)
        below_idx = findeq(below, 1.0)
        self.y[yidx[below_idx]] = ymin[below_idx]
        self.zymin[yidx[below_idx]] = 0

        idx = list(above_idx) + list(below_idx)
        self.g[yidx[idx]] = 0

        if len(idx) > 0:
            self.factorize = True

    def anti_windup(self, xidx, xmin, xmax):
        """State variable anti-windup limiter"""
        self.zxmax = ones(self.n, 1)
        self.zxmin = ones(self.n, 1)

        xidx = matrix(xidx)

        x_above = ageb(self.x[xidx], xmax)
        f_above = ageb(self.f[xidx], 0.0)
        above = aandb(x_above, f_above)
        above_idx = findeq(above, 1.0)
        self.x[xidx[above_idx]] = xmax[above_idx]
        self.zxmax[xidx[above_idx]] = 0

        x_below = aleb(self.x[xidx], xmin)
        f_below = aleb(self.f[xidx], 0.0)
        below = aandb(x_below, f_below)
        below_idx = findeq(below, 1.0)
        self.x[xidx[below_idx]] = xmin[below_idx]
        self.zxmin[xidx[below_idx]] = 0

        idx = list(above_idx) + list(below_idx)
        self.f[xidx[idx]] = 0

        if len(idx) > 0:
            self.factorize = True

    def reset_Ac(self):
        if sum(self.zxmin) == self.n and sum(self.zxmax) == self.n \
                and sum(self.zymin) == self.n and sum(self.zymax) == self.n:
            return
        idx1 = findeq(aandb(self.zxmin, self.zxmax), 0.0)
        idx2 = findeq(aandb(self.zymin, self.zymax), 0.0)
        idx2 = [i + self.n for i in idx2]

        idx = matrix(idx1 + idx2)
        H = spmatrix(1.0, idx, idx, (self.m + self.n, self.m + self.n))
        I = spdiag([1.0] * (self.m + self.n)) - H
        self.Ac = I * (self.Ac * I) - H
        self.q[idx1] = 0

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
        old_val = []
        if type(row) is int:
            row = [row]
        if type(col) is int:
            col = [col]
        if type(row) is range:
            row = list(row)
        if type(col) is range:
            col = list(col)
        for i, j in zip(row, col):
            old_val.append(self.system.DAE.__dict__[m][i, j])
        self.system.DAE.__dict__[m] -= spmatrix(old_val, row, col, size, 'd')
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