from cvxopt import matrix, spmatrix, sparse, spdiag


class DAE(object):
    """Class for numerical Differential Algebraic Equations (DAE)"""
    def __init__(self, system):
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
        self.x = matrix(0.0, (self.n, 1), 'd')

    def init_y(self):
        self.y = matrix(0.0, (self.m, 1), 'd')

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
            yfill = matrix(0.0, (yext, 1), 'd')
            self.y = matrix([self.y, yfill], (self.m, 1), 'd')
            self.g = matrix([self.g, yfill], (self.m, 1), 'd')
        if xext:
            xfill = matrix(0.0, (xext, 1), 'd')
            self.x = matrix([self.x, xfill], (self.n, 1), 'd')
            self.f = matrix([self.f, xfill], (self.n, 1), 'd')

    def algeb_windup(self, idx):
        H = spmatrix(1.0, idx, idx, (self.m, self.m))
        I = spdiag([1.0] * self.m) - H
        self.Gy = I * (self.Gy * I) + H
