"""Wind power classes"""
from numpy.random import weibull, uniform  # NOQA
from numpy import arange, log

from cvxopt import matrix, mul, spmatrix, div  # NOQA
from .base import ModelBase

from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA


class WindBase(ModelBase):
    """Base class for wind time series"""

    def __init__(self, system, name):
        super(WindBase, self).__init__(system, name)
        self._group = 'Wind'
        self.param_remove('Sn')
        self.param_remove('Vn')
        self._data.update({
            'T': 1,
            'Vwn': 13,
            'dt': 0.1,
            'rho': 1.225,
        })
        self._descr.update({
            'T': 'Filter time constant',
            'Vwn': 'Wind speed base',
            'dt': 'Sampling time step',
            'rho': 'Air density',
        })
        self._units.update({
            'T': 's',
            'Vwn': 'm/s',
            'rho': 'kg/m^3',
            'dt': 's',
        })
        self._params.extend(['T', 'Vwn', 'dt', 'rho'])
        self._states.extend(['vw'])
        self._fnamex.extend(['v_w'])
        self._algebs.extend(['ws'])
        self._fnamey.extend(['w_s'])
        self._zeros.extend(['T', 'dt'])
        self.time, self.speed, self.t0, self.tf = list(), list(), 0, 0
        self.calls.update({
            'init1': True,
            'gcall': True,
            'fcall': True,
            'jac0': True,
        })

    def setup(self):
        super(WindBase, self).setup()
        # self.vwa = ones(self.n, 1)
        #   todo: remove this after wind turbine _init
        # self.system.dae.x[self.vw] = ones(self.n, 1)

    def servcall(self, dae):
        self.iT = div(1, self.T)
        self.t0 = self.system.tds.config.t0
        self.tf = self.system.tds.config.tf

    def init1(self, dae):
        self.servcall(dae)
        self.time = [0] * self.n
        self.speed = [0] * self.n
        for i in range(self.n):
            self.time[i] = list(
                arange(self.t0, self.tf + self.dt[i], self.dt[i]))
            self.speed[i] = [0] * len(self.time[i])

        self.generate(dae)

        dae.y[self.ws] = dae.x[self.vw]
        self.speed[:][0] = dae.y[self.ws]

    def generate(self, dae):
        """Generate the wind speed time and data points"""
        pass

    def windspeed(self, t):
        """Return the wind speed list at time `t`"""
        ws = [0] * self.n

        for i in range(self.n):
            q = int(t / self.dt[i])
            q_prev = 0 if q == 0 else q - 1

            r = t % self.dt[i]
            r = 0 if abs(r) < 1e-6 else r

            if r == 0:
                ws[i] = self.speed[i][q]
            else:
                t1 = self.time[i][q_prev]
                s1 = self.speed[i][q_prev]
                s2 = self.speed[i][q]
                ws[i] = s1 + (t - t1) * (s2 - s1) / self.dt[i]

        return matrix(ws)

    def gcall(self, dae):
        dae.g[self.ws] = self.windspeed(dae.t) - dae.y[self.ws]

    def fcall(self, dae):
        dae.f[self.vw] = mul(dae.y[self.ws] - dae.x[self.vw], self.iT)

    def jac0(self, dae):
        dae.add_jac(Gy0, -1, self.ws, self.ws)
        dae.add_jac(Fx0, -self.iT, self.vw, self.vw)
        dae.add_jac(Fy0, self.iT, self.vw, self.ws)


class Weibull(WindBase):
    """Weibull distribution wind speed class"""

    def __init__(self, system, name):
        super(Weibull, self).__init__(system, name)
        self._name = 'Weibull'
        self._data.update({'c': 5.0, 's': 2.0})
        self._descr.update({
            'c': 'Scale factor',
            's': 'Shape factor',
        })
        self._zeros.extend(['c', 's'])
        self._params.extend(['c', 's'])
        self._init()

    def generate(self, dae):
        for i in range(self.n):
            npoint = len(self.time[i])
            if self.c[i] <= 0.0:
                self.c[i] = 5.0
            if self.s[i] <= 0.0:
                self.s[i] = 2.0
            sample = (-log(uniform(0, 1,
                                   (npoint, 1))) / self.c[i])**(1 / self.s[i])

            avg = sum(sample) / npoint
            sample[0] = dae.x[self.vw[i]]
            sample[1:] = abs(sample[1:] - avg + 1) * sample[0]

            self.speed[i] = matrix(sample)


class ConstWind(WindBase):
    """Constant wind power class"""

    def __init__(self, system, name):
        super(ConstWind, self).__init__(system, name)
        self._name = 'ConstWind'
        self._init()

    def generate(self, dae):
        for i in range(self.n):
            sample = [dae.x[self.vw[i]]] * len(self.time[i])
            self.speed[i] = list(sample)
