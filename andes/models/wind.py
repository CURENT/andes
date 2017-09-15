"""Wind power classes"""
from math import floor

from numpy.random import weibull
from numpy import arange

from cvxopt import matrix, mul, spmatrix, div, sin, cos
from .base import ModelBase
from ..utils.math import *
from ..consts import *


class WindBase(ModelBase):
    """Base class for wind time series"""
    def __init__(self, system, name):
        super(WindBase, self).__init__(system, name)
        self._group = 'Wind'
        self.remove_param('Sn')
        self.remove_param('Vn')
        self._data.update({'T': 1,
                           'Vwn': 15,
                           'dt': 0.1,
                           'rho': 1.225,
                           })
        self._descr.update({'T': 'Filter time constant',
                            'Vwn': 'Wind speed base',
                            'dt': 'Sampling time step',
                            'rho': 'Air density',
                            })
        self._units.update({'T': 's',
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
        self.calls.update({'init1': True, 'gcall': True,
                           'fcall': True, 'jac0': True,
                           })

    def setup(self):
        super(WindBase, self).setup()
        # self.vwa = ones(self.n, 1)  # todo: remove this after wind turbine init
        # self.system.DAE.x[self.vw] = ones(self.n, 1)

    def servcall(self, dae):
        self.iT = div(1, self.T)
        self.t0 = self.system.TDS.t0
        self.tf = self.system.TDS.tf

    def init1(self, dae):
        self.servcall(dae)
        self.time = [0] * self.n
        self.speed = [0] * self.n
        for i in range(self.n):
            self.time[i] = list(arange(self.t0, self.tf + self.dt[i], self.dt[i]))
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
            if t in self.time[i]:
                idx = self.time[i].index(t)
                ws[i] = self.speed[i][idx]
            else:
                loca = floor(t/self.dt[i])
                t1 = self.time[i][loca]
                s1 = self.speed[i][loca]
                s2 = self.speed[i][loca + 1]
                ws[i] = s1 + (t - t1) * (s2 - s1)/self.dt[i]
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
        self._descr.update({'c': 'Scale factor', 's': 'Shape factor',})
        self._zeros.extend(['c', 's'])
        self._params.extend(['c', 's'])
        self._inst_meta()

    def generate(self, dae):
        for i in range(self.n):
            npoint = len(self.time[i])
            if self.c[i] <= 0.0:
                self.c[i] = 5.0
            if self.s[i] <= 0.0:
                self.s[i] = 2.0
            sample = self.c[i] * weibull(self.s[i], npoint)
            sample[0] = dae.x[self.vw[i]]
            sample_avg = sum(sample[1:]) / (npoint-1)
            k = sample[0] / sample_avg

            sample, sample_avg = matrix(sample), matrix(sample_avg)

            sample[1:] *= k
            self.speed[i] = sample


class ConstWind(WindBase):
    """Constant wind power class"""
    def __init__(self, system, name):
        super(ConstWind, self).__init__(system, name)
        self._name = 'ConstWind'
        self._inst_meta()

    def generate(self, dae):
        for i in range(self.n):
            sample = [dae.x[self.vw[i]]] * len(self.time[i])
            self.speed[i] = list(sample)

