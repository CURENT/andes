from cvxopt import matrix, sparse, spmatrix
from cvxopt import mul, div, log, sin, cos
from .base import ModelBase
from ..consts import *
from ..utils.math import *


class BusFreq(ModelBase):
    """Bus frequency estimation based on angle derivative"""
    def __init__(self, system, name):
        super(BusFreq, self).__init__(system, name)
        self._group = 'Measurement'
        self._name = 'BusFreq'
        self._data.update({'bus': None,
                           'tf': 0.1,
                           'tw': 1.0,
                           })
        self._params.extend(['tf', 'tw'])
        self._descr.update({'bus': 'bus idx',
                            'tf': 'low-pass filter time constant',
                            'tw': 'washout filter time constant',
                            })
        self._units.update({'bus': 'na',
                            'tf': 'sec',
                            'tw': 'sec',
                            })
        self._mandatory.extend(['bus'])
        # self._algebs.extend(['dw'])
        # self._fnamey.extend(['\\delta\\omega'])
        self._states.extend(['xt', 'w'])
        self._fnamex.extend(['x_\\theta', '\\omega'])
        self.calls.update({'init1': True, 'fcall': True,
                           'jac0': True,
                           })
        self._service.extend(['itf', 'itw', 'iwn', 'a0', 'dw'])
        self._inst_meta()

    def init1(self, dae):
        self.copy_param(model='Bus', src='a', dest='a', fkey=self.bus)
        self.itf = div(self.u, self.tf)
        self.itw = div(self.u, self.tw)
        self.iwn = div(self.u, self.system.Settings.wb)
        self.a0 = dae.y[self.a]

        dae.x[self.xt] = zeros(self.n, 1)
        dae.x[self.w] = ones(self.n, 1)
        # dae.y[self.dw] = zeros(self.n, 1)

    def fcall(self, dae):
        self.dw = -dae.x[self.xt] + mul(self.iwn, self.itf, dae.y[self.a] - self.a0)
        dae.f[self.xt] = mul(self.itf, mul(self.iwn, self.itf, dae.y[self.a] - self.a0) - dae.x[self.xt])
        dae.f[self.w] = mul(self.dw + 1 - dae.x[self.w], self.itw)

    def jac0(self, dae):
        dae.add_jac(Fy0, mul(self.itf **2, self.iwn), self.xt, self.a)
        dae.add_jac(Fx0, -self.itf, self.xt, self.xt)

        dae.add_jac(Fx0, -self.itw, self.w, self.w)

        dae.add_jac(Fx0, -self.itw, self.w, self.xt)
        dae.add_jac(Fy0, mul(self.itw, self.iwn, self.itf), self.w, self.a)

