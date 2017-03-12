from cvxopt import matrix, spdiag, mul, div, log, exp, spmatrix
from .base import ModelBase
from ..consts import *
from ..utils.math import *


class PQ(ModelBase):
    """Static PQ load class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'PQ'
        self._group = 'StaticLoad'
        self._category = 'Load'
        self._data.update({'bus': None,
                           'p': 0,
                           'q': 0,
                           'owner': 0,
                           'vmax': 1.1,
                           'vmin': 0.9,
                           })
        self._units.update({'bus': 'na',
                            'p': 'pu',
                            'owner': 'na',
                            'vmax': 'pu',
                            'vmin': 'pu',
                            })
        self._params.extend(['p', 'q', 'vmax', 'vmin'])
        self._descr.update({'bus': 'bus number',
                            'p': 'constant p value',
                            'q': 'constant q value',
                            'owner': 'owner code',
                            'vmax': 'max voltage before switching to Z',
                            'vmin': 'min voltage before switching to Z'
                            })
        self._ac = {'bus': ['a', 'v']}
        self._powers = ['p', 'q']
        self._service = ['p0', 'q0', 'v0', 'below', 'above', 'normal']  # p0 and q0 are used during computation
        self.calls.update({'gcall': True, 'gycall': True,
                           'init0': True, 'init1': True,
                           'pflow': True, 'shunt': True,
                           })
        self._inst_meta()

    def init0(self, dae):
        """Set initial p and q for power flow"""
        self.p0 = matrix(self.p, (self.n, 1), 'd')
        self.q0 = matrix(self.q, (self.n, 1), 'd')

    def init1(self, dae):
        """Set initial voltage for time domain simulation"""
        self.v0 = matrix(dae.y[self.v])

    def gcall(self, dae):
        k = ones(self.n, 1)

        if self.system.Settings.forcez:
            if self.v0:
                k = div(dae.y[self.v] ** 2, self.v0 ** 2)
        elif self.system.Settings.forcepq:
            pass
        else:
            k = zeros(self.n, 1)

            self.below = altb(dae.y[self.v], self.vmin)
            k += mul(self.below, div(dae.y[self.v] ** 2, self.vmin ** 2))

            self.above = agtb(dae.y[self.v], self.vmax)
            k += mul(self.above, div(dae.y[self.v] ** 2, self.vmax ** 2))

            normal = nota(aorb(self.below, self.above))
            k += mul(normal, ones(self.n, 1))

        k = mul(self.u, k)
        self.p0 = mul(k, self.p)
        self.q0 = mul(k, self.q)

        dae.g += spmatrix(self.p0, self.a, [0] * self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(self.q0, self.v, [0] * self.n, (dae.m, 1), 'd')

    def gycall(self, dae):
        k = zeros(self.n, 1)
        if self.system.Settings.forcepq:
            return
        elif self.system.Settings.forcez:
            if self.v0:
                k = div(2 * dae.y[self.v], self.v0 ** 2)
        else:
            k += mul(self.below, div(2 * dae.y[self.v], self.vmin ** 2))
            k += mul(self.above, div(2 * dae.y[self.v], self.vmax ** 2))
        k = mul(self.u, k)

        dae.add_jac(Gy, mul(self.p, k), self.a, self.v)
        dae.add_jac(Gy, mul(self.q, k), self.v, self.v)
