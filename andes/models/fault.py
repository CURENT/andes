from cvxopt import matrix, mul, spmatrix, div
from .base import ModelBase
from ..consts import *
from ..utils.math import *


class Fault(ModelBase):
    """3-phase to ground fault class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'Fault'
        self._name = 'Fault'
        self._data.update({'bus': None,
                           'fn': 60.0,
                           'tf': None,
                           'tc': None,
                           'rf': 0,
                           'xf': 1e-6,
                           })
        self._params.extend(['bus',
                             'fn',
                             'tf',
                             'tc',
                             'rf',
                             'xf',
                             ])
        self._descr.update({'bus': 'bus number of fault',
                            'fn': 'rated frequency',
                            'tf': 'fault occurrence time',
                            'tc': 'fault clearing time',
                            'rf': 'fault resistance',
                            'xf': 'fault impedance',
                            })
        self._ac = {'bus': ['a', 'v']}
        self._z.extend(['rf', 'xf'])
        self._mandatory.extend(['bus', 'tf'])
        self._service = ['gf', 'bf', 'time', 'volt0', 'angle0']
        self.calls.update({'gcall': True,
                           'gycall': True})
        self._inst_meta()
        self.active = 0

    def setup(self):
        super().setup()
        self.xf += 1e-8
        Y = div(1, self.rf + 1j*self.xf)
        self.gf = Y.real()
        self.bf = Y.imag()
        self.u = zeros(self.n, 1)

    def get_times(self):
        if not self.n:
            return []
        t = matrix(list(self.tf) + list(self.tc))
        return list(t) + list(t - 1e-6)

    def istime(self, t):
        return t in self.get_times()

    def check_time(self, actual_time):
        """Check time and apply faults"""
        if self.time != actual_time:
            self.time = actual_time
        else:
            return

        for i in range(self.n):
            if self.tf[i] == self.time:
                self.system.Log.info('\n <Fault> Applying fault on Bus <{}> at t={}.'.format(self.bus[i], self.tf[i]))
                self.u[i] = 1
                self.active += 1
                self.angle0 = self.system.DAE.y[self.system.Bus.a]
                self.volt0 = self.system.DAE.y[self.system.Bus.n:]
                self.system.DAE.factorize = True

            elif self.tc[i] == self.time:
                self.system.Log.info('\n <Fault> Clearing fault on Bus <{}> at t={}.'.format(self.bus[i], self.tc[i]))
                self.u[i] = 0
                self.active -= 1
                self.system.DAE.y[self.system.Bus.n:] = self.volt0
                # self.system.DAE.y[self.a] = self.anglepre
                self.system.DAE.factorize = True

    def gcall(self, dae):
        if not self.active:
            return
        V2 = mul(self.u, dae.y[self.v] ** 2)
        p = mul(self.gf, V2)
        q = mul(self.bf, V2)
        dae.g += spmatrix(p, self.a, [0] * self.n, (dae.m, 1), 'd')
        dae.g -= spmatrix(q, self.v, [0] * self.n, (dae.m, 1), 'd')

    def gycall(self, dae):
        if not self.active:
            return
        V = mul(2, self.u, dae.y[self.v])
        dae.add_jac(Gy,  mul(self.gf, V), self.a, self.v)
        dae.add_jac(Gy, -mul(self.bf, V), self.v, self.v)

    def insert(self, idx=None, name=None, **kwargs):
        if self.n:
            self._param2list()

        self.add(idx, name, **kwargs)
        self.setup()
