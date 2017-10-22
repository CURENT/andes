from cvxopt import matrix, spmatrix
from cvxopt import mul, div, exp
from ..consts import *
from .base import ModelBase


class ACE(ModelBase):
    def __init__(self, system, name):
        super(ACE, self).__init__(system, name)
        self._data.update({'area': None,
                           })
        self._mandatory.extend(['area'])
        self._algebs.extend(['e'])
        self.calls.update({'gcall': True, 'init1': True,
                           'jac0': True,
                           })
        self._service.extend(['P0', 'Q0'])
        self._fnamey.extend(['\epsilon'])
        self._inst_meta()

    def init1(self, dae):
        self.copy_param('Area', src='area_P0', dest='P0', fkey=self.area)
        self.copy_param('Area', src='area_Q0', dest='Q0', fkey=self.area)

    def gcall(self, dae):
        P = self.read_param('Area', src='area_P0', fkey=self.area)

        dae.g[self.e] = dae.y[self.e] - (P - self.P0)

    def jac0(self, dae):
        dae.add_jac(Gy0, 1, self.e, self.e)


class AGC(ModelBase):
    def __init__(self, system, name):
        super(AGC, self).__init__(system, name)
        self._data.update({'coi': None,
                           'ace': None,
                           'beta': 0,
                           'Ki': 0,
                           })
        self._params.extend(['beta', 'Ki'])
        self._algebs.extend(['ACE'])
        self._states.extend(['AGC'])
        self._fnamey.extend(['ACE'])
        self._fnamex.extend(['AGC'])

        self._mandatory.extend(['coi', 'ace'])
        self.calls.update({'init1': True, 'gcall': True,
                           'jac0': True, 'fcall': True,

                           })
        self._service.extend(['pm'])
        self._inst_meta()

    def init1(self, dae):
        self.pm = [[]] * self.n
        self.copy_param('ACE', src='e', fkey=self.ace)
        self.copy_param('COI', src='syn', fkey=self.coi)
        self.copy_param('COI', src='omega', dest='comega', fkey=self.coi)
        self.copy_param('COI', src='M', dest='M', fkey=self.coi)
        self.copy_param('COI', src='Mtot', dest='Mtot', fkey=self.coi)
        for idx in range(self.n):
            self.pm[idx] = self.read_param('Synchronous', src='pm', fkey=self.syn[idx])

    def gcall(self, dae):
        dae.g[self.ACE] = -mul(self.beta, (dae.y[self.comega] - 1)) - dae.y[self.e] - dae.y[self.ACE]
        for idx in range(self.n):
            dae.g[self.pm[idx]] += div(self.M[idx], self.Mtot[idx]) * dae.x[self.AGC[idx]]

    def fcall(self, dae):
        dae.f[self.AGC] = mul(self.Ki, dae.y[self.ACE])

    def jac0(self, dae):
        dae.add_jac(Gy0, -1, self.ACE, self.ACE)
        dae.add_jac(Gy0, - self.beta, self.ACE, self.comega)
        dae.add_jac(Gy0, -1, self.ACE, self.e)

        dae.add_jac(Fy0, self.Ki, self.AGC, self.ACE)

    def gycall(self, dae):
        for idx in range(self.n):
            dae.add_jac(Gx0, div(self.M[idx], self.Mtot[idx]), self.pm[idx], self.AGC[idx])

