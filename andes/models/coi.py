from cvxopt import matrix, spmatrix
from cvxopt import mul, div, exp
from ..consts import *
from .base import ModelBase


class COI(ModelBase):
    def __init__(self, system, name):
        super(COI, self).__init__(system, name)
        self._data.update({'syn': None,
                           })
        self._algebs.extend(['delta', 'omega'])
        self.calls.update({'init1': True, 'gcall': True,
                           'fcall': True, 'jac0': True,
                           })
        self._service.extend(['H', 'M', 'Mtot', 'usyn', 'gdelta', 'gomega'])
        self._fnamey.extend(['\delta', '\omega'])
        self._inst_meta()

    def init1(self, dae):
        for item in self._service:
            self.__dict__[item] = [[]] * self.n

        dae.y[self.omega] = 1
        for idx, item in enumerate(self.syn):
            self.usyn[idx] = self.read_param('Synchronous', src='u', fkey=item)
            self.M[idx] = self.read_param('Synchronous', src='M', fkey=item)
            self.Mtot[idx] = sum(self.M[idx])
            self.gdelta[idx] = self.read_param('Synchronous', src='delta', fkey=item)
            self.gomega[idx] = self.read_param('Synchronous', src='omega', fkey=item)

            dae.y[self.delta[idx]] = sum(mul(self.M[idx], dae.x[self.gdelta[idx]])) / self.Mtot[idx]

    def gcall(self, dae):
        for idx in range(self.n):
            dae.g[self.omega[idx]] = dae.y[self.omega[idx]] - sum(mul(self.M[idx], dae.x[self.gomega[idx]])) / self.Mtot[idx]
            dae.g[self.delta[idx]] = dae.y[self.delta[idx]] - sum(mul(self.M[idx], dae.x[self.gdelta[idx]])) / self.Mtot[idx]

    def jac0(self, dae):
        dae.add_jac(Gy0, 1, self.omega, self.omega)
        dae.add_jac(Gy0, 1, self.delta, self.delta)

    def fcall(self, dae):
        for idx in range(self.n):
            dae.f[self.gdelta[idx]] += 2*pi*self.system.Settings.freq *(1 - dae.y[self.omega[idx]])

    def fxcall(self, dae):
        for idx in range(self.n):
            dae.add_jac(Fy, -2*pi*self.system.Settings.freq, self.gdelta[idx], self.omega[idx])
            dae.add_jac(Gx, -self.M[idx] / self.Mtot[idx], self.omega[idx], self.gomega[idx])
            dae.add_jac(Gx, -self.M[idx] / self.Mtot[idx], self.delta[idx], self.delta[idx])


