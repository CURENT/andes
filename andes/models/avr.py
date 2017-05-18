from cvxopt import matrix, spmatrix
from cvxopt import mul, div
from ..consts import *
from .base import ModelBase


class AVR3(ModelBase):
    """Automatic Voltage Regulator Type III"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'AVR'
        self._name = 'AVR3'
        self._service.extend(['v0', 'vref0', 'T1T2'])
        self._params.extend(['syn', 'T1', 'T2', 'Tr', 'Te', 'vfmax', 'vfmin', 'K0', 's0'])
        self._algebs.extend(['vref'])
        self._times.extend(['T2', 'Te', 'Tr'])
        self._fnamex.extend(['v_{m}', 'v_{r}', 'v_{fout}'])
        self._mandatory.extend(['syn'])
        self._states.extend(['vm', 'vr', 'vfout'])
        self._fnamey.extend(['v_{ref}'])
        self._units.update({'T1': 's', 'T2': 's', 'Te': 's', 'Tr': 's', 'vfmax': 'pu', 'vfmin': 'pu'})
        self._data.update(
            {'T1': 0.01, 'T2': 0.1, 'Tr': 0.001, 'K0': 20, 'vfmax': 5, 'Te': 1.0, 'syn': 0, 'vfmin': -5, 's0': True})
        self._descr.update(
            {'T1': 'Regulator zero', 'T2': 'Regularot Pole', 'Tr': 'Measurement time constant', 'K0': 'Regulator gain',
             'vfmax': 'Maximum field voltage', 'Te': 'Field circuit time constant', 'syn': 'Generator id',
             'vfmin': 'Minimum field voltage', 's0': 'Enable excitation voltage feedback'})
        self.calls.update({'fxcall': True, 'gycall': False, 'fcall': True, 'gcall': True, 'init1': True, 'jac0': True})
        self._inst_meta()

    def servcall(self, dae):
        self.copy_param('Synchronous', 'v', 'v', self.syn)
        self.copy_param('Synchronous', 'vf', 'vf', self.syn)
        self.copy_param('Synchronous', 'vf0', 'vf0', self.syn)
        self.v0 = dae.y[self.v]
        self.vref0 = dae.y[self.v]
        self.T1T2 = mul(self.T1, div(1, self.T2))

    def init1(self, dae):
        self.servcall(dae)
        dae.y[self.vref] = dae.y[self.v]
        dae.x[self.vfout] = dae.y[self.vf]
        dae.x[self.vm] = dae.y[self.v]

    def gcall(self, dae):
        dae.g[self.vref] = self.vref0 - dae.y[self.vref]
        dae.g += spmatrix(self.vf0 - dae.x[self.vfout], self.vf, [0]*self.n, (dae.m, 1), 'd')

    def fcall(self, dae):
        dae.f[self.vm] = mul(div(1, self.Tr), dae.y[self.v] - dae.x[self.vm])
        dae.f[self.vr] = mul(div(1, self.T2), -dae.x[self.vr] + mul(self.K0, 1 - self.T1T2, dae.y[self.vref] - dae.x[self.vm]))
        vfout = mul(1 + mul(self.s0, -1 + mul(dae.y[self.v], div(1, self.v0))), self.vf0 + dae.x[self.vr] + mul(self.K0, self.T1T2, dae.y[self.vref] - dae.x[self.vm]))
        dae.f[self.vfout] = div(vfout - dae.x[self.vfout], self.Te)
        dae.anti_windup(self.vfout, self.vfmin, self.vfmax)

    def fxcall(self, dae):
        dae.add_jac(Fx, - mul(self.K0, self.T1T2, 1 + mul(self.s0, -1 + mul(dae.y[self.v], div(1, self.v0)))), self.vfout, self.vm)
        dae.add_jac(Fx, 1 + mul(self.s0, -1 + mul(dae.y[self.v], div(1, self.v0))), self.vfout, self.vr)
        dae.add_jac(Fy, mul(self.K0, self.T1T2, 1 + mul(self.s0, -1 + mul(dae.y[self.v], div(1, self.v0)))), self.vfout, self.vref)
        dae.add_jac(Fy, mul(self.s0, div(1, self.v0), self.vf0 + dae.x[self.vr] + mul(self.K0, self.T1T2, dae.y[self.vref] - dae.x[self.vm])), self.vfout, self.v)

    def jac0(self, dae):
        dae.add_jac(Gy0, -1, self.vref, self.vref)
        dae.add_jac(Gx0, -1, self.v, self.vfout)
        dae.add_jac(Fx0, - div(1, self.Tr), self.vm, self.vm)
        dae.add_jac(Fx0, - mul(self.K0, div(1, self.T2), 1 - self.T1T2), self.vr, self.vm)
        dae.add_jac(Fx0, - div(1, self.T2), self.vr, self.vr)
        dae.add_jac(Fy0, div(1, self.Tr), self.vm, self.v)
        dae.add_jac(Fy0, mul(self.K0, div(1, self.T2), 1 - self.T1T2), self.vr, self.vref)
