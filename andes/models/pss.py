from cvxopt import matrix, spmatrix
from cvxopt import mul, div

from .base import ModelBase
from .measurement import BusFreq

from ..utils.math import aeb
from ..consts import *


class PSS1(ModelBase):
    """Stabilizer ST2CUT with dual input signals"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'PSS'
        self._name = 'PSS1'
        self._fnamey.extend(['In_1', 'In_2', 'In', 'x_3', 'x_4', 'x_5', 'x_6', 'V_{SS}', 'V_{ST}'])
        self._zeros.extend(['T4'])
        self._states.extend(['x1', 'x2', 'u3', 'u4', 'u5', 'u6'])
        self._params.extend(
            ['Ic1', 'Ic2', 'K1', 'T1', 'K2', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'lsmax', 'lsmin',
             'vcu', 'vcl'])
        self._times.extend(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'])
        self._mandatory.extend(['avr'])
        self._algebs.extend(['In1', 'In2', 'In', 'x3', 'x4', 'x5', 'x6', 'vss', 'vst'])
        self._fnamex.extend(['x_1', 'x_2', 'u_3', 'u_4', 'u_5', 'u_6'])
        self._service.extend(
            ['Ic23', 'Ic12', 'T34', 'toSg', 'Ic15', 'T910', 'Ic21', 'Ic14', 'Ic13', 'Ic22', 'Ic24', 'T56', 'T78', 'v0',
             'Ic11', 'Ic25'])
        self._units.update(
            {'vcu': 'pu', 'T4': 's', 'T5': 's', 'vcl': 'pu', 'T10': 's', 'lsmin': 'pu', 'T9': 's', 'T2': 's', 'T7': 's',
             'T8': 's', 'lsmax': 'pu', 'T6': 's', 'T1': 's', 'T3': 's'})
        self._data.update(
            {'K2': 0, 'vcu': 0.25, 'Ic2': 0, 'T4': 1, 'T5': 1.5, 'lsmin': -0.2, 'vcl': -0.25, 'avr': 0, 'T8': 0.15,
             'T9': 1.5, 'T2': 0.02, 'T7': 1.5, 'K1': 0, 'lsmax': 0.2, 'Ic1': 0, 'T10': 0.15, 'T6': 0.15, 'T1': 0.02,
             'T3': 1})
        self._descr.update({'K2': 'Input 2 gain', 'vcu': 'cutoff upper limit', 'Ic2': 'Input 2 control switch',
                            'T4': 'Washout time constant (denominator)', 'T5': 'LL1 time constant (pole)',
                            'lsmin': 'PSS output minimum limit', 'vcl': 'cutoff lower limit',
                            'Ic1': 'Input 1 control switch', 'T8': 'LL2 time constant (pole)',
                            'T9': 'LL3 time constant (zero)', 'T2': 'Input 2 time constant',
                            'T7': 'LL2 time constant (zero)', 'K1': 'Input 1 gain', 'lsmax': 'PSS output maximum limit',
                            'T10': 'LL3 time constant (pole)', 'avr': 'Exciter id', 'T1': 'Input 1 time constant',
                            'T3': 'Washout time constant (numerator)'})
        self.calls.update({'gcall': True, 'fcall': True, 'fxcall': False, 'init1': True, 'jac0': True, 'gycall': False})
        self._inst_meta()

    def servcall(self, dae):
        self.copy_param('AVR', 'syn', 'syn', self.avr)
        self.copy_param('Synchronous', 'bus', 'bus', self.syn)
        self.copy_param('Synchronous', 'Sn', 'Sg', self.syn)
        self.copy_param('Synchronous', 'v', 'v', self.syn)
        self.copy_param('Synchronous', 'vf', 'vf', self.syn)
        self.copy_param('Synchronous', 'pm', 'pm', self.syn)
        self.copy_param('Synchronous', 'omega', 'omega', self.syn)
        self.copy_param('Synchronous', 'p', 'p', self.syn)
        self.copy_param('BusFreq', 'w', 'w', self.bus)
        self.T34 = mul(self.T3, div(1, self.T4))
        self.T56 = mul(self.T5, div(1, self.T6))
        self.T78 = mul(self.T7, div(1, self.T8))
        self.T910 = mul(self.T9, div(1, self.T10))
        self.toSg = div(self.system.Settings.mva, self.Sg)
        self.v0 = dae.y[self.v]
        self.update_ctrl()

    def update_ctrl(self):
        self.Ic11 = aeb(self.Ic1, 1)
        self.Ic12 = aeb(self.Ic1, 2)
        self.Ic13 = aeb(self.Ic1, 3)
        self.Ic14 = aeb(self.Ic1, 4)
        self.Ic15 = aeb(self.Ic1, 5)
        self.Ic21 = aeb(self.Ic2, 1)
        self.Ic22 = aeb(self.Ic2, 2)
        self.Ic23 = aeb(self.Ic2, 3)
        self.Ic24 = aeb(self.Ic2, 4)
        self.Ic25 = aeb(self.Ic2, 5)
        self.vtmax = self.v0 + self.vcu
        self.vtmin = self.v0 + self.vcl

    def init1(self, dae):
        self.servcall(dae)
        dae.x[self.x1] = mul(dae.y[self.In1], self.u)
        dae.x[self.x2] = mul(dae.y[self.In2], self.u)
        dae.y[self.In1] = mul(self.u, mul(self.Ic11, -1 + dae.x[self.omega]) + mul(self.Ic12, -1 + dae.x[self.w]) + mul(self.Ic15, dae.y[self.v]) + mul(self.Ic13, dae.y[self.p], self.toSg) + mul(self.Ic14, dae.y[self.pm], self.toSg))
        dae.y[self.In2] = mul(self.u, mul(self.Ic21, -1 + dae.x[self.omega]) + mul(self.Ic22, -1 + dae.x[self.w]) + mul(self.Ic25, dae.y[self.v]) + mul(self.Ic23, dae.y[self.p], self.toSg) + mul(self.Ic24, dae.y[self.pm], self.toSg))
        dae.y[self.In] = mul(self.u, dae.y[self.In1] + dae.y[self.In2])
        dae.x[self.u3] = mul(dae.y[self.In], self.T34, self.u)

    def gcall(self, dae):
        dae.g[self.In1] = mul(self.u, -dae.y[self.In1] + mul(self.Ic11, -1 + dae.x[self.omega]) + mul(self.Ic12, -1 + dae.x[self.w]) + mul(self.Ic15, dae.y[self.v]) + mul(self.Ic13, dae.y[self.p], self.toSg) + mul(self.Ic14, dae.y[self.pm], self.toSg))
        dae.g[self.In2] = mul(self.u, -dae.y[self.In2] + mul(self.Ic21, -1 + dae.x[self.omega]) + mul(self.Ic22, -1 + dae.x[self.w]) + mul(self.Ic25, dae.y[self.v]) + mul(self.Ic23, dae.y[self.p], self.toSg) + mul(self.Ic24, dae.y[self.pm], self.toSg))
        dae.g[self.In] = mul(self.u, dae.y[self.In1] + dae.y[self.In2] - dae.y[self.In])
        dae.g[self.x3] = mul(self.u, -dae.x[self.u3] - dae.y[self.x3] + mul(dae.y[self.In], self.T34))
        dae.g[self.x4] = mul(self.u, dae.x[self.u4] - dae.y[self.x4] + mul(self.T56, dae.y[self.x3]))
        dae.g[self.x5] = mul(self.u, dae.x[self.u5] - dae.y[self.x5] + mul(self.T78, dae.y[self.x4]))
        dae.g[self.x6] = mul(self.u, dae.x[self.u6] - dae.y[self.x6] + mul(self.T910, dae.y[self.x5]))
        dae.g[self.vss] = mul(self.u, dae.y[self.x6] - dae.y[self.vss])
        dae.hard_limit(self.vss, self.lsmin, self.lsmax)
        dae.g[self.vst] = mul(self.u, dae.y[self.vss] - dae.y[self.vst])
        dae.hard_limit_remote(self.vst, self.v, rtype='y', rmin=self.vtmin, rmax=self.vtmax, min_yset=0, max_yset=0)
        dae.g += spmatrix(mul(self.u, dae.y[self.vst]), self.vf, [0] * self.n, (dae.m, 1), 'd')

    def fcall(self, dae):
        dae.f[self.x1] = mul(self.u, div(1, self.T1), -dae.x[self.x1] + mul(dae.y[self.In1], self.K1))
        dae.f[self.x2] = mul(self.u, div(1, self.T2), -dae.x[self.x2] + mul(dae.y[self.In2], self.K2))
        dae.f[self.u3] = mul(self.u, div(1, self.T4), -dae.x[self.u3] + mul(dae.y[self.In], self.T34))
        dae.f[self.u4] = mul(self.u, div(1, self.T6), -dae.x[self.u4] + mul(dae.y[self.x3], 1 - self.T56))
        dae.f[self.u5] = mul(self.u, div(1, self.T8), -dae.x[self.u5] + mul(dae.y[self.x4], 1 - self.T78))
        dae.f[self.u6] = mul(self.u, div(1, self.T10), -dae.x[self.u6] + mul(dae.y[self.x5], 1 - self.T910))

    def jac0(self, dae):
        dae.add_jac(Gy0, mul(self.Ic15, self.u), self.In1, self.v)
        dae.add_jac(Gy0, - self.u, self.In1, self.In1)
        dae.add_jac(Gy0, mul(self.Ic14, self.toSg, self.u), self.In1, self.pm)
        dae.add_jac(Gy0, mul(self.Ic13, self.toSg, self.u), self.In1, self.p)
        dae.add_jac(Gy0, mul(self.Ic25, self.u), self.In2, self.v)
        dae.add_jac(Gy0, mul(self.Ic24, self.toSg, self.u), self.In2, self.pm)
        dae.add_jac(Gy0, - self.u, self.In2, self.In2)
        dae.add_jac(Gy0, mul(self.Ic23, self.toSg, self.u), self.In2, self.p)
        dae.add_jac(Gy0, self.u, self.In, self.In2)
        dae.add_jac(Gy0, self.u, self.In, self.In1)
        dae.add_jac(Gy0, - self.u, self.In, self.In)
        dae.add_jac(Gy0, - self.u, self.x3, self.x3)
        dae.add_jac(Gy0, mul(self.T34, self.u), self.x3, self.In)
        dae.add_jac(Gy0, mul(self.T56, self.u), self.x4, self.x3)
        dae.add_jac(Gy0, - self.u, self.x4, self.x4)
        dae.add_jac(Gy0, - self.u, self.x5, self.x5)
        dae.add_jac(Gy0, mul(self.T78, self.u), self.x5, self.x4)
        dae.add_jac(Gy0, - self.u, self.x6, self.x6)
        dae.add_jac(Gy0, mul(self.T910, self.u), self.x6, self.x5)
        dae.add_jac(Gy0, self.u, self.vss, self.x6)
        dae.add_jac(Gy0, - self.u, self.vss, self.vss)
        dae.add_jac(Gy0, self.u, self.vst, self.vss)
        dae.add_jac(Gy0, - self.u, self.vst, self.vst)
        dae.add_jac(Gy0, self.u, self.vf, self.vst)
        dae.add_jac(Gx0, mul(self.Ic11, self.u), self.In1, self.omega)
        dae.add_jac(Gx0, mul(self.Ic12, self.u), self.In1, self.w)
        dae.add_jac(Gx0, mul(self.Ic21, self.u), self.In2, self.omega)
        dae.add_jac(Gx0, mul(self.Ic22, self.u), self.In2, self.w)
        dae.add_jac(Gx0, - self.u, self.x3, self.u3)
        dae.add_jac(Gx0, self.u, self.x4, self.u4)
        dae.add_jac(Gx0, self.u, self.x5, self.u5)
        dae.add_jac(Gx0, self.u, self.x6, self.u6)
        dae.add_jac(Fx0, - mul(self.u, div(1, self.T1)), self.x1, self.x1)
        dae.add_jac(Fx0, - mul(self.u, div(1, self.T2)), self.x2, self.x2)
        dae.add_jac(Fx0, - mul(self.u, div(1, self.T4)), self.u3, self.u3)
        dae.add_jac(Fx0, - mul(self.u, div(1, self.T6)), self.u4, self.u4)
        dae.add_jac(Fx0, - mul(self.u, div(1, self.T8)), self.u5, self.u5)
        dae.add_jac(Fx0, - mul(self.u, div(1, self.T10)), self.u6, self.u6)
        dae.add_jac(Fy0, mul(self.K1, self.u, div(1, self.T1)), self.x1, self.In1)
        dae.add_jac(Fy0, mul(self.K2, self.u, div(1, self.T2)), self.x2, self.In2)
        dae.add_jac(Fy0, mul(self.T34, self.u, div(1, self.T4)), self.u3, self.In)
        dae.add_jac(Fy0, mul(self.u, div(1, self.T6), 1 - self.T56), self.u4, self.x3)
        dae.add_jac(Fy0, mul(self.u, div(1, self.T8), 1 - self.T78), self.u5, self.x4)
        dae.add_jac(Fy0, mul(self.u, div(1, self.T10), 1 - self.T910), self.u6, self.x5)
        dae.add_jac(Gy0, 1e-6, self.In1, self.In1)
        dae.add_jac(Gy0, 1e-6, self.In2, self.In2)
        dae.add_jac(Gy0, 1e-6, self.In, self.In)
        dae.add_jac(Gy0, 1e-6, self.x3, self.x3)
        dae.add_jac(Gy0, 1e-6, self.x4, self.x4)
        dae.add_jac(Gy0, 1e-6, self.x5, self.x5)
        dae.add_jac(Gy0, 1e-6, self.x6, self.x6)
        dae.add_jac(Gy0, 1e-6, self.vss, self.vss)
        dae.add_jac(Gy0, 1e-6, self.vst, self.vst)
