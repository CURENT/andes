from cvxopt import matrix, spmatrix
from cvxopt import mul, div

from .base import ModelBase
from .measurement import BusFreq  # NOQA

from ..utils.math import aeqb, sdiv

from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA


class PSS1(ModelBase):
    """Stabilizer ST2CUT with dual input signals"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'PSS'
        self._name = 'PSS1'
        self._fnamey.extend([
            'In_1', 'In_2', 'In', 'x_3', 'x_4', 'x_5', 'x_6', 'V_{SS}',
            'V_{ST}'
        ])
        self._zeros.extend(['T4'])
        self._states.extend(['x1', 'x2', 'u3', 'u4', 'u5', 'u6'])
        self._params.extend([
            'Ic1', 'Ic2', 'K1', 'T1', 'K2', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7',
            'T8', 'T9', 'T10', 'lsmax', 'lsmin', 'vcu', 'vcl', 'd1', 'd2', 'd3'
        ])
        self._times.extend(
            ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'])
        self._mandatory.extend(['avr'])
        self._algebs.extend(
            ['In1', 'In2', 'In', 'x3', 'x4', 'x5', 'x6', 'vss', 'vst'])
        self._fnamex.extend(['x_1', 'x_2', 'u_3', 'u_4', 'u_5', 'u_6'])
        self._service.extend([
            'Ic23', 'Ic12', 'T34', 'toSg', 'Ic15', 'T910', 'Ic21', 'Ic14',
            'Ic13', 'Ic22', 'Ic24', 'T56', 'T78', 'v0', 'Ic11', 'Ic25', 'u0',
            'bus'
        ])
        self._units.update({
            'vcu': 'pu',
            'T4': 's',
            'T5': 's',
            'vcl': 'pu',
            'T10': 's',
            'lsmin': 'pu',
            'T9': 's',
            'T2': 's',
            'T7': 's',
            'T8': 's',
            'lsmax': 'pu',
            'T6': 's',
            'T1': 's',
            'T3': 's'
        })
        self._data.update({
            'K2': 0,
            'vcu': 0.25,
            'Ic2': 0,
            'T4': 1,
            'T5': 1.5,
            'lsmin': -0.2,
            'vcl': -0.25,
            'avr': 0,
            'T8': 0.15,
            'T9': 1.5,
            'T2': 0.02,
            'T7': 1.5,
            'K1': 0,
            'lsmax': 0.2,
            'Ic1': 0,
            'T10': 0.15,
            'T6': 0.15,
            'T1': 0.02,
            'T3': 1,
            'd1': 1,
            'd2': 1,
            'd3': 1
        })
        self._descr.update({
            'K2': 'Input 2 gain',
            'vcu': 'cutoff upper limit offset over Vt',
            'Ic2': 'Input 2 control switch',
            'T4': 'Washout time constant (denominator)',
            'T5': 'LL1 time constant (zero)',
            'lsmin': 'PSS output minimum limit',
            'vcl': 'cutoff lower limit offset over Vt',
            'Ic1': 'Input 1 control switch',
            'T8': 'LL2 time constant (pole)',
            'T9': 'LL3 time constant (zero)',
            'T2': 'Input 2 time constant',
            'T7': 'LL2 time constant (zero)',
            'K1': 'Input 1 gain',
            'lsmax': 'PSS output maximum limit',
            'T10': 'LL3 time constant (pole)',
            'avr': 'Exciter id',
            'T1': 'Input 1 time constant',
            'T3': 'Washout time constant (numerator)',
            'T6': 'LL1 time constant (pole)'
        })
        self.calls.update({
            'gcall': True,
            'fcall': True,
            'fxcall': False,
            'init1': True,
            'jac0': True,
            'gycall': False
        })
        self._zeros.extend(['T1', 'T2', 'T4'])
        self.param_remove('Vn')
        self.param_remove('Sn')
        self._init()
        # todo: fix T6, T8 or T10 == 10. Ignore the filter if they are zeros.

    def servcall(self, dae):
        self.copy_data_ext('AVR', 'syn', 'syn', self.avr)
        self.copy_data_ext('AVR', 'u', 'uavr', self.avr)
        self.copy_data_ext('Synchronous', 'u', 'usyn', self.syn)
        self.copy_data_ext('Synchronous', 'bus', 'bus', self.syn)
        self.copy_data_ext('Synchronous', 'Sn', 'Sg', self.syn)
        self.copy_data_ext('Synchronous', 'v', 'v', self.syn)
        self.copy_data_ext('Synchronous', 'vf', 'vf', self.syn)
        self.copy_data_ext('Synchronous', 'pm', 'pm', self.syn)
        self.copy_data_ext('Synchronous', 'omega', 'omega', self.syn)
        self.copy_data_ext('Synchronous', 'p', 'p', self.syn)
        self.copy_data_ext('BusFreq', 'w', 'w', self.bus)
        self.T34 = sdiv(self.T3, self.T4)
        self.T56 = sdiv(self.T5, self.T6)
        self.T78 = sdiv(self.T7, self.T8)
        self.T910 = sdiv(self.T9, self.T10)
        self.set_flag('T6', 'd1', reset_val=True)
        self.set_flag('T8', 'd2', reset_val=True)
        self.set_flag('T10', 'd3', reset_val=True)
        self.toSg = div(self.system.mva, self.Sg)
        self.v0 = dae.y[self.v]
        self.update_ctrl()

    def set_flag(self, value, flag, reset_val=False):
        """Set a flag to 0 if the corresponding value is 0"""
        if not self.__dict__[flag]:
            self.__dict__[flag] = matrix(1.0, (len(self.__dict__[value]), 1),
                                         'd')
        for idx, item in enumerate(self.__dict__[value]):
            if item == 0:
                self.__dict__[flag][idx] = 0
                if reset_val:
                    self.__dict__[value][idx] = 1

    def update_ctrl(self):
        self.u0 = mul(self.u, self.uavr,
                      self.usyn)  # effective PSS connectivity status
        self.Ic11 = aeqb(self.Ic1, 1)
        self.Ic12 = aeqb(self.Ic1, 2)
        self.Ic13 = aeqb(self.Ic1, 3)
        self.Ic14 = aeqb(self.Ic1, 4)
        self.Ic15 = aeqb(self.Ic1, 5)
        self.Ic21 = aeqb(self.Ic2, 1)
        self.Ic22 = aeqb(self.Ic2, 2)
        self.Ic23 = aeqb(self.Ic2, 3)
        self.Ic24 = aeqb(self.Ic2, 4)
        self.Ic25 = aeqb(self.Ic2, 5)
        # ignore the hard limiters if vcu == 0 or vcl == 0
        self.vcu += mul(aeqb(self.vcu, 0.0), 9999)
        self.vcl += mul(aeqb(self.vcl, 0.0), -9999)
        self.vtmax = self.v0 + self.vcu
        self.vtmin = self.v0 + self.vcl

    def init1(self, dae):
        self.servcall(dae)
        dae.y[self.In1] = mul(
            self.u0,
            mul(self.Ic11, -1 + dae.x[self.omega]) + mul(
                self.Ic12, -1 + dae.x[self.w]) + mul(self.Ic15, dae.y[self.v])
            + mul(self.Ic13, dae.y[self.p], self.toSg) + mul(
                self.Ic14, dae.y[self.pm], self.toSg))
        dae.y[self.In2] = mul(
            self.u0,
            mul(self.Ic21, -1 + dae.x[self.omega]) + mul(
                self.Ic22, -1 + dae.x[self.w]) + mul(self.Ic25, dae.y[self.v])
            + mul(self.Ic23, dae.y[self.p], self.toSg) + mul(
                self.Ic24, dae.y[self.pm], self.toSg))
        dae.x[self.x1] = mul(self.K1, dae.y[self.In1], self.u0)
        dae.x[self.x2] = mul(self.K2, dae.y[self.In2], self.u0)
        dae.y[self.In] = mul(self.u0, dae.y[self.In1] + dae.y[self.In2])
        dae.x[self.u3] = mul(dae.y[self.In], self.T34, self.u0)

    def gcall(self, dae):
        dae.g[self.In1] = mul(
            self.u0,
            -dae.y[self.In1] + mul(self.Ic11, -1 + dae.x[self.omega]) + mul(
                self.Ic12, -1 + dae.x[self.w]) + mul(self.Ic15, dae.y[self.v])
            + mul(self.Ic13, dae.y[self.p], self.toSg) + mul(
                self.Ic14, dae.y[self.pm], self.toSg))
        dae.g[self.In2] = mul(
            self.u0,
            -dae.y[self.In2] + mul(self.Ic21, -1 + dae.x[self.omega]) + mul(
                self.Ic22, -1 + dae.x[self.w]) + mul(self.Ic25, dae.y[self.v])
            + mul(self.Ic23, dae.y[self.p], self.toSg) + mul(
                self.Ic24, dae.y[self.pm], self.toSg))
        dae.g[self.In] = mul(
            self.u0, dae.y[self.In1] + dae.y[self.In2] - dae.y[self.In])
        dae.g[self.x3] = mul(
            self.u0,
            -dae.x[self.u3] - dae.y[self.x3] + mul(dae.y[self.In], self.T34))
        dae.g[self.x4] = mul(
            self.u0,
            dae.x[self.u4] - dae.y[self.x4] + mul(self.T56, dae.y[self.x3]))
        dae.g[self.x5] = mul(
            self.u0,
            dae.x[self.u5] - dae.y[self.x5] + mul(self.T78, dae.y[self.x4]))
        dae.g[self.x6] = mul(
            self.u0,
            dae.x[self.u6] - dae.y[self.x6] + mul(self.T910, dae.y[self.x5]))
        dae.g[self.vss] = mul(self.u0, dae.y[self.x6] - dae.y[self.vss])
        dae.hard_limit(self.vss, self.lsmin, self.lsmax)
        dae.g[self.vst] = mul(self.u0, dae.y[self.vss] - dae.y[self.vst])
        dae.hard_limit_remote(
            self.vst,
            self.v,
            rtype='y',
            rmin=self.vtmin,
            rmax=self.vtmax,
            min_yset=0,
            max_yset=0)
        dae.g += spmatrix(
            mul(self.u0, dae.y[self.vst]), self.vf, [0] * self.n, (dae.m, 1),
            'd')

    def fcall(self, dae):
        dae.f[self.x1] = mul(self.u0, div(1, self.T1),
                             -dae.x[self.x1] + mul(dae.y[self.In1], self.K1))
        dae.f[self.x2] = mul(self.u0, div(1, self.T2),
                             -dae.x[self.x2] + mul(dae.y[self.In2], self.K2))
        dae.f[self.u3] = mul(self.u0, div(1, self.T4),
                             -dae.x[self.u3] + mul(dae.y[self.In], self.T34))
        dae.f[self.u4] = mul(
            self.u0, self.d1, div(1, self.T6),
            -dae.x[self.u4] + mul(dae.y[self.x3], 1 - self.T56))
        dae.f[self.u5] = mul(
            self.u0, self.d2, div(1, self.T8),
            -dae.x[self.u5] + mul(dae.y[self.x4], 1 - self.T78))
        dae.f[self.u6] = mul(
            self.u0, self.d3, div(1, self.T10),
            -dae.x[self.u6] + mul(dae.y[self.x5], 1 - self.T910))

    def jac0(self, dae):
        dae.add_jac(Gy0, mul(self.Ic15, self.u0), self.In1, self.v)
        dae.add_jac(Gy0, -self.u0, self.In1, self.In1)
        dae.add_jac(Gy0, mul(self.Ic14, self.toSg, self.u0), self.In1, self.pm)
        dae.add_jac(Gy0, mul(self.Ic13, self.toSg, self.u0), self.In1, self.p)
        dae.add_jac(Gy0, mul(self.Ic25, self.u0), self.In2, self.v)
        dae.add_jac(Gy0, mul(self.Ic24, self.toSg, self.u0), self.In2, self.pm)
        dae.add_jac(Gy0, -self.u0, self.In2, self.In2)
        dae.add_jac(Gy0, mul(self.Ic23, self.toSg, self.u0), self.In2, self.p)
        dae.add_jac(Gy0, self.u0, self.In, self.In2)
        dae.add_jac(Gy0, self.u0, self.In, self.In1)
        dae.add_jac(Gy0, -self.u0, self.In, self.In)
        dae.add_jac(Gy0, -self.u0, self.x3, self.x3)
        dae.add_jac(Gy0, mul(self.T34, self.u0), self.x3, self.In)
        dae.add_jac(Gy0, mul(self.T56, self.u0), self.x4, self.x3)
        dae.add_jac(Gy0, -self.u0, self.x4, self.x4)
        dae.add_jac(Gy0, -self.u0, self.x5, self.x5)
        dae.add_jac(Gy0, mul(self.T78, self.u0), self.x5, self.x4)
        dae.add_jac(Gy0, -self.u0, self.x6, self.x6)
        dae.add_jac(Gy0, mul(self.T910, self.u0), self.x6, self.x5)
        dae.add_jac(Gy0, self.u0, self.vss, self.x6)
        dae.add_jac(Gy0, -self.u0, self.vss, self.vss)
        dae.add_jac(Gy0, self.u0, self.vst, self.vss)
        dae.add_jac(Gy0, -self.u0, self.vst, self.vst)
        dae.add_jac(Gy0, self.u0, self.vf, self.vst)
        dae.add_jac(Gx0, mul(self.Ic11, self.u0), self.In1, self.omega)
        dae.add_jac(Gx0, mul(self.Ic12, self.u0), self.In1, self.w)
        dae.add_jac(Gx0, mul(self.Ic21, self.u0), self.In2, self.omega)
        dae.add_jac(Gx0, mul(self.Ic22, self.u0), self.In2, self.w)
        dae.add_jac(Gx0, -self.u0, self.x3, self.u3)
        dae.add_jac(Gx0, self.u0, self.x4, self.u4)
        dae.add_jac(Gx0, self.u0, self.x5, self.u5)
        dae.add_jac(Gx0, self.u0, self.x6, self.u6)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.T1)), self.x1, self.x1)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.T2)), self.x2, self.x2)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.T4)), self.u3, self.u3)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.T6)), self.u4, self.u4)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.T8)), self.u5, self.u5)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.T10)), self.u6, self.u6)
        dae.add_jac(Fy0, mul(self.K1, self.u0, div(1, self.T1)), self.x1,
                    self.In1)
        dae.add_jac(Fy0, mul(self.K2, self.u0, div(1, self.T2)), self.x2,
                    self.In2)
        dae.add_jac(Fy0, mul(self.T34, self.u0, div(1, self.T4)), self.u3,
                    self.In)
        dae.add_jac(Fy0, mul(self.u0, div(1, self.T6), 1 - self.T56), self.u4,
                    self.x3)
        dae.add_jac(Fy0, mul(self.u0, div(1, self.T8), 1 - self.T78), self.u5,
                    self.x4)
        dae.add_jac(Fy0, mul(self.u0, div(1, self.T10), 1 - self.T910),
                    self.u6, self.x5)
        dae.add_jac(Gy0, 1e-6, self.In1, self.In1)
        dae.add_jac(Gy0, 1e-6, self.In2, self.In2)
        dae.add_jac(Gy0, 1e-6, self.In, self.In)
        dae.add_jac(Gy0, 1e-6, self.x3, self.x3)
        dae.add_jac(Gy0, 1e-6, self.x4, self.x4)
        dae.add_jac(Gy0, 1e-6, self.x5, self.x5)
        dae.add_jac(Gy0, 1e-6, self.x6, self.x6)
        dae.add_jac(Gy0, 1e-6, self.vss, self.vss)
        dae.add_jac(Gy0, 1e-6, self.vst, self.vst)


class PSS2(ModelBase):
    """Stabilizer IEEEST"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'PSS'
        self._name = 'PSS2'
        self._algebs.extend(['In', 'v1', 'v2', 'v3', 'v4', 'vss', 'vst'])
        self._fnamex.extend(['q_0', 'q_1', 'q_2', 'q_3', 'x_1', 'x_2', 'x_3'])
        self._fnamey.extend(
            ['In', 'v_1', 'v_2', 'v_3', 'v_4', 'V_{SS}', 'V_{ST}'])
        self._mandatory.extend(['avr'])
        self._params.extend([
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'Ks', 'T1', 'T2', 'T3', 'T4',
            'T5', 'T6', 'lsmax', 'lsmin', 'vcu', 'vcl', 'Ic'
        ])
        self._service.extend([
            'Ic3', 'Ic5', 'Ic1', 'H2', 'H1', 'H3', 'T34', 'T56', 'T12', 'toSg',
            'Ic2', 'Ic6', 'Ic4', 'u0', 'KsT56', 'H4'
        ])
        self._states.extend(['q0', 'q1', 'q2', 'q3', 'x1', 'x2', 'x3'])
        self._times.extend(['T1', 'T2', 'T3', 'T4', 'T5', 'T6'])
        self._data.update({
            'A4': 1,
            'lsmax': 0.1,
            'u': 1,
            'A1': 1,
            'Ks': -2,
            'lsmin': -0.1,
            'avr': 0,
            'T4': 0.75,
            'T6': 0.2,
            'T2': 0.02,
            'A5': 1,
            'T5': 1,
            'T1': 0.02,
            'A2': 1,
            'A6': 1,
            'T3': 1,
            'Ic': 0,
            'A3': 1,
            'vcu': 1.2,
            'vcl': 0.8
        })
        self._descr.update({
            'A4':
            'Filter-1 denominator coefficient',
            'lsmax':
            'PSS output maximum limit',
            'A6':
            'Filter-1 numerator coefficient',
            'A1':
            'Filter-1 denominator coefficient (1 + s*A1 + s^2 * A2)',
            'Ks':
            'Gain',
            'lsmin':
            'PSS output minimum limit',
            'avr':
            'Exciter id',
            'T4':
            'Filter-3 denominator time constant',
            'T6':
            'Washout denominator time constant',
            'A5':
            'Filter-1 numerator coefficient (1 + s*A5 + s^2 * A6)',
            'T5':
            'Washout numerator time constant',
            'T1':
            'Filter-2 numerator time constant',
            'A2':
            'Filter-1 denominator coefficient',
            'T2':
            'Filter-2 denominator time constant',
            'T3':
            'Filter-3 numerator time constant',
            'Ic':
            'Input signal selector (1) shaft speed dev (2) bus freq dev '
            '(3) P in syn base (4) Pm in syn base (5) Vbus',
            'A3':
            'Filter-1 denominator coefficient (1 + s*A3 + s^2 * A4)',
            'vcu':
            'Cutoff upper limit',
            'vcl':
            'Cutoff lower limit'
        })
        self._units.update({
            'A4': 'pu',
            'lsmax': 'pu',
            'A6': 'pu',
            'A1': 'pu',
            'Ks': 'pu',
            'lsmin': 'pu',
            'vcl': 'pu',
            'T4': 's',
            'T6': 's',
            'A5': 'pu',
            'T5': 's',
            'T1': 's',
            'A2': 'pu',
            'T2': 's',
            'T3': 's',
            'A3': 'pu',
            'vcu': 'pu'
        })
        self._zeros.extend(['T2', 'T4', 'T6', 'A2', 'A4'])
        self.calls.update({
            'fcall': True,
            'fxcall': False,
            'jac0': True,
            'init1': True,
            'gycall': False,
            'gcall': True
        })
        self.param_remove('Vn')
        self.param_remove('Sn')
        self._init()

    def servcall(self, dae):
        self.copy_data_ext('AVR', 'syn', 'syn', self.avr)
        self.copy_data_ext('AVR', 'u', 'uavr', self.avr)
        self.copy_data_ext('Synchronous', 'bus', 'bus', self.syn)
        self.copy_data_ext('Synchronous', 'u', 'usyn', self.syn)
        self.copy_data_ext('Synchronous', 'Sn', 'Sg', self.syn)
        self.copy_data_ext('Synchronous', 'omega', 'omega', self.syn)
        self.copy_data_ext('Synchronous', 'v', 'v', self.syn)
        self.copy_data_ext('Synchronous', 'p', 'p', self.syn)
        self.copy_data_ext('Synchronous', 'pm', 'pm', self.syn)
        self.copy_data_ext('Synchronous', 'vf', 'vf', self.syn)
        self.copy_data_ext('BusFreq', 'w', 'w', self.bus)
        self.u0 = mul(self.u, self.uavr, self.usyn)
        self.H1 = self.A1 + self.A3
        self.H2 = self.A2 + mul(self.A1, self.A3, self.A4)
        self.H3 = mul(self.A1, self.A4) + mul(self.A2, self.A3)
        self.H4 = mul(self.A2, self.A4)
        self.T12 = mul(self.T1, div(1, self.T2))
        self.T34 = mul(self.T3, div(1, self.T4))
        self.T56 = mul(self.T5, div(1, self.T6))
        self.KsT56 = mul(self.Ks, self.T5, div(1, self.T6))
        self.toSg = self.system.mva * div(1, self.Sg)
        self.update_ctrl()

    def update_ctrl(self):
        self.Ic1 = aeqb(self.Ic, 1)
        self.Ic2 = aeqb(self.Ic, 2)
        self.Ic3 = aeqb(self.Ic, 3)
        self.Ic4 = aeqb(self.Ic, 4)
        self.Ic5 = aeqb(self.Ic, 5)
        self.lsmax += mul(aeqb(self.lsmax, 0.0), 9999)
        self.lsmin += mul(aeqb(self.lsmin, 0.0), -9999)

    def init1(self, dae):
        self.servcall(dae)
        dae.y[self.In] = mul(
            self.u0,
            mul(self.Ic1, -1 + dae.x[self.omega]) + mul(
                self.Ic2, -1 + dae.x[self.w]) + mul(self.Ic5, dae.y[self.v]) +
            mul(self.Ic3, dae.y[self.p], self.toSg) + mul(
                self.Ic4, dae.y[self.pm], self.toSg))
        dae.x[self.q0] = dae.y[self.In]
        dae.y[self.v1] = dae.y[self.In]
        dae.x[self.x1] = mul(dae.y[self.v1], 1 - self.T12)
        dae.y[self.v2] = dae.y[self.v1]
        dae.x[self.x2] = mul(dae.y[self.v2], 1 - self.T34)
        dae.y[self.v3] = dae.y[self.v2]
        dae.x[self.x3] = mul(self.KsT56, dae.y[self.v3])

    def gcall(self, dae):
        dae.g[self.In] = mul(
            self.u0,
            -dae.y[self.In] + mul(self.Ic1, -1 + dae.x[self.omega]) + mul(
                self.Ic2, -1 + dae.x[self.w]) + mul(self.Ic5, dae.y[self.v]) +
            mul(self.Ic3, dae.y[self.p], self.toSg) + mul(
                self.Ic4, dae.y[self.pm], self.toSg))
        dae.g[self.v1] = mul(
            self.u0, dae.x[self.q0] - dae.y[self.v1] + mul(
                self.A5, dae.x[self.q1]) + mul(self.A6, dae.x[self.q2]))
        dae.g[self.v2] = mul(
            self.u0,
            dae.x[self.x1] - dae.y[self.v2] + mul(self.T12, dae.y[self.v1]))
        dae.g[self.v3] = mul(
            self.u0,
            dae.x[self.x2] - dae.y[self.v3] + mul(self.T34, dae.y[self.v2]))
        dae.g[self.v4] = mul(
            self.u0,
            -dae.y[self.v4] - dae.x[self.x3] + mul(self.KsT56, dae.y[self.v3]))
        dae.g[self.vss] = mul(self.u0, dae.y[self.v4] - dae.y[self.vss])
        dae.hard_limit(self.vss, self.lsmin, self.lsmax)
        dae.g[self.vst] = mul(self.u0, dae.y[self.vss] - dae.y[self.vst])
        dae.g += spmatrix(
            mul(self.u0, dae.y[self.vst]), self.vf, [0] * self.n, (dae.m, 1),
            'd')

    def fcall(self, dae):
        dae.f[self.q0] = mul(dae.x[self.q1], self.u0)
        dae.f[self.q1] = mul(dae.x[self.q2], self.u0)
        dae.f[self.q2] = mul(dae.x[self.q3], self.u0)
        dae.f[self.q3] = mul(
            self.u0, div(1, self.H4),
            dae.y[self.In] - dae.x[self.q0] - mul(self.H1, dae.x[self.q1]) -
            mul(self.H2, dae.x[self.q2]) - mul(self.H3, dae.x[self.q3]))
        dae.f[self.x1] = mul(
            self.u0, div(1, self.T2),
            -dae.x[self.x1] + mul(dae.y[self.v1], 1 - self.T12))
        dae.f[self.x2] = mul(
            self.u0, div(1, self.T4),
            -dae.x[self.x2] + mul(dae.y[self.v2], 1 - self.T34))
        dae.f[self.x3] = mul(self.u0, div(1, self.T6),
                             -dae.x[self.x3] + mul(self.KsT56, dae.y[self.v3]))

    def jac0(self, dae):
        dae.add_jac(Gy0, mul(self.Ic5, self.u0), self.In, self.v)
        dae.add_jac(Gy0, -self.u0, self.In, self.In)
        dae.add_jac(Gy0, mul(self.Ic4, self.toSg, self.u0), self.In, self.pm)
        dae.add_jac(Gy0, mul(self.Ic3, self.toSg, self.u0), self.In, self.p)
        dae.add_jac(Gy0, -self.u0, self.v1, self.v1)
        dae.add_jac(Gy0, mul(self.T12, self.u0), self.v2, self.v1)
        dae.add_jac(Gy0, -self.u0, self.v2, self.v2)
        dae.add_jac(Gy0, -self.u0, self.v3, self.v3)
        dae.add_jac(Gy0, mul(self.T34, self.u0), self.v3, self.v2)
        dae.add_jac(Gy0, -self.u0, self.v4, self.v4)
        dae.add_jac(Gy0, mul(self.KsT56, self.u0), self.v4, self.v3)
        dae.add_jac(Gy0, self.u0, self.vss, self.v4)
        dae.add_jac(Gy0, -self.u0, self.vss, self.vss)
        dae.add_jac(Gy0, -self.u0, self.vst, self.vst)
        dae.add_jac(Gy0, self.u0, self.vst, self.vss)
        dae.add_jac(Gy0, self.u0, self.vf, self.vst)
        dae.add_jac(Gx0, mul(self.Ic2, self.u0), self.In, self.w)
        dae.add_jac(Gx0, mul(self.Ic1, self.u0), self.In, self.omega)
        dae.add_jac(Gx0, mul(self.A5, self.u0), self.v1, self.q1)
        dae.add_jac(Gx0, self.u0, self.v1, self.q0)
        dae.add_jac(Gx0, mul(self.A6, self.u0), self.v1, self.q2)
        dae.add_jac(Gx0, self.u0, self.v2, self.x1)
        dae.add_jac(Gx0, self.u0, self.v3, self.x2)
        dae.add_jac(Gx0, -self.u0, self.v4, self.x3)
        dae.add_jac(Fx0, self.u0, self.q0, self.q1)
        dae.add_jac(Fx0, self.u0, self.q1, self.q2)
        dae.add_jac(Fx0, self.u0, self.q2, self.q3)
        dae.add_jac(Fx0, -mul(self.H3, self.u0, div(1, self.H4)), self.q3,
                    self.q3)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.H4)), self.q3, self.q0)
        dae.add_jac(Fx0, -mul(self.H2, self.u0, div(1, self.H4)), self.q3,
                    self.q2)
        dae.add_jac(Fx0, -mul(self.H1, self.u0, div(1, self.H4)), self.q3,
                    self.q1)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.T2)), self.x1, self.x1)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.T4)), self.x2, self.x2)
        dae.add_jac(Fx0, -mul(self.u0, div(1, self.T6)), self.x3, self.x3)
        dae.add_jac(Fy0, mul(self.u0, div(1, self.H4)), self.q3, self.In)
        dae.add_jac(Fy0, mul(self.u0, div(1, self.T2), 1 - self.T12), self.x1,
                    self.v1)
        dae.add_jac(Fy0, mul(self.u0, div(1, self.T4), 1 - self.T34), self.x2,
                    self.v2)
        dae.add_jac(Fy0, mul(self.KsT56, self.u0, div(1, self.T6)), self.x3,
                    self.v3)
        dae.add_jac(Gy0, 1e-6, self.In, self.In)
        dae.add_jac(Gy0, 1e-6, self.v1, self.v1)
        dae.add_jac(Gy0, 1e-6, self.v2, self.v2)
        dae.add_jac(Gy0, 1e-6, self.v3, self.v3)
        dae.add_jac(Gy0, 1e-6, self.v4, self.v4)
        dae.add_jac(Gy0, 1e-6, self.vss, self.vss)
        dae.add_jac(Gy0, 1e-6, self.vst, self.vst)
