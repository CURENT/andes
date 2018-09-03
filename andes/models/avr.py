from cvxopt import matrix, spmatrix  # NOQA
from cvxopt import mul, div, exp  # NOQA

from .base import ModelBase
from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA


class AVR1(ModelBase):
    """
    Automatic Voltage Regulator Type I, DC exciter simplified from IEEE DC1
    """

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'AVR'
        self._name = 'AVR1'
        self._mandatory.extend(['syn'])
        self._states.extend(['vm', 'vr1', 'vr2', 'vfout'])
        self._fnamex.extend(['v_{m}', 'v_{r1}', 'v_{r2}', 'v_{fout}'])
        self._service.extend(['vref0', 'KfTf', 'usyn', 'u0', 'iTa'])
        self._fnamey.extend(['v_{ref}'])
        self._times.extend(['Tr', 'Te', 'Ta', 'Tf'])
        self._algebs.extend(['vref'])
        self._params.extend([
            'Ka', 'Ke', 'Kf', 'Ta', 'Tf', 'Tr', 'Te', 'vrmax', 'vrmin', 'Ae',
            'Be'
        ])
        self._descr.update({
            'busr': 'Regulated voltage idx',
            'vrmax': 'Maximum regulator voltage',
            'Kf': 'Stabilizer gain',
            'Ka': 'Amplifier gain',
            'syn': 'Generator id',
            'Ta': 'Amplifier time constant',
            'vrmin': 'minimum regulator voltage',
            'Tr': 'Measurement time constant',
            'Be': '2nd ceiling coefficient',
            'Tf': 'Stabilizer time constant',
            'Ae': '1st ceiling coefficient',
            'Ke': 'Field circuit integral deviation',
            'Te': 'Field circuit time constant'
        })
        self._data.update({
            'busr': None,
            'vrmax': 5,
            'Kf': 0.063,
            'Ka': 20,
            'syn': 0,
            'Ta': 0.2,
            'vrmin': -5,
            'Tr': 0.001,
            'Be': 0.9,
            'Tf': 0.35,
            'Ae': 0.0006,
            'Ke': 1,
            'Te': 1.0
        })
        self.calls.update({
            'jac0': True,
            'gycall': False,
            'gcall': True,
            'fcall': True,
            'fxcall': True,
            'init1': True
        })
        self._zeros.extend(['Ta', 'Tr'])
        self._init()

    def servcall(self, dae):
        self.copy_data_ext('Synchronous', 'u', 'usyn', self.syn)
        self.copy_data_ext('Synchronous', 'vf0', 'vf0', self.syn)
        self.copy_data_ext('Synchronous', 'vf', 'vf', self.syn)
        self.copy_data_ext('Synchronous', 'v', 'v', self.syn)
        self.KfTf = mul(self.Kf, div(1, self.Tf))
        self.iTa = div(self.u, self.Ta)
        self.u0 = mul(self.u, self.usyn)

    def init1(self, dae):
        self.servcall(dae)
        dae.x[self.vfout] = dae.y[self.vf]
        self.vref0 = div(mul(dae.y[self.vf], self.Ke + self.Se),
                         self.Ka) + dae.y[self.v]
        dae.y[self.vref] = self.vref0
        dae.x[self.vm] = dae.y[self.v]
        dae.x[self.vr1] = mul(dae.y[self.vf], self.Ke + self.Se)
        dae.x[self.vr2] = -mul(dae.y[self.vf], self.KfTf)
        self.check_limit('vfout', vmin='vrmin', vmax='vrmax')

    def gcall(self, dae):
        dae.g[self.vref] = self.vref0 - dae.y[self.vref]
        dae.g += spmatrix(
            mul(self.u0, self.vf0 - dae.x[self.vfout]), self.vf, [0] * self.n,
            (dae.m, 1), 'd')

    def fcall(self, dae):
        dae.f[self.vm] = mul(div(1, self.Tr), dae.y[self.v] - dae.x[self.vm])
        vr1 = mul(
            self.Ka, dae.y[self.vref] - dae.x[self.vm] - dae.x[self.vr2] - mul(
                self.KfTf, dae.x[self.vfout]))
        dae.f[self.vr1] = div(vr1 - dae.x[self.vr1], self.Ta)
        dae.anti_windup(self.vr1, self.vrmin, self.vrmax)
        dae.f[self.vr2] = mul(
            div(1, self.Tf),
            -dae.x[self.vr2] - mul(self.KfTf, dae.x[self.vfout]))
        dae.f[self.vfout] = mul(
            div(1, self.Te),
            dae.x[self.vr1] - mul(dae.x[self.vfout], self.Ke + self.Se))

    def jac0(self, dae):
        dae.add_jac(Gy0, -1, self.vref, self.vref)
        dae.add_jac(Gx0, -self.u0, self.vf, self.vfout)
        dae.add_jac(Fx0, -div(1, self.Tr), self.vm, self.vm)
        dae.add_jac(Fx0, -mul(self.Ka, self.iTa), self.vr1, self.vm)
        dae.add_jac(Fx0, -mul(self.Ka, self.iTa), self.vr1, self.vr2)
        dae.add_jac(Fx0, -div(1, self.Ta), self.vr1, self.vr1)
        dae.add_jac(Fx0, -mul(self.Ka, self.KfTf, self.iTa), self.vr1,
                    self.vfout)
        dae.add_jac(Fx0, -div(1, self.Tf), self.vr2, self.vr2)
        dae.add_jac(Fx0, -mul(self.KfTf, div(1, self.Tf)), self.vr2,
                    self.vfout)
        dae.add_jac(Fy0, div(1, self.Tr), self.vm, self.v)
        dae.add_jac(Fy0, mul(self.Ka, self.iTa), self.vr1, self.vref)

    def fxcall(self, dae):
        dae.add_jac(Fx, mul(div(1, self.Te), -self.Ke - self.dSe), self.vfout,
                    self.vfout)
        dae.add_jac(Fx, div(self.Ke + self.dSe, self.Te), self.vfout, self.vr1)

    @property
    def Se(self):
        dae = self.system.dae
        vfout = dae.x[self.vfout]
        return mul(self.Ae, exp(mul(self.Be, abs(vfout))))

    @property
    def dSe(self):
        dae = self.system.dae
        vfout = dae.x[self.vfout]
        return mul(self.Ae, exp(mul(self.Be, abs(vfout)))) \
            + mul(self.Ae, self.Be, abs(vfout), exp(mul(self.Be, abs(vfout))))


class AVR2(ModelBase):
    """Automatic Voltage Regulator Type II with fast response and high gain"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'AVR'
        self._name = 'AVR2'
        self._states.extend(['vm', 'vr1', 'vr2', 'vfout'])
        self._algebs.extend(['vref', 'vr'])
        self._service.extend(['T21', 'vref0', 'T43'])
        self._fnamey.extend(['v_{ref}', 'v_r'])
        self._fnamex.extend(['v_{m}', 'v_{r1}', 'v_{r2}', 'v_{fout}'])
        self._times.extend(['T1', 'T2', 'T3', 'T4'])
        self._zeros.extend(['Te', 'Tr', 'T1', 'T3'])
        self._mandatory.extend(['syn'])
        self._params.extend([
            'K0', 'T1', 'T2', 'T3', 'T4', 'Tr', 'Te', 'vrmax', 'vrmin', 'Ae',
            'Be'
        ])
        self._data.update({
            'K0': 200,
            'Ae': 0.0006,
            'T3': 0.05,
            'Te': 1.0,
            'syn': 0,
            'vrmin': -5,
            'vrmax': 5,
            'Tr': 0.001,
            'T1': 0.01,
            'T4': 1,
            'Be': 0.9,
            'T2': 0.1
        })
        self._descr.update({
            'K0': 'Regulator gain',
            'vrmin': 'minimum regulator voltage',
            'Ae': '1st ceiling coefficient',
            'Tr': 'Measurement time constant',
            'syn': 'Generator id',
            'T1': 'Regulator zero',
            'Be': '2nd ceiling coefficient',
            'Te': 'Field circuit time constant',
            'vrmax': 'Maximum regulator voltage',
            'T2': 'Regulator Pole'
        })
        self.calls.update({
            'fcall': True,
            'init1': True,
            'gcall': True,
            'fxcall': True,
            'gycall': False,
            'jac0': True
        })
        self._init()

    def servcall(self, dae):
        self.copy_data_ext('Synchronous', 'v', 'v', self.syn)
        self.copy_data_ext('Synchronous', 'u', 'usyn', self.syn)
        self.copy_data_ext('Synchronous', 'v', 'v', self.syn)
        self.copy_data_ext('Synchronous', 'vf', 'vf', self.syn)
        self.copy_data_ext('Synchronous', 'vf0', 'vf0', self.syn)
        self.T43 = mul(self.T4, div(1, self.T3))
        self.T21 = mul(self.T2, div(1, self.T1))
        self.u0 = mul(self.u, self.usyn)

    def init1(self, dae):
        self.servcall(dae)
        dae.x[self.vfout] = dae.y[self.vf]
        self.vref0 = dae.y[self.v] + mul(dae.y[self.vf], div(1, self.K0),
                                         1 + self.Se)
        dae.y[self.vref] = self.vref0
        dae.x[self.vm] = dae.y[self.v]
        dae.x[self.vr1] = mul(self.K0, 1 - self.T21,
                              self.vref0 - dae.y[self.v])
        dae.y[self.vr] = mul(dae.y[self.vf], 1 + self.Se)
        dae.x[self.vr2] = mul(
            div(1, self.K0), 1 - self.T43, dae.x[self.vr1] + mul(
                self.K0, self.T21, dae.y[self.vref] - dae.x[self.vm]))

    def gcall(self, dae):
        dae.g[self.vref] = self.vref0 - dae.y[self.vref]
        dae.g[self.vr] = -dae.y[self.vr] + mul(self.K0, dae.x[self.vr2]) + mul(
            self.T43, dae.x[self.vr1] + mul(self.K0, self.T21,
                                            dae.y[self.vref] - dae.x[self.vm]))
        dae.hard_limit(self.vr, self.vrmin, self.vrmax)
        dae.g += spmatrix(
            mul(self.u0, self.vf0 - dae.x[self.vfout]), self.vf, [0] * self.n,
            (dae.m, 1), 'd')

    def fcall(self, dae):
        dae.f[self.vm] = mul(div(1, self.Tr), dae.y[self.v] - dae.x[self.vm])
        dae.f[self.vr1] = mul(
            div(1, self.T1), -dae.x[self.vr1] + mul(
                self.K0, 1 - self.T21, dae.y[self.vref] - dae.x[self.vm]))
        dae.f[self.vr2] = mul(
            div(1, self.K0), div(1, self.T3),
            mul(
                1 - self.T43, dae.x[self.vr1] + mul(
                    self.K0, self.T21, dae.y[self.vref] - dae.x[self.vm])) -
            mul(self.K0, dae.x[self.vr2]))
        dae.f[self.vfout] = mul(
            div(1, self.Te),
            dae.y[self.vr] - mul(dae.x[self.vfout], 1 + self.Se))

    def jac0(self, dae):
        dae.add_jac(Gy0, -1, self.vref, self.vref)
        dae.add_jac(Gy0, mul(self.K0, self.T21, self.T43), self.vr, self.vref)
        dae.add_jac(Gy0, -1, self.vr, self.vr)
        dae.add_jac(Gx0, self.T43, self.vr, self.vr1)
        dae.add_jac(Gx0, self.K0, self.vr, self.vr2)
        dae.add_jac(Gx0, -mul(self.K0, self.T21, self.T43), self.vr, self.vm)
        dae.add_jac(Gx0, -self.u0, self.vf, self.vfout)
        dae.add_jac(Fx0, -div(1, self.Tr), self.vm, self.vm)
        dae.add_jac(Fx0, -div(1, self.T1), self.vr1, self.vr1)
        dae.add_jac(Fx0, -mul(self.K0, div(1, self.T1), 1 - self.T21),
                    self.vr1, self.vm)
        dae.add_jac(Fx0, mul(div(1, self.K0), div(1, self.T3), 1 - self.T43),
                    self.vr2, self.vr1)
        dae.add_jac(Fx0, -div(1, self.T3), self.vr2, self.vr2)
        dae.add_jac(Fx0, -mul(self.T21, div(1, self.T3), 1 - self.T43),
                    self.vr2, self.vm)
        dae.add_jac(Fy0, div(1, self.Tr), self.vm, self.v)
        dae.add_jac(Fy0, mul(self.K0, div(1, self.T1), 1 - self.T21), self.vr1,
                    self.vref)
        dae.add_jac(Fy0, mul(self.T21, div(1, self.T3), 1 - self.T43),
                    self.vr2, self.vref)

    def fxcall(self, dae):
        dae.add_jac(Fy, div(1 + self.dSe, self.Te), self.vfout, self.vr)
        dae.add_jac(Fx, -div(1 + self.dSe, self.Te), self.vfout, self.vfout)

    @property
    def Se(self):
        dae = self.system.dae
        vfout = dae.x[self.vfout]
        return mul(self.Ae, exp(mul(self.Be, abs(vfout))))

    @property
    def dSe(self):
        dae = self.system.dae
        vfout = dae.x[self.vfout]
        return mul(self.Ae, exp(mul(self.Be, abs(vfout)))) \
            + mul(self.Ae, self.Be, abs(vfout), exp(mul(self.Be, abs(vfout))))


class AVR3(ModelBase):
    """Automatic Voltage Regulator Type III"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'AVR'
        self._name = 'AVR3'
        self._service.extend(['v0', 'vref0', 'T1T2'])
        self._params.extend(
            ['T1', 'T2', 'Tr', 'Te', 'vfmax', 'vfmin', 'K0', 's0'])
        self._algebs.extend(['vref'])
        self._times.extend(['T2', 'Te', 'Tr'])
        self._fnamex.extend(['v_{m}', 'v_{r}', 'v_{fout}'])
        self._mandatory.extend(['syn'])
        self._states.extend(['vm', 'vr', 'vfout'])
        self._fnamey.extend(['v_{ref}'])
        self._units.update({
            'T1': 's',
            'T2': 's',
            'Te': 's',
            'Tr': 's',
            'vfmax': 'pu',
            'vfmin': 'pu'
        })
        self._data.update({
            'T1': 0.01,
            'T2': 0.1,
            'Tr': 0.001,
            'K0': 20,
            'vfmax': 5,
            'Te': 1.0,
            'syn': 0,
            'vfmin': -5,
            's0': True
        })
        self._descr.update({
            'T1': 'Regulator zero',
            'T2': 'Regularot Pole',
            'Tr': 'Measurement time constant',
            'K0': 'Regulator gain',
            'vfmax': 'Maximum field voltage',
            'Te': 'Field circuit time constant',
            'syn': 'Generator id',
            'vfmin': 'Minimum field voltage',
            's0': 'Enable excitation voltage feedback'
        })
        self.calls.update({
            'fxcall': True,
            'gycall': False,
            'fcall': True,
            'gcall': True,
            'init1': True,
            'jac0': True
        })
        self._init()

    def servcall(self, dae):
        self.copy_data_ext('Synchronous', 'u', 'usyn', self.syn)
        self.copy_data_ext('Synchronous', 'v', 'v', self.syn)
        self.copy_data_ext('Synchronous', 'vf', 'vf', self.syn)
        self.copy_data_ext('Synchronous', 'vf0', 'vf0', self.syn)
        self.T1T2 = mul(self.T1, div(1, self.T2))
        self.iTe = div(self.u, self.Te)
        self.u0 = mul(self.u, self.usyn)

    def init1(self, dae):
        self.servcall(dae)
        self.v0 = dae.y[self.v]
        self.vref0 = dae.y[self.v]
        dae.y[self.vref] = dae.y[self.v]
        dae.x[self.vfout] = dae.y[self.vf]
        dae.x[self.vm] = dae.y[self.v]

    def gcall(self, dae):
        dae.g[self.vref] = self.vref0 - dae.y[self.vref]
        dae.g += spmatrix(
            mul(self.u0, self.vf0 - dae.x[self.vfout]), self.vf, [0] * self.n,
            (dae.m, 1), 'd')

    def fcall(self, dae):
        dae.f[self.vm] = mul(div(1, self.Tr), dae.y[self.v] - dae.x[self.vm])
        dae.f[self.vr] = mul(
            div(1, self.T2), -dae.x[self.vr] + mul(
                self.K0, 1 - self.T1T2, dae.y[self.vref] - dae.x[self.vm]))
        vfout = mul(
            1 + mul(self.s0, -1 + mul(dae.y[self.v], div(1, self.v0))),
            self.vf0 + dae.x[self.vr] + mul(self.K0, self.T1T2,
                                            dae.y[self.vref] - dae.x[self.vm]))
        dae.f[self.vfout] = div(vfout - dae.x[self.vfout], self.Te)
        dae.anti_windup(self.vfout, self.vfmin, self.vfmax)

    def fxcall(self, dae):
        dae.add_jac(
            Fx,
            -mul(self.iTe, self.K0, self.T1T2,
                 1 + mul(self.s0, -1 + mul(dae.y[self.v], div(1, self.v0)))),
            self.vfout, self.vm)
        dae.add_jac(
            Fx, self.iTe + mul(self.iTe, self.s0,
                               -1 + mul(dae.y[self.v], div(1, self.v0))),
            self.vfout, self.vr)
        dae.add_jac(
            Fy,
            mul(self.iTe, self.K0, self.T1T2,
                1 + mul(self.s0, -1 + mul(dae.y[self.v], div(1, self.v0)))),
            self.vfout, self.vref)
        dae.add_jac(
            Fy,
            mul(
                self.iTe, self.s0, div(1, self.v0),
                self.vf0 + dae.x[self.vr] + mul(
                    self.K0, self.T1T2, dae.y[self.vref] - dae.x[self.vm])),
            self.vfout, self.v)

    def jac0(self, dae):
        dae.add_jac(Gy0, -1, self.vref, self.vref)
        dae.add_jac(Gx0, -self.u0, self.vf, self.vfout)
        dae.add_jac(Fx0, -div(1, self.Tr), self.vm, self.vm)
        dae.add_jac(Fx0, -mul(self.K0, div(1, self.T2), 1 - self.T1T2),
                    self.vr, self.vm)
        dae.add_jac(Fx0, -div(1, self.T2), self.vr, self.vr)
        dae.add_jac(Fy0, div(1, self.Tr), self.vm, self.v)
        dae.add_jac(Fy0, mul(self.K0, div(1, self.T2), 1 - self.T1T2), self.vr,
                    self.vref)
        dae.add_jac(Fx0, -div(1, self.Te), self.vfout, self.vfout)
