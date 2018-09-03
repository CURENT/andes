from cvxopt import matrix, sparse, spmatrix  # NOQA
from cvxopt import mul, div, log, sin, cos  # NOQA
from .base import ModelBase

from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA

from ..utils.math import zeros, ones


class BusFreq(ModelBase):
    """Bus frequency estimation based on angle derivative"""

    def __init__(self, system, name):
        super(BusFreq, self).__init__(system, name)
        self._group = 'Measurement'
        self._name = 'BusFreq'
        self.param_remove('Vn')
        self._data.update({
            'bus': None,
            'Tf': 0.1,
            'Tw': 1.0,
            'Td': 0.5,
        })
        self._params.extend(['Tf', 'Tw', 'Td'])
        self._descr.update({
            'bus':
            'bus idx',
            'Tf':
            'low-pass filter time constant',
            'Tw':
            'washout filter time constant for phase angle',
            'Td':
            'washout filter time constant for frequency',
        })
        self._units.update({
            'bus': 'na',
            'Tf': 'sec',
            'Tw': 'sec',
        })
        self._mandatory.extend(['bus'])
        self._states.extend(['xt', 'w', 'xdw'])
        self._fnamex.extend(['x_\\theta', '\\omega', 'x_\\omega'])
        self._algebs.extend(['dwdt'])
        self._fnamey.extend(['\\frac{d\\omega}{dt}'])

        self.calls.update({
            'init1': True,
            'fcall': True,
            'gcall': True,
            'jac0': True,
        })
        self._service.extend(['iTf', 'iTw', 'iwn', 'a0', 'dw', 'iTd'])
        self._init()

    def init1(self, dae):
        self.copy_data_ext(model='Bus', field='a', dest='a', idx=self.bus)
        self.iTf = div(self.u, self.Tf)
        self.iTd = div(self.u, self.Td)
        self.iTw = div(self.u, self.Tw)
        self.iwn = div(self.u, self.system.wb)
        self.a0 = dae.y[self.a]

        dae.x[self.xt] = zeros(self.n, 1)
        dae.x[self.w] = ones(self.n, 1)
        # dae.y[self.dw] = zeros(self.n, 1)
        dae.x[self.xdw] = self.iTd

    def fcall(self, dae):
        self.dw = -dae.x[self.xt] + mul(self.iwn, self.iTf,
                                        dae.y[self.a] - self.a0)
        dae.f[self.xt] = mul(
            self.iTf,
            mul(self.iwn, self.iTf, dae.y[self.a] - self.a0) - dae.x[self.xt])
        dae.f[self.w] = mul(self.dw + 1 - dae.x[self.w], self.iTw)
        dae.f[self.xdw] = mul(self.iTd,
                              mul(self.iTd, dae.x[self.w]) - dae.x[self.xdw])

    def gcall(self, dae):
        dae.g[self.dwdt] = (
            mul(self.iTd, dae.x[self.w]) - dae.x[self.xdw]) - dae.y[self.dwdt]

    def jac0(self, dae):
        dae.add_jac(Fy0, mul(self.iTf**2, self.iwn), self.xt, self.a)
        dae.add_jac(Fx0, -self.iTf, self.xt, self.xt)

        dae.add_jac(Fx0, -self.iTw, self.w, self.w)

        dae.add_jac(Fx0, -self.iTw, self.w, self.xt)
        dae.add_jac(Fy0, mul(self.iTw, self.iwn, self.iTf), self.w, self.a)

        dae.add_jac(Gx0, self.iTd, self.dwdt, self.w)
        dae.add_jac(Gx0, -1, self.dwdt, self.xdw)

        dae.add_jac(Fx0, self.iTd**2, self.xdw, self.w)
        dae.add_jac(Fx0, -self.iTd, self.xdw, self.xdw)

        dae.add_jac(Gy0, -1, self.dwdt, self.dwdt)


class PMU(ModelBase):
    """Phasor measurement unit described by low-pass filters"""

    def __init__(self, system, name):
        super(PMU, self).__init__(system, name)
        self._group = 'Measurement'
        self._data.update({
            'Tv': 0.1,
            'Ta': 0.1,
            'fn': 60,
            'bus': None,
        })
        self._params.extend(['Tv', 'Ta', 'fn'])
        self._descr.update({
            'Tv': 'Voltage magnitude time constant',
            'Ta': 'Voltage angle time constant',
            'fn': 'Frequency base',
            'bus': 'Bus idx'
        })
        self._states.extend(['vm', 'am'])
        self._fnamex.extend(['V_m', '\\theta_m'])
        self._mandatory.extend(['bus'])
        self._service.extend(['iTv', 'iTa'])
        self._init()
        self.calls.update({
            'fcall': True,
            'jac0': True,
            'init1': True,
        })

    def init1(self, dae):
        self.copy_data_ext(model='Bus', field='a', idx=self.bus)
        self.copy_data_ext(model='Bus', field='v', idx=self.bus)
        self.iTv = div(self.u, self.Tv)
        self.iTa = div(self.u, self.Ta)
        dae.x[self.am] = dae.y[self.a]
        dae.x[self.vm] = dae.y[self.v]

    def fcall(self, dae):
        dae.f[self.vm] = mul(dae.y[self.v] - dae.x[self.vm], self.iTv)
        dae.f[self.am] = mul(dae.y[self.a] - dae.x[self.am], self.iTa)

    def jac0(self, dae):
        dae.add_jac(Fy0, self.iTv, self.vm, self.v)
        dae.add_jac(Fx0, -self.iTv + 1e-6, self.vm, self.vm)

        dae.add_jac(Fy0, self.iTa, self.am, self.a)
        dae.add_jac(Fx0, -self.iTa + 1e-6, self.am, self.am)
