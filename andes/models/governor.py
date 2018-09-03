from cvxopt import matrix, sparse, spmatrix  # NOQA
from cvxopt import mul, div, log, sin, cos  # NOQA

from .base import ModelBase
from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA


class GovernorBase(ModelBase):
    """Turbine governor base class"""

    def __init__(self, system, name):
        super(GovernorBase, self).__init__(system, name)
        self._group = 'Governor'
        self.param_remove('Vn')
        self._data.update({
            'gen': None,
            'pmax': 999.0,
            'pmin': 0.0,
            'R': 0.05,
            'wref0': 1.0,
        })
        self._descr.update({
            'gen': 'Generator index',
            'pmax': 'Maximum turbine output in Syn Sn',
            'pmin': 'Minimum turbine output in Syn Sn',
            'R': 'Speed regulation droop',
            'wref0': 'Initial reference speed',
        })
        self._units.update({
            'pmax': 'pu',
            'pmin': 'pu',
            'wref0': 'pu',
            'R': 'pu'
        })
        self._params.extend(['pmax', 'pmin', 'R', 'wref0'])
        self._algebs.extend(['wref', 'pout'])
        self._fnamey.extend(['\\omega_{ref}', 'P_{out}'])
        self._service.extend(['pm0', 'gain'])
        self._mandatory.extend(['gen', 'R'])
        self._powers.extend(['pmax', 'pmin'])
        self.calls.update({
            'init1': True,
            'gcall': True,
            'fcall': True,
            'jac0': True,
        })

    def base(self):
        if not self.n:
            return
        self.copy_data_ext(
            model='Synchronous', field='Sn', dest='Sn', idx=self.gen)
        super(GovernorBase, self).base()
        self.R = self.system.mva * div(self.R, self.Sn)

    def init1(self, dae):
        self.gain = div(1.0, self.R)

        # values
        self.copy_data_ext(
            model='Synchronous', field='pm0', dest='pm0', idx=self.gen)

        # indices
        self.copy_data_ext(
            model='Synchronous', field='omega', dest='omega', idx=self.gen)
        self.copy_data_ext(
            model='Synchronous', field='pm', dest='pm', idx=self.gen)

        self.init_limit(
            key='pm0', lower=self.pmin, upper=self.pmax, limit=True)
        dae.y[self.wref] = self.wref0
        dae.y[self.pout] = self.pm0

    def gcall(self, dae):
        dae.g[self.pm] += self.pm0 - mul(
            self.u, dae.y[self.pout])  # update the Syn.pm equations
        dae.g[self.wref] = dae.y[self.wref] - self.wref0

    def jac0(self, dae):
        dae.add_jac(Gy0, -self.u, self.pm, self.pout)

        dae.add_jac(Gy0, 1.0, self.wref, self.wref)


class TG1(GovernorBase):
    """Turbine governor model"""

    def __init__(self, system, name):
        super(TG1, self).__init__(system, name)
        self._name = "TG1"
        self._data.update({
            'T3': 0.0,
            'T4': 12.0,
            'T5': 50.0,
            'Tc': 0.56,
            'Ts': 0.1,
        })
        self._params.extend(['T3', 'T4', 'T5', 'Tc', 'Ts'])
        self._descr.update({
            'T3': 'Transient gain time constant',
            'T4': 'Power fraction time constant',
            'T5': 'Reheat time constant',
            'Tc': 'Servo time constant',
            'Ts': 'Governor time constant',
        })
        self._units.update({
            'T3': 's',
            'T4': 's',
            'T5': 's',
            'Tc': 's',
            'Ts': 's'
        })
        self._mandatory.extend(['T5', 'Tc', 'Ts'])
        self._service.extend(['iTs', 'iTc', 'iT5', 'k1', 'k2', 'k3', 'k4'])
        self._states.extend(['xg1', 'xg2', 'xg3'])
        self._fnamex.extend(['x_{g1}', 'x_{g2}', 'x_{g3}'])
        self._algebs.extend(['pin'])
        self._fnamey.extend(['P_{in}'])
        self._init()

    def init1(self, dae):
        super(TG1, self).init1(dae)
        self.iTs = div(1, self.Ts)
        self.iTc = div(1, self.Tc)
        self.iT5 = div(1, self.T5)
        self.k1 = mul(self.T3, self.iTc)
        self.k2 = 1 - self.k1
        self.k3 = mul(self.T4, self.iT5)
        self.k4 = 1 - self.k3

        dae.x[self.xg1] = mul(self.u, self.pm0)
        dae.x[self.xg2] = mul(self.u, self.k2, self.pm0)
        dae.x[self.xg3] = mul(self.u, self.k4, self.pm0)
        dae.y[self.pin] = self.pm0

    def fcall(self, dae):
        dae.f[self.xg1] = mul(self.u, dae.y[self.pin] - dae.x[self.xg1],
                              self.iTs)
        dae.f[self.xg2] = mul(self.u,
                              mul(self.k2, dae.x[self.xg1]) - dae.x[self.xg2],
                              self.iTc)
        dae.f[self.xg3] = mul(
            self.u,
            mul(self.k4, dae.x[self.xg2] + mul(self.k1, dae.x[self.xg1])) -
            dae.x[self.xg3], self.iT5)

    def gcall(self, dae):
        dae.g[self.pin] = self.pm0 + mul(
            self.gain, dae.y[self.wref] - dae.x[self.omega]) - dae.y[self.pin]
        dae.hard_limit(self.pin, self.pmin, self.pmax)

        dae.g[self.pout] = dae.x[self.xg3] + mul(
            self.k3,
            dae.x[self.xg2] + mul(self.k1, dae.x[self.xg1])) - dae.y[self.pout]
        super(TG1, self).gcall(dae)

    def jac0(self, dae):
        super(TG1, self).jac0(dae)
        dae.add_jac(Gy0, -self.u + 1e-6, self.pin, self.pin)

        dae.add_jac(Gx0, -mul(self.u, self.gain), self.pin, self.omega)
        dae.add_jac(Gy0, mul(self.u, self.gain), self.pin, self.wref)

        dae.add_jac(Fx0, -mul(self.u, self.iTs) + 1e-6, self.xg1, self.xg1)
        dae.add_jac(Fy0, mul(self.u, self.iTs), self.xg1, self.pin)

        dae.add_jac(Fx0, mul(self.u, self.k2, self.iTc), self.xg2, self.xg1)
        dae.add_jac(Fx0, -mul(self.u, self.iTc), self.xg2, self.xg2)

        dae.add_jac(Fx0, mul(self.u, self.k4, self.iT5), self.xg3, self.xg2)
        dae.add_jac(Fx0, mul(self.u, self.k4, self.k1, self.iT5), self.xg3,
                    self.xg1)
        dae.add_jac(Fx0, -mul(self.u, self.iT5), self.xg3, self.xg3)

        dae.add_jac(Gx0, self.u, self.pout, self.xg3)
        dae.add_jac(Gx0, mul(self.u, self.k3), self.pout, self.xg2)
        dae.add_jac(Gx0, mul(self.u, self.k3, self.k1), self.pout, self.xg1)
        dae.add_jac(Gy0, -self.u + 1e-6, self.pout, self.pout)


class TG2(GovernorBase):
    """Simplified governor model"""

    def __init__(self, system, name):
        super(TG2, self).__init__(system, name)
        self._name = 'TG2'
        self._data.update({
            'T1': 0.2,
            'T2': 10.0,
        })
        self._descr.update({
            'T1': 'Transient gain time constant',
            'T2': 'Governor time constant',
        })
        self._units.update({'T1': 's', 'T2': 's'})
        self._params.extend(['T1', 'T2'])
        self._service.extend(['T12', 'iT2'])
        self._mandatory.extend(['T2'])
        self._states.extend(['xg'])
        self._fnamex.extend(['x_g'])
        self._init()

    def init1(self, dae):
        super(TG2, self).init1(dae)
        self.T12 = div(self.T1, self.T2)
        self.iT2 = div(1, self.T2)

    def fcall(self, dae):
        dae.f[self.xg] = mul(
            self.iT2,
            mul(self.gain, 1 - self.T12, self.wref0 - dae.x[self.omega]) -
            dae.x[self.xg])

    def gcall(self, dae):
        pm = dae.x[self.xg] + self.pm0 + mul(self.gain, self.T12,
                                             self.wref0 - dae.x[self.omega])
        dae.g[self.pout] = pm - dae.y[self.pout]
        dae.hard_limit(self.pout, self.pmin, self.pmax)
        super(TG2, self).gcall(dae)

    def jac0(self, dae):
        super(TG2, self).jac0(dae)
        dae.add_jac(Fx0, -self.iT2, self.xg, self.xg)
        dae.add_jac(Fx0, -mul(self.iT2, self.gain, 1 - self.T12), self.xg,
                    self.omega)

        dae.add_jac(Gx0, 1.0, self.pout, self.xg)
        dae.add_jac(Gx0, -mul(self.gain, self.T12), self.pout, self.omega)
        dae.add_jac(Gy0, -1.0, self.pout, self.pout)
