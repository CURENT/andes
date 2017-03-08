from cvxopt import matrix, sparse, spmatrix
from cvxopt import mul, div, log, sin, cos
from .base import ModelBase
from ..consts import *
from ..utils.math import *


class GovernorBase(ModelBase):
    """Turbine governor base class"""
    def __init__(self, system, name):
        super(GovernorBase, self).__init__(system, name)
        self._group = 'Governor'
        self.remove_param('Vn')
        self.remove_param('Sn')
        self._data.update({'gen': None,
                           'pmax': 1.0,
                           'pmin': 0.0,
                           'R': 0.05,
                           'wref0': 1.0,
                           })
        self._descr.update({'gen': 'generator index',
                            'pmax': 'maximum turbine output',
                            'pmin': 'minimum turbine output',
                            'R': 'speed regulation droop',
                            'wref0': 'initial reference speed',
                            })
        self._params.extend(['pmax', 'pmin', 'R', 'wref0'])
        self._algebs.extend(['wref', 'pout'])
        self._fnamey.extend(['\\omega_{ref}', 'P_{order}', 'P_{out}'])
        self._states.extend(['xg'])
        self._fnamex.extend(['x_g'])
        self._service.extend(['pm0', 'gain'])
        self.calls.update({'init1': True, 'gcall': True,
                           'fcall': True, 'jac0': True,
                           })

    def init1(self, dae):
        self.gain = div(1.0, self.R)

        # values
        self.copy_param(model='Synchronous', src='Sn', dest='Sn', fkey=self.gen)
        self.copy_param(model='Synchronous', src='pm0', dest='pm0', fkey=self.gen)

        # indices
        self.copy_param(model='Synchronous', src='omega', dest='omega', fkey=self.gen)
        self.copy_param(model='Synchronous', src='pm', dest='pm', fkey=self.gen)

        self.limit_check(key='pm0', lower=self.pmin, upper=self.pmax, limit=True)
        dae.y[self.wref] = self.wref0
        dae.y[self.pout] = self.pm0

    def gcall(self, dae):
        pin0 = self.pm0 + mul(self.gain, self.wref0 - dae.x[self.omega])
        pin = algeb_limiter(pin0, self.pmin, self.pmax)

        dae.g[self.pm] += mul(self.u, dae.y[self.pout]) - self.pm0
        dae.g[self.wref] += dae.y[self.wref] - self.wref0

    def jac0(self, dae):
        dae.add_jac(Gy0, self.u, self.pm, self.pout)

        dae.add_jac(Gy0, 1.0, self.wref, self.wref)


class TG2(GovernorBase):
    """Simplified governor model"""
    def __init__(self, system, name):
        super(TG2, self).__init__(system, name)
        self._name = 'TG2'
        self._data.update({'T1': 0.2,
                           'T2': 10.0,
                           })
        self._params.extend(['T1', 'T2'])
        self._service.extend(['T12', 'iT2'])
        self._inst_meta()

    def init1(self, dae):
        super(TG2, self).init1(dae)
        self.T12 = div(self.T1, self.T2)
        self.iT2 = div(1, self.T2)

        dae.x[self.xg] = zeros(self.n, 1)

    def fcall(self, dae):
        dae.f[self.xg] = mul(self.iT2, mul(self.gain, 1 - self.T12, self.wref0 - dae.x[self.omega]) - dae.x[self.xg])

    def gcall(self, dae):
        super(TG2, self).gcall(dae)
        pm = dae.x[self.xg] + self.pm0 + mul(self.gain, self.T12, self.wref0 - dae.x[self.omega])
        dae.g[self.pout] = algeb_limiter(pm, self.pmin, self.pmax) - dae.y[self.pout]

    def jac0(self, dae):
        super(TG2, self).jac0(dae)
        dae.add_jac(Fx0, -self.iT2, self.xg, self.xg)
        dae.add_jac(Fx0, -mul(self.iT2, self.gain, 1 - self.T12), self.xg, self.omega)

        dae.add_jac(Gx0, 1.0, self.pout, self.xg)
        dae.add_jac(Gx0, -mul(self.gain, self.T12), self.pout, self.omega)
        dae.add_jac(Gy0, -1.0, self.pout, self.pout)
