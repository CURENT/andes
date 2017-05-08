from cvxopt import matrix, spmatrix
from .base import ModelBase
from ..consts import *
from ..utils.math import *


class Stagen(ModelBase):
    """Static generator base class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'StaticGen'
        self._data.update({'bus': None,
                           'busr': None,
                           'pg': 0.0,
                           'qg': 0.0,
                           'pmax': 1.0,
                           'pmin': 0.0,
                           'qmax': 0.0,
                           'qmin': 0.0,
                           'v0': 1.0,
                           'vmax': 1.4,
                           'vmin': 0.6,
                           'ra': 0.01,
                           'xs': 0.3,
                           })
        self._units.update({'bus': 'na',
                            'busr': 'na',
                            'pg': 'pu',
                            'qg': 'pu',
                            'pmax': 'pu',
                            'pmin': 'pu',
                            'qmax': 'pu',
                            'v0': 'pu',
                            'vmax': 'pu',
                            'vmin': 'pu',
                            })
        self._params.extend(['v0',
                             'pg',
                             'qg',
                             'pmax',
                             'pmin',
                             'qmax',
                             'qmin',
                             'vmax',
                             'vmin',
                             'ra',
                             'xs',
                             ])
        self._descr.update({'bus': 'the idx of the installed bus',
                            'busr': 'the idx of remotely controlled bus',
                            'pg': 'active power set point',
                            'qg': 'reactive power set point',
                            'pmax': 'maximum active power output',
                            'pmin': 'minimum active power output',
                            'qmax': 'maximim reactive power output',
                            'qmin': 'minimum reactive power output',
                            'v0': 'voltage set point',
                            'vmax': 'maximum voltage voltage',
                            'vmin': 'minimum allowed voltage',
                            })
        self._ac = {'bus': ['a', 'v']}
        # self._powers = ['pg', 'qg', 'pmax', 'pmin', 'qmax', 'qmin']
        self._voltages = ['v0', 'vmax', 'vmin']
        self._service = []
        self.calls.update({'gcall': True, 'gycall': True,
                           'init0': True, 'pflow': True,
                           'jac0': True, 'stagen': True,
                           })


class PV(Stagen):
    """Static PV generator for power flow"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'PV'
        self._algebs.extend(['q'])
        self._unamey = ['Q']
        self._fnamey = ['Q']
        self._service.extend(['qlim', 'above', 'below'])
        self._inst_meta()

    def init0(self, dae):
        """Set initial voltage and reactive power for PQ. Overwrites Bus.voltage values"""
        dae.y[self.v] = self.v0
        dae.y[self.q] = mul(self.u, self.qg)

    def gcall(self, dae):
        if self.system.SPF.pv2pq and self.system.SPF.iter >= self.system.SPF.ipv2pq:
            d_min = dae.y[self.q] - self.qmin
            d_max = dae.y[self.q] - self.qmax
            idx_asc = sort_idx(d_min)
            idx_desc = sort_idx(d_max, reverse=True)

            nabove = nbelow = self.system.SPF.npv2pq
            nconv = min(self.system.SPF.npv2pq, self.n)

            for i in range(nconv-1, -1, -1):
                if d_min[idx_asc[i]] >= 0:
                    nbelow -= 1
                if d_max[idx_desc[i]] <= 0:
                    nabove -= 1

            self.below = idx_asc[0:nbelow] if nbelow else []
            self.above = idx_desc[0:nabove] if nabove else []
            mq = matrix(self.q)
            dae.y[mq[self.below]] = self.qmin[self.below]
            dae.y[mq[self.above]] = self.qmax[self.above]
            self.qlim = list(set(list(mq[self.below]) + list(mq[self.above])))

        dae.g -= spmatrix(mul(self.u, self.pg), self.a, [0] * self.n, (dae.m, 1), 'd')
        dae.g -= spmatrix(mul(self.u, dae.y[self.q]), self.v, [0] * self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(mul(self.u, dae.y[self.v] - self.v0), self.q, [0] * self.n, (dae.m, 1), 'd')

        if self.qlim:
            dae.g[self.qlim] = 0

    def gycall(self, dae):
        for q in self.qlim:
            v = self.v[self.q.index(q)]
            self.set_jac('Gy0', 0.0, q, v)
            self.set_jac('Gy0', 0.0, v, q)
            self.set_jac('Gy0', 1.0, q, q)

    def jac0(self, dae):
        dae.set_jac('Gy0', -1e-6, self.v, self.v)
        dae.set_jac('Gy0', -self.u, self.v, self.q)
        dae.set_jac('Gy0', self.u, self.q, self.v)
        dae.set_jac('Gy0', self.u - 1 - 1e-6, self.q, self.q)

    def disable_gen(self, idx):
        """Disable a PV element for TDS"""
        self.u[self.int[idx]] = 0
        self.system.DAE.factorize = True


class Slack(PV):
    """Static slack generator"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'SW'
        self._data.update({'a0': 0.0})
        self._params.extend(['a0'])
        self._units.update({'a0': 'rad'})
        self._descr.update({'a0': 'reference angle'})
        self._algebs.extend(['p'])
        self._unamey.extend(['P'])
        self._fnamey.extend(['P'])
        # self._service.extend(['a0'])
        self.calls.update({'gycall': False
                           })
        self._inst_meta()

    def init0(self, dae):
        super().init0(dae)
        self.a0 = self.system.Bus.angle[self.a]
        dae.y[self.p] = mul(self.u, self.pg)

    def gcall(self, dae):
        dae.g[self.a] -= mul(self.u, dae.y[self.p])
        dae.g[self.v] -= mul(self.u, dae.y[self.q])
        dae.g[self.q] = mul(self.u, dae.y[self.v] - self.v0)
        dae.g[self.p] = mul(self.u, dae.y[self.a] - self.a0)

    def jac0(self, dae):
        super().jac0(dae)
        dae.set_jac('Gy0', -self.u, self.a, self.p)
        dae.set_jac('Gy0', self.u, self.p, self.a)
        dae.set_jac('Gy0', self.u - 1 - 1e-6, self.p, self.p)
