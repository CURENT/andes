import logging

from cvxopt import matrix, mul

from .base import ModelBase
from ..utils.math import sort_idx
from andes.models.base import Model, ModelData  # NOQA
from andes.core.param import DataParam, NumParam, ExtParam  # NOQA
from andes.core.var import Algeb, State, ExtAlgeb  # NOQA
from andes.core.limiter import Comparer, OrderedLimiter  # NOQA
logger = logging.getLogger(__name__)


class Stagen(ModelBase):
    """Static generator base class"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'StaticGen'
        self._data.update({
            'bus': None,
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
        self._units.update({
            'bus': 'na',
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
        self._params.extend([
            'v0',
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
        self._descr.update({
            'bus': 'the idx of the installed bus',
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
        self.calls.update({
            'gcall': True,
            'gycall': True,
            'init0': True,
            'pflow': True,
            'jac0': True,
            'stagen': True,
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
        self._init()

    def init0(self, dae):
        """
        Set initial voltage and reactive power for PQ.
        Overwrites Bus.voltage values
        """
        dae.y[self.v] = self.v0
        dae.y[self.q] = mul(self.u, self.qg)

    def gcall(self, dae):
        update_qlim = False
        if self.system.pflow.config.pv2pq:
            pflow = self.system.pflow
            if pflow.niter >= pflow.config.ipv2pq:
                update_qlim = True
            elif len(pflow.iter_mis) > 0 and pflow.iter_mis[-1] <= min(0.01, 1e4 * pflow.config.tol):
                update_qlim = True

        qlim_new = []
        if update_qlim is True:
            d_min = dae.y[self.q] - self.qmin
            d_max = dae.y[self.q] - self.qmax
            idx_asc = sort_idx(d_min)
            idx_desc = sort_idx(d_max, reverse=True)

            nabove = nbelow = int(self.system.pflow.config.npv2pq)
            nconv = int(min(self.system.pflow.config.npv2pq, self.n))

            for i in range(nconv - 1, -1, -1):
                if d_min[idx_asc[i]] >= 0:
                    nbelow -= 1
                if d_max[idx_desc[i]] <= 0:
                    nabove -= 1

            self.below = idx_asc[0:nbelow] if nbelow else []
            self.above = idx_desc[0:nabove] if nabove else []
            mq = matrix(self.q)
            dae.y[mq[self.below]] = self.qmin[self.below]
            dae.y[mq[self.above]] = self.qmax[self.above]
            qlim_new = list(set(list(mq[self.below]) + list(mq[self.above])))

            if qlim_new:
                # refactorize the DAE when limit is hit. It allow resetting the Jacobian elements
                self.system.dae.factorize = True

        self.qlim = list(set(list(self.qlim) + qlim_new))

        p_inj = -mul(self.u, self.pg)
        q_inj = -mul(self.u, dae.y[self.q])
        v_mis = mul(self.u, dae.y[self.v] - self.v0)

        # TODO: improve readability
        for a, v, q, pi, qi, dv in zip(self.a, self.v, self.q, p_inj, q_inj,
                                       v_mis):
            dae.g[a] += pi
            dae.g[v] += qi
            dae.g[q] = dv  # not an interface equation

        if self.qlim:
            dae.g[self.qlim] = 0

    def gycall(self, dae):
        for q in self.qlim:
            v = self.v[self.q.index(q)]
            dae.set_jac('Gy', 0.0, q, v)
            dae.set_jac('Gy', 0.0, v, q)
            dae.set_jac('Gy', 1.0, q, q)

    def jac0(self, dae):
        dae.set_jac('Gy0', -1e-6, self.v, self.v)
        dae.set_jac('Gy0', -self.u, self.v, self.q)
        dae.set_jac('Gy0', self.u, self.q, self.v)
        dae.set_jac('Gy0', -1e-6, self.q, self.q)

    def disable_gen(self, idx):
        """
        Disable a PV element for TDS

        Parameters
        ----------
        idx

        Returns
        -------

        """
        self.u[self.uid[idx]] = 0
        self.system.dae.factorize = True


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
        self.calls.update({'gycall': False})
        self._init()

    def init0(self, dae):
        super().init0(dae)
        self.a0 = self.system.Bus.angle[self.a]
        dae.y[self.p] = mul(self.u, self.pg)

    def gcall(self, dae):
        p_inj = -mul(self.u, dae.y[self.p])
        q_inj = -mul(self.u, dae.y[self.q])
        v_mis = mul(self.u, dae.y[self.v] - self.v0)
        a_mis = mul(self.u, dae.y[self.a] - self.a0)
        for a, v, p, q, pi, qi, dv, da in zip(self.a, self.v, self.p, self.q,
                                              p_inj, q_inj, v_mis, a_mis):
            dae.g[a] += pi
            dae.g[v] += qi
            dae.g[q] = dv
            dae.g[p] = da

        # dae.g[self.a] -= mul(self.u, dae.y[self.p])
        # dae.g[self.v] -= mul(self.u, dae.y[self.q])
        # dae.g[self.q] = mul(self.u, dae.y[self.v] - self.v0)
        # dae.g[self.p] = mul(self.u, dae.y[self.a] - self.a0)

    def jac0(self, dae):
        super().jac0(dae)
        dae.set_jac('Gy0', -self.u, self.a, self.p)
        dae.set_jac('Gy0', self.u, self.p, self.a)
        dae.set_jac('Gy0', 1e-6, self.p, self.p)


class PVData(ModelData):
    def __init__(self):
        super().__init__()
        self.Sn = NumParam(default=100.0, info="Power rating", non_zero=True)
        self.Vn = NumParam(default=110.0, info="AC voltage rating", non_zero=True)

        self.bus = DataParam(info="the idx of the installed bus")
        self.busr = DataParam(info="the idx of remotely controlled bus")
        self.p0 = NumParam(default=0.0, info="active power set point", power=True)
        self.q0 = NumParam(default=0.0, info="reactive power set point", power=True)

        self.pmax = NumParam(default=1.0, info="maximum active power output")
        self.pmin = NumParam(default=0.0, info="minimum active power output")
        self.qmax = NumParam(default=0.0, info="maximim reactive power output")
        self.qmin = NumParam(default=0.0, info="minimum reactive power output")

        self.v0 = NumParam(default=1.0, info="voltage set point")
        self.vmax = NumParam(default=1.4, info="maximum voltage voltage")
        self.vmin = NumParam(default=0.6, info="minimum allowed voltage")
        self.ra = NumParam(default=0.01, info='armature resistance')
        self.xs = NumParam(default=0.3, info='armature reactance')


class SlackData(PVData):
    def __init__(self):
        super().__init__()
        self.a0 = NumParam(default=0.0, info="reference angle set point")


class PVModel(Model):
    def __init__(self, system=None, name=None):
        super().__init__(system, name)

        self.flags['pflow'] = True

        self.a = ExtAlgeb(model='BusNew', src='a', indexer=self.bus)
        self.v = ExtAlgeb(model='BusNew', src='v', indexer=self.bus, v_setter=True)

        self.p = Algeb(info='actual active power generation', unit='pu')
        self.q = Algeb(info='actual reactive power generation', unit='pu')

        self.q_lim = OrderedLimiter(var=self.q, lower=self.qmin, upper=self.qmax,
                                    n_select=2)

        # initialization equations
        self.v.v_init = 'v0'
        self.p.v_init = 'p0'
        self.q.v_init = 'q0'

        # injections into buses have negative values
        self.a.e_symbolic = "-u * p"
        self.v.e_symbolic = "-u * q"

        self.p.e_symbolic = "u * (-p + p0)"
        self.q.e_symbolic = "u * (q_lim_zi * (v - v0) + \
                                  q_lim_zl * (q - qmin) + \
                                  q_lim_zu * (q - qmax))"


class PVNew(PVData, PVModel):
    def __init__(self, system=None, name=None):
        PVData.__init__(self)
        PVModel.__init__(self, system, name)


class SlackNew(SlackData, PVModel):
    def __init__(self, system=None, name=None):
        SlackData.__init__(self)
        PVModel.__init__(self, system, name)

        self.a.v_init = 'a0'

        self.p_lim = OrderedLimiter(var=self.p, lower=self.pmin, upper=self.pmax)

        self.p.e_symbolic = "u * (p_lim_zi * (a - a0) + \
                                  p_lim_zl * (p - pmin) + \
                                  p_lim_zu * (p - pmax))"
