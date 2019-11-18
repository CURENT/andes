import logging
from andes.core.model import Model, ModelData, ModelConfig  # NOQA
from andes.core.param import DataParam, NumParam, ExtParam  # NOQA
from andes.core.var import Algeb, State, ExtAlgeb  # NOQA
from andes.core.limiter import Comparer, SortedLimiter  # NOQA
from andes.core.service import Service  # NOQa
logger = logging.getLogger(__name__)


class PVData(ModelData):
    def __init__(self):
        super().__init__()
        self.Sn = NumParam(default=100.0, info="Power rating", non_zero=True)
        self.Vn = NumParam(default=110.0, info="AC voltage rating", non_zero=True)

        self.bus = DataParam(info="the idx of the installed bus")
        self.busr = DataParam(info="the idx of remotely controlled bus")
        self.p0 = NumParam(default=0.0, info="active power set point", power=True)
        self.q0 = NumParam(default=0.0, info="reactive power set point", power=True)

        self.pmax = NumParam(default=999.0, info="maximum active power output", power=True)
        self.pmin = NumParam(default=-1.0, info="minimum active power output", power=True)
        self.qmax = NumParam(default=999.0, info="maximim reactive power output", power=True)
        self.qmin = NumParam(default=-999.0, info="minimum reactive power output", power=True)

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
    """
    PV generator model (power flow) with q limit and PV-PQ conversion
    """
    def __init__(self, system=None, name=None, config=None):
        super().__init__(system, name, config)

        self.flags.update({'pflow': True,
                           'collate': True})

        self.config.add(pv2pq=1, npv2pq=1)

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus)
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, v_setter=True)

        self.p = Algeb(info='actual active power generation', unit='pu')
        self.q = Algeb(info='actual reactive power generation', unit='pu')

        # TODO: implement switching starting from the second iteration
        self.q_lim = SortedLimiter(var=self.q, lower=self.qmin, upper=self.qmax,
                                   enable=self.config.pv2pq,
                                   n_select=self.config.npv2pq)

        # variable initialization equations
        self.v.v_init = 'v0'
        self.p.v_init = 'p0'
        self.q.v_init = 'q0'

        # injections into buses have negative values
        self.a.e_str = "-u * p"
        self.v.e_str = "-u * q"

        # power injection equations g(y) = 0
        self.p.e_str = "u * (-p + p0)"
        self.q.e_str = "u * (q_lim_zi * (v - v0) + \
                                  q_lim_zl * (q - qmin) + \
                                  q_lim_zu * (q - qmax))"


class PV(PVData, PVModel):
    def __init__(self, system=None, name=None, config=None):
        PVData.__init__(self)
        PVModel.__init__(self, system, name, config)


class Slack(SlackData, PVModel):
    def __init__(self, system=None, name=None, config=None):
        SlackData.__init__(self)
        PVModel.__init__(self, system, name, config)

        self.a.v_setter = True
        self.a.v_init = 'a0'

        self.p_lim = SortedLimiter(var=self.p, lower=self.pmin, upper=self.pmax)

        self.p.e_str = "u * (p_lim_zi * (a - a0) + \
                                  p_lim_zl * (p - pmin) + \
                                  p_lim_zu * (p - pmax))"
