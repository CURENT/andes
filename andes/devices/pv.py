import logging
from collections import OrderedDict
from andes.core.model import Model, ModelData  # NOQA
from andes.core.param import NumParam, IdxParam # NOQA
from andes.core.var import Algeb, ExtAlgeb  # NOQA
from andes.core.limiter import SortedLimiter  # NOQA

logger = logging.getLogger(__name__)


class PVData(ModelData):
    def __init__(self):
        super().__init__()
        self.Sn = NumParam(default=100.0, info="Power rating", non_zero=True, tex_name=r'S_n')
        self.Vn = NumParam(default=110.0, info="AC voltage rating", non_zero=True, tex_name=r'V_n')

        self.bus = IdxParam(model='Bus', info="the idx of the installed bus")
        self.busr = IdxParam(model='Bus', info="the idx of remotely controlled bus")
        self.p0 = NumParam(default=0.0, info="active power set point", power=True, tex_name=r'p_0')
        self.q0 = NumParam(default=0.0, info="reactive power set point", power=True, tex_name=r'q_0')

        self.pmax = NumParam(default=999.0, info="maximum active power output", power=True, tex_name=r'p_{max}')
        self.pmin = NumParam(default=-1.0, info="minimum active power output", power=True, tex_name=r'p_{min}')
        self.qmax = NumParam(default=999.0, info="maximim reactive power output", power=True, tex_name=r'q_{max}')
        self.qmin = NumParam(default=-999.0, info="minimum reactive power output", power=True, tex_name=r'q_{min}')

        self.v0 = NumParam(default=1.0, info="voltage set point", tex_name=r'v_0')
        self.vmax = NumParam(default=1.4, info="maximum voltage voltage", tex_name=r'v_{max}')
        self.vmin = NumParam(default=0.6, info="minimum allowed voltage", tex_name=r'v_{min}')
        self.ra = NumParam(default=0.01, info='armature resistance', tex_name='r_a')
        self.xs = NumParam(default=0.3, info='armature reactance', tex_name='x_s')


class SlackData(PVData):
    def __init__(self):
        super().__init__()
        self.a0 = NumParam(default=0.0, info="reference angle set point", tex_name=r'\theta_0')


class PVModel(Model):
    """
    PV generator model (power flow) with q limit and PV-PQ conversion
    """
    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.group = 'StaticGen'
        self.flags.update({'pflow': True,
                           'collate': True})

        self.config.add(OrderedDict((('pv2pq', 1),
                                     ('npv2pq', 1))))

        self.tex_names.update({'qlim_zi': 'z_{qi}',
                               'qlim_zl': 'z_{ql}',
                               'qlim_zu': 'z_{qu}'})

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta')
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, v_setter=True, tex_name=r'V')

        self.p = Algeb(info='actual active power generation', unit='pu', tex_name=r'p')
        self.q = Algeb(info='actual reactive power generation', unit='pu', tex_name='q')

        # TODO: implement switching starting from the second iteration
        self.qlim = SortedLimiter(var=self.q, lower=self.qmin, upper=self.qmax,
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
        self.p.e_str = "u * (p0 - p)"
        self.q.e_str = "u * (qlim_zi * (v0 - v) + \
                             qlim_zl * (qmin - q) + \
                             qlim_zu * (qmax - q))"


class PV(PVData, PVModel):
    def __init__(self, system=None, config=None):
        PVData.__init__(self)
        PVModel.__init__(self, system, config)


class Slack(SlackData, PVModel):
    def __init__(self, system=None, config=None):
        SlackData.__init__(self)
        PVModel.__init__(self, system, config)

        self.tex_names.update({'plim_zi': 'z_{pi}',
                               'plim_zl': 'z_{pl}',
                               'plim_zu': 'z_{pu}'})
        self.a.v_setter = True
        self.a.v_init = 'a0'

        self.plim = SortedLimiter(var=self.p, lower=self.pmin, upper=self.pmax)

        self.p.e_str = "u * (plim_zi * (a0 - a) + \
                             plim_zl * (pmin - p) + \
                             plim_zu * (pmax - p))"
