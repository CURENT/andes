import logging
from collections import OrderedDict
from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam, DataParam
from andes.core.var import Algeb, ExtAlgeb
from andes.core.service import BackRef
from andes.core.discrete import SortedLimiter

logger = logging.getLogger(__name__)


class PVData(ModelData):
    def __init__(self):
        super().__init__()
        self.Sn = NumParam(default=100.0, info="Power rating", non_zero=True, tex_name=r'S_n')
        self.Vn = NumParam(default=110.0, info="AC voltage rating", non_zero=True, tex_name=r'V_n')
        self.subidx = DataParam(info='index for generators on the same bus', export=False)
        self.bus = IdxParam(model='Bus', info="idx of the installed bus", mandatory=True)
        self.busr = IdxParam(model='Bus', info="bus idx for remote voltage control")
        self.p0 = NumParam(default=0.0, info="active power set point in system base", tex_name=r'p_0', unit='p.u.')
        self.q0 = NumParam(default=0.0, info="reactive power set point in system base", tex_name=r'q_0',
                           unit='p.u.')

        self.pmax = NumParam(default=999.0, info="maximum active power in system base",
                             tex_name=r'p_{max}', unit='p.u.')
        self.pmin = NumParam(default=-1.0, info="minimum active power in system base",
                             tex_name=r'p_{min}', unit='p.u.')
        self.qmax = NumParam(default=999.0, info="maximim reactive power in system base",
                             tex_name=r'q_{max}', unit='p.u.')
        self.qmin = NumParam(default=-999.0, info="minimum reactive power in system base",
                             tex_name=r'q_{min}', unit='p.u.')

        self.v0 = NumParam(default=1.0, info="voltage set point", tex_name=r'v_0')
        self.vmax = NumParam(default=1.4, info="maximum voltage voltage", tex_name=r'v_{max}')
        self.vmin = NumParam(default=0.6, info="minimum allowed voltage", tex_name=r'v_{min}')
        self.ra = NumParam(default=0.0, info='armature resistance', tex_name='r_a')
        self.xs = NumParam(default=0.3, info='armature reactance', tex_name='x_s')


class SlackData(PVData):
    def __init__(self):
        super().__init__()
        self.a0 = NumParam(default=0.0, info="reference angle set point", tex_name=r'\theta_0')


class PVModel(Model):
    """
    PV generator model (power flow) with q limit and PV-PQ conversion.
    """
    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        self.group = 'StaticGen'
        self.flags.pflow = True
        self.flags.tds = True
        self.flags.tds_init = False

        self.config.add(OrderedDict((('pv2pq', 0),
                                     ('npv2pq', 0),
                                     ('min_iter', 2),
                                     ('err_tol', 0.01),
                                     ('abs_violation', 1),
                                     )))
        self.config.add_extra("_help",
                              pv2pq="convert PV to PQ in PFlow at Q limits",
                              npv2pq="max. # of conversion each iteration, 0 - auto",
                              min_iter="iteration number starting from which to enable switching",
                              err_tol="iteration error below which to enable switching",
                              abs_violation='use absolute (1) or relative (0) limit violation',
                              )

        self.config.add_extra("_alt",
                              pv2pq=(0, 1),
                              npv2pq=">=0",
                              min_iter='int',
                              err_tol='float',
                              abs_violation=(0, 1),
                              )
        self.config.add_extra("_tex",
                              pv2pq="z_{pv2pq}",
                              npv2pq="n_{pv2pq}",
                              min_iter="sw_{iter}",
                              err_tol=r"\epsilon_{tol}"
                              )

        self.SynGen = BackRef()

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta')
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, v_setter=True, tex_name=r'V')

        self.p = Algeb(info='actual active power generation', unit='p.u.', tex_name=r'p', diag_eps=True)
        self.q = Algeb(info='actual reactive power generation', unit='p.u.', tex_name='q', diag_eps=True)

        self.qlim = SortedLimiter(u=self.q, lower=self.qmin, upper=self.qmax,
                                  enable=self.config.pv2pq,
                                  n_select=self.config.npv2pq,
                                  min_iter=self.config.min_iter,
                                  err_tol=self.config.err_tol,
                                  abs_violation=self.config.abs_violation,
                                  )

        # variable initialization equations
        self.v.v_str = 'v0'
        self.p.v_str = 'p0'
        self.q.v_str = 'q0'

        # injections into buses have negative values
        self.a.e_str = "-u * p"
        self.v.e_str = "-u * q"

        # power injection equations g(y) = 0
        self.p.e_str = "u * (p0 - p)"
        self.q.e_str = "u*(qlim_zi * (v0-v) + "\
                       "qlim_zl * (qmin-q) + "\
                       "qlim_zu * (qmax-q))"

        # NOTE: the line below fails at TDS
        # self.q.e_str = "Piecewise((qmin - q, q < qmin), (qmax - q, q > qmax), (v0 - v, True))"


class PV(PVData, PVModel):
    """
    Static PV generator with reactive power limit checking
    and PV-to-PQ conversion.

    `pv2pq = 1` turns on the conversion.
    It starts  from iteration `min_iter` or when the convergence
    error drops below `err_tol`.

    The PV-to-PQ conversion first ranks the reactive violations.
    A maximum number of `npv2pq` PVs above the upper limit, and
    a maximum of `npv2pq` PVs below the lower limit will be
    converted to PQ, which sets the reactive power to `pmax` or
    `pmin`.

    If `pv2pq` is `1` (enabled) and `npv2pq` is `0`, heuristics
    will be used to determine the number of PVs to be converted
    for each iteration.
    """

    def __init__(self, system=None, config=None):
        PVData.__init__(self)
        PVModel.__init__(self, system, config)


class Slack(SlackData, PVModel):
    """
    Slack generator.
    """

    def __init__(self, system=None, config=None):
        SlackData.__init__(self)
        PVModel.__init__(self, system, config)

        self.config.add(OrderedDict((('av2pv', 0),
                                     )))
        self.config.add_extra("_help",
                              av2pv="convert Slack to PV in PFlow at P limits",
                              )
        self.config.add_extra("_alt",
                              av2pv=(0, 1),
                              )
        self.config.add_extra("_tex",
                              av2pv="z_{av2pv}",
                              )
        self.a.v_setter = True
        self.a.v_str = 'a0'

        self.plim = SortedLimiter(u=self.p, lower=self.pmin, upper=self.pmax,
                                  enable=self.config.av2pv)

        self.p.e_str = "u*(plim_zi * (a0-a) + "\
                       "plim_zl * (pmin-p) + "\
                       "plim_zu * (pmax-p))"
