import logging
from collections import OrderedDict
from andes.core.model import Model, ModelData
from andes.core.param import IdxParam, NumParam
from andes.core.var import ExtAlgeb
from andes.core.discrete import Limiter
from andes.core.service import ExtService, ConstService
logger = logging.getLogger(__name__)


class PQData(ModelData):
    def __init__(self):
        super().__init__()
        self.bus = IdxParam(model='Bus',
                            info="linked bus idx",
                            mandatory=True,
                            )

        self.Vn = NumParam(default=110,
                           info="AC voltage rating",
                           unit='kV',
                           non_zero=True,
                           tex_name=r'V_n',
                           )
        self.p0 = NumParam(default=0,
                           info='active power load in system base',
                           power=False,
                           tex_name=r'p_0',
                           unit='p.u.',
                           )
        self.q0 = NumParam(default=0,
                           info='reactive power load in system base',
                           power=False,
                           tex_name=r'q_0',
                           unit='p.u.',
                           )
        self.vmax = NumParam(default=1.2,
                             info='max voltage before switching to impedance',
                             tex_name=r'v_{max}',
                             )
        self.vmin = NumParam(default=0.8,
                             info='min voltage before switching to impedance',
                             tex_name=r'v_{min}',
                             )

        self.owner = IdxParam(model='Owner', info="owner idx")


class PQ(PQData, Model):
    """
    PQ load model.

    Implements an automatic pq2z conversion during power flow when the voltage
    is outside [vmin, vmax]. The conversion can be turned off by setting `pq2z`
    to 0 in the Config file.

    Before time-domain simulation, PQ load will be converted to impedance,
    current source, and power source based on the weights in the Config file.

    Weights (p2p, p2i, p2z) corresponds to the weights for constant power,
    constant current and constant impedance. p2p, p2i and p2z must be in
    decimal numbers and sum up exactly to 1. The same rule applies to
    (q2q, q2i, q2z).
    """
    def __init__(self, system=None, config=None):
        PQData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'StaticLoad'
        # ``tds`` flag is needed to retrieve initial voltage (for constant Z/I conversion)
        self.flags.update({'pflow': True,
                           'tds': True,
                           })
        self.config.add(OrderedDict((('pq2z', 1),
                                     ('p2p', 0.0),
                                     ('p2i', 0.0),
                                     ('p2z', 1.0),
                                     ('q2q', 0.0),
                                     ('q2i', 0.0),
                                     ('q2z', 1.0),
                                     )))
        self.config.add_extra("_help",
                              pq2z="pq2z conversion if out of voltage limits",
                              p2p="P constant power percentage for TDS. Must have (p2p+p2i+p2z)=1",
                              p2i="P constant current percentage",
                              p2z="P constant impedance percentage",
                              q2q="Q constant power percentage for TDS. Must have (q2q+q2i+q2z)=1",
                              q2i="Q constant current percentage",
                              q2z="Q constant impedance percentage",
                              )
        self.config.add_extra("_alt",
                              pq2z="(0, 1)",
                              p2p="float",
                              p2i="float",
                              p2z="float",
                              q2q="float",
                              q2i="float",
                              q2z="float",
                              )
        self.config.add_extra("_tex",
                              pq2z="z_{pq2z}",
                              p2p=r"\gamma_{p2p}",
                              p2i=r"\gamma_{p2i}",
                              p2z=r"\gamma_{p2z}",
                              q2q=r"\gamma_{q2q}",
                              q2i=r"\gamma_{q2i}",
                              q2z=r"\gamma_{q2z}",
                              )

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          )

        self.v0 = ExtService(src='v',
                             model='Bus',
                             indexer=self.bus,
                             tex_name='V_0',
                             info='Initial voltage magnitude from power flow'
                             )
        self.a0 = ExtService(src='a',
                             model='Bus',
                             indexer=self.bus,
                             tex_name=r'\theta_0',
                             info='Initial voltage angle from power flow'
                             )

        # Rub, Xub, Rlb, Xlb are constant for both PF and TDS.
        self.Rub = ConstService(info='Equivalent resistance at voltage upper bound',
                                v_str='p0 / vmax**2',
                                tex_name='R_{ub}'
                                )
        self.Xub = ConstService(info='Equivalent reactance at voltage upper bound',
                                v_str='q0 / vmax**2',
                                tex_name='X_{ub}'
                                )
        self.Rlb = ConstService(info='Equivalent resistance at voltage lower bound',
                                v_str='p0 / vmin**2',
                                tex_name='R_{lb}'
                                )
        self.Xlb = ConstService(info='Equivalent reactance at voltage lower bound',
                                v_str='q0 / vmin**2',
                                tex_name='X_{lb}'
                                )

        # Ppf, Qpf, Req, Xeq, Ipeq, Iqeq are only meaningful after initializing TDS
        self.Ppf = ConstService(info='Actual P in power flow',
                                v_str='(p0 * vcmp_zi + Rlb * vcmp_zl * v0**2 + Rub * vcmp_zu * v0**2)',
                                tex_name='P_{pf}')
        self.Qpf = ConstService(info='Actual Q in power flow',
                                v_str='(q0 * vcmp_zi + Xlb * vcmp_zl * v0**2 + Xub * vcmp_zu * v0**2)',
                                tex_name='Q_{pf}')
        self.Req = ConstService(info='Equivalent resistance at steady state',
                                v_str='Ppf / v0**2',
                                tex_name='R_{eq}'
                                )
        self.Xeq = ConstService(info='Equivalent reactance at steady state',
                                v_str='Qpf / v0**2',
                                tex_name='X_{eq}'
                                )
        self.Ipeq = ConstService(info='Equivalent active current source at steady state',
                                 v_str='Ppf / v0',
                                 tex_name='I_{peq}'
                                 )
        self.Iqeq = ConstService(info='Equivalent reactive current source at steady state',
                                 v_str='Qpf / v0',
                                 tex_name='I_{qeq}'
                                 )

        self.vcmp = Limiter(u=self.v,
                            lower=self.vmin,
                            upper=self.vmax,
                            enable=self.config.pq2z,
                            )

        # Note: the "or" condition "|" is not supported in sympy equation strings.
        # They will simply be ignored.

        # To modify P and Q during TDS, use `alter` to set values to `Ppf` and `Qpf`
        # after, before simulation, setting `config.p2p=1` and `config.q2q=1`.

        self.a.e_str = "u * (dae_t <= 0) * " \
                       "(p0 * vcmp_zi + Rlb * vcmp_zl * v**2 + Rub * vcmp_zu * v**2) + " \
                       "u * (dae_t > 0) * " \
                       "(p2p * Ppf + p2i * Ipeq * v + p2z * Req * v**2)"

        self.v.e_str = "u * (dae_t <= 0) * " \
                       "(q0 * vcmp_zi + Xlb * vcmp_zl * v**2 + Xub * vcmp_zu * v**2) + " \
                       "u * (dae_t > 0) * " \
                       "(q2q * Qpf + q2i * Iqeq * v + q2z * Xeq * v**2)"
