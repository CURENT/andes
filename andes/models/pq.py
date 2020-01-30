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
        self.vmax = NumParam(default=1.1,
                             info='max voltage before switching to impedance',
                             tex_name=r'v_{max}',
                             )
        self.vmin = NumParam(default=0.9,
                             info='min voltage before switching to impedance',
                             tex_name=r'v_{min}',
                             )

        self.owner = IdxParam(model='Owner', info="owner idx")


class PQ(PQData, Model):
    def __init__(self, system=None, config=None):
        PQData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'StaticLoad'
        # ``tds`` flag is added to retrieve initial voltage (for constant Z or constant I conversion)
        self.flags.update({'pflow': True,
                           'tds': True,
                           })
        self.config.add(OrderedDict((('pq2z', 1),
                                     ('p2p', 0),
                                     ('p2i', 0),  # not in use
                                     ('p2z', 1),
                                     ('q2q', 0),
                                     ('q2i', 0),  # not in use
                                     ('q2z', 1),
                                     )))

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

        self.Req = ConstService(info='Equivalent resistance',
                                v_str='p0 / v0**2',
                                )
        self.Xeq = ConstService(info='Equivalent reactance',
                                v_str='q0 / v0**2',
                                )

        self.vcmp = Limiter(u=self.v,
                            lower=self.vmin,
                            upper=self.vmax,
                            enable=self.config.pq2z,
                            )

        self.a.e_str = "u * ((dae_t <= 0) | (p2p == 1)) * " \
                       "(p0 * vcmp_zi + " \
                       "p0 * vcmp_zl * (v ** 2 / vmin ** 2) + "\
                       "p0 * vcmp_zu * (v ** 2 / vmax ** 2)) + " \
                       "u * ((dae_t > 0) & (p2p != 1)) * " \
                       "(p2p * p0 + p2z * Req * v**2)"

        self.v.e_str = "u * ((dae_t <= 0) | (q2q == 1)) * " \
                       "(q0 * vcmp_zi + " \
                       "q0 * vcmp_zl * (v ** 2 / vmin ** 2) + " \
                       "q0 * vcmp_zu * (v ** 2 / vmax ** 2)) + " \
                       "u * ((dae_t > 0) & (q2q != 1)) * " \
                       "(q2q * q0 + q2z * Xeq * v**2)"

        # Experimental Zone Below
        # self.v_ref = Algeb(info="Voltage reference for PI")
        # self.kp = Service()
        # self.ki = Service()

        # self.kp.e_str = "1"
        # self.ki.e_str = "1"
        # self.pi = PIController(self.v, self.v_ref, self.kp, self.ki,
        #                        info='PI controller for voltage')
