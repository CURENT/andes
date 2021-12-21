"""
HYGOV hydro governor model.
"""

from andes.core import Algeb, ConstService, NumParam, State
from andes.core.block import Integrator, Lag, DeadBand1, AntiWindupRate
from andes.core.service import VarService
from andes.models.governor.tgbase import TGBase, TGBaseData


class HYGOVData(TGBaseData):
    """
    HYGOV data
    """

    def __init__(self):
        super().__init__()
        self.R = NumParam(info='Speed regulation gain (mach. base default)',
                          tex_name='R',
                          default=0.05,
                          unit='p.u.',
                          ipower=True,
                          )
        self.r = NumParam(info='Temporary droop (R<r)',
                          tex_name='r',
                          default=1,
                          unit='p.u.',
                          ipower=True,
                          )
        self.GMAX = NumParam(info='Maximum governor response',
                             tex_name='G_{max}',
                             unit='p.u.',
                             default=1.0,
                             power=True,
                             )
        self.GMIN = NumParam(info='Minimum governor response',
                             tex_name='G_{min}',
                             unit='p.u.',
                             default=0.0,
                             power=True,
                             )
        self.VELM = NumParam(info='Gate velocity limit',
                             tex_name='VELM',
                             unit='p.u.',
                             default=0.3,
                             power=True,
                             )

        self.Tf = NumParam(info='Filter time constant',
                           default=0.05,
                           tex_name='T_f')
        self.Tr = NumParam(info='Governor time constant',
                           default=1,
                           tex_name='T_r')
        self.Tg = NumParam(info='Servo time constant',
                           default=0.05,
                           tex_name='T_g')
        self.Dt = NumParam(info='Turbine damping coefficient',
                           default=0.0,
                           tex_name='D_t',
                           power=True,
                           )
        self.qNL = NumParam(info='No-load flow at nominal head',
                            default=0.1,
                            tex_name='q_{NL}',
                            power=True,
                            )
        self.Tw = NumParam(info='Water inertia time constant constant',
                           default=1,
                           tex_name='T_w')
        self.At = NumParam(info='Turbine gain',
                           default=1,
                           tex_name='A_t')

        # nonlinear points, keep for future use
        # self.GV0 = NumParam(info='Governor point 0',
        #                     default=0.1,
        #                     tex_name='GV_0')
        # self.PGV0 = NumParam(info='Governor power point 0',
        #                      default=0,
        #                      tex_name='PGV_0')
        # self.GV1 = NumParam(info='Governor point 1',
        #                     default=0.15,
        #                     tex_name='GV_1')
        # self.PGV1 = NumParam(info='Governor power point 1',
        #                      default=0.1,
        #                      tex_name='PGV_1')
        # self.GV2 = NumParam(info='Governor point 2',
        #                     default=0.3,
        #                     tex_name='GV_2')
        # self.PGV2 = NumParam(info='Governor power point 2',
        #                      default=0.5,
        #                      tex_name='PGV_2')
        # self.GV3 = NumParam(info='Governor point 3',
        #                     default=0.5,
        #                     tex_name='GV_3')
        # self.PGV3 = NumParam(info='Governor power point 3',
        #                      default=0.7,
        #                      tex_name='PGV_3')
        # self.GV4 = NumParam(info='Governor point 4',
        #                     default=0.6,
        #                     tex_name='GV_4')
        # self.PGV4 = NumParam(info='Governor power point 4',
        #                      default=0.8,
        #                      tex_name='PGV_4')
        # self.GV5 = NumParam(info='Governor point 5',
        #                     default=0.7,
        #                     tex_name='GV_5')
        # self.PGV5 = NumParam(info='Governor power point 5',
        #                      default=0.9,
        #                      tex_name='PGV_5')


class HYGOVDBData(HYGOVData):
    """
    HYGOVDB data.
    """

    def __init__(self):
        HYGOVData.__init__(self)
        self.dbL = NumParam(info='Lower bound of deadband',
                            tex_name='db_L',
                            default=0.0,
                            unit='p.u.',
                            )
        self.dbU = NumParam(info='Upper bound of deadband',
                            tex_name='db_U',
                            default=0.0,
                            unit='p.u.',
                            )


class HYGOVModel(TGBase):
    """
    Implement HYGOV model.

    The input lead-lag filter is ignored.

    The ``g`` signal (LAG) is initialized to the initial value of
    ``q`` (Integrator) to simplify the initializaiton equations.
    """

    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        self.VELMn = ConstService(v_str='-VELM',
                                  tex_name='-VELM',
                                  )
        self.tr = ConstService(v_str='r * Tr',
                               tex_name='r*Tr',
                               )
        self.gr = ConstService(v_str='1/r',
                               tex_name='1/r',
                               )
        self.ratel = ConstService(v_str='- VELM - gr',
                                  tex_name='rate_l',
                                  )
        self.rateu = ConstService(v_str='VELM - gr',
                                  tex_name='rate_u',
                                  )
        self.q0 = ConstService(v_str='tm0 / At + qNL',
                               tex_name='q_0',
                               )
        self.pref = Algeb(info='Reference power input',
                          tex_name='P_{ref}',
                          v_str='R * q0',
                          e_str='R * q0 - pref',
                          )

        self.wd = Algeb(info='Generator speed deviation',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        v_str='0',
                        e_str='ue * (omega - wref) - wd',
                        )
        self.pd = Algeb(info='Pref plus speed deviation times gain',
                        unit='p.u.',
                        tex_name="P_d",
                        v_str='0',
                        e_str='ue * (- wd + pref + paux - R * dg) - pd',
                        )

        self.LG = Lag(u=self.pd,
                      K=1,
                      T=self.Tf,
                      info='filter after speed deviation (e)',
                      )

        self.gtpos = State(info='State in gate position (c)',
                           unit='rad',
                           v_str='q0',
                           tex_name=r'\delta',
                           e_str='LG_y')

        self.dgl = VarService(tex_name='dg_{lower}', info='dg lower limit',
                              v_str='- VELM - gr * LG_y',)
        self.dgu = VarService(tex_name='dg_{upper}', info='dg upper limit',
                              v_str='VELM - gr * LG_y',)
        self.dg_lim = AntiWindupRate(u=self.gtpos, lower=self.GMIN, upper=self.GMAX,
                                     rate_lower=self.dgl, rate_upper=self.dgu,
                                     tex_name='lim_{dg}',
                                     info='gate velocity and position limiter',
                                     )

        self.dg = Algeb(info='desired gate (c)',
                        unit='p.u.',
                        tex_name="dg",
                        v_str='q0',
                        e_str='gtpos + gr * LG_y - dg',
                        )

        self.LAG = Lag(u=self.dg,
                       K=1,
                       T=self.Tg,
                       info='gate opening (g)',
                       )
        self.h = Algeb(info='turbine head',
                       unit='p.u.',
                       tex_name="h",
                       e_str='q_y**2 / LAG_y**2 - h',
                       v_str='1',
                       )
        self.q = Integrator(u="1 - q_y**2 / LAG_y**2",
                            T=self.Tw, K=1,
                            y0='q0',
                            check_init=False,
                            info="turbine flow (q)"
                            )

        self.pout.e_str = 'ue * (At * h * (q_y - qNL) - Dt * wd * LAG_y) - pout'


class HYGOVDBModel(HYGOVModel):
    """
    Model HYGOV with deadband.
    """

    def __init__(self, system, config):
        HYGOVModel.__init__(self, system, config)
        self.DB = DeadBand1(u=self.wd, center=0.0, lower=self.dbL,
                            upper=self.dbU, tex_name='DB',
                            info='deadband for speed deviation',
                            )
        self.pd.e_str = 'ue * (- DB_y + pref + paux - R * dg) - pd'


class HYGOV(HYGOVData, HYGOVModel):
    """
    HYGOV turbine governor model.

    Implements the PSS/E HYGOV model without deadband.

    Reference:

    [1] PSSE, Model Library, HYGOV
    """

    def __init__(self, system, config):
        HYGOVData.__init__(self)
        HYGOVModel.__init__(self, system, config)


class HYGOVDB(HYGOVDBData, HYGOVDBModel):
    """
    HYGOV turbine governor model with speed input deadband.
    """

    def __init__(self, system, config):
        HYGOVDBData.__init__(self)
        HYGOVDBModel.__init__(self, system, config)
