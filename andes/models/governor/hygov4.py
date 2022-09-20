"""
HYGOV4 hydro governor model.
"""

from andes.core import Algeb, ConstService, NumParam
from andes.core.block import Integrator, IntegratorAntiWindup, Lag, Washout, GainLimiter
from andes.models.governor.tgbase import TGBase, TGBaseData


class HYGOV4Data(TGBaseData):
    """
    HYGOV4 Data
    """

    def __init__(self):
        super().__init__()
        self.Rperm = NumParam(info='Speed Regulation Gain (mach. base default)',
                              tex_name='R_{perm}',
                              default=0.5,
                              unit='p.u.',
                              ipower=True,
                              )
        self.Rtemp = NumParam(info='Temporary Droop (Rtemp < Rperm)',
                              tex_name='R_{temp}',
                              default=1,
                              unit='p.u.',
                              ipower=True,
                              )
        self.UO = NumParam(info='Maximum Gate opening velocity',
                           tex_name='U_O',
                           default=1,
                           unit='p.u.',
                           power=True,
                           )
        self.UC = NumParam(info='Maximum Gate closing velocity',
                           tex_name='U_C',
                           default=0,
                           unit='p.u.',
                           power=True,
                           )
        self.PMAX = NumParam(info='Maximum Gate opening',
                             tex_name='P_{MAX}',
                             default=1,
                             unit='p.u.',
                             power=True,
                             )
        self.PMIN = NumParam(info='Minimum Gate opening',
                             tex_name='P_{MIN}',
                             default=0,
                             unit='p.u.',
                             power=True,
                             )
        self.Tp = NumParam(info='Pilot servo time constant',
                           default=0.05,
                           tex_name='T_p'
                           )
        self.Tg = NumParam(info='Gate servo time constant',
                           default=0.05,
                           tex_name='T_g'
                           )
        self.Tr = NumParam(info='Dashpot time constant',
                           default=0.05,
                           tex_name='T_r'
                           )
        self.Tw = NumParam(info='Water inertia time constant',
                           default=1,
                           tex_name='T_w'
                           )
        self.At = NumParam(info='Turbine gain',
                           default=1,
                           tex_name='A_t'
                           )
        self.Dturb = NumParam(info='Turbine Damping Factor',
                              default=0.0,
                              tex_name='D_{turb}',
                              power=True,
                              )
        self.Hdam = NumParam(info='Head available at dam',
                             default=1,
                             tex_name='H_{dam}',
                             power=True,
                             )
        self.qNL = NumParam(info='No-Load flow at nominal head',
                            default=0.1,
                            tex_name='q_{NL}',
                            power=True,
                            )

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


class HYGOV4Model(TGBase):
    """
    Implement HYGOV4 model.
    """

    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        self.iTg = ConstService(v_str='u/Tg',
                                tex_name='1/T_g',
                                )
        self.R = ConstService(v_str='Rtemp + Rperm',
                              tex_name='R_{temp} + R_{perm}',
                              )
        self.TrRtemp = ConstService(v_str='Rtemp * Tr',
                                    tex_name='R_{temp} * T_r',
                                    )
        self.q0 = ConstService(v_str='tm0 / (At * Hdam) + qNL',
                               tex_name='q_0',
                               )
        self.pref = Algeb(info='Reference power input',
                          tex_name='P_{ref}',
                          v_str='Rperm * (q0 / (Hdam ** 0.5))',
                          e_str='Rperm * (q0 / (Hdam ** 0.5)) - pref'
                          )

        self.wd = Algeb(info='Generator speed deviation',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        v_str='0',
                        e_str='ue * (omega - wref) - wd',
                        )

        self.SV = GainLimiter(u='LAG_y', K=self.iTg, R=1,
                              upper=self.UO, lower=self.UC,
                              info='servo gain and limiters'
                              )

        self.GATE = IntegratorAntiWindup(u='SV_y', upper=self.PMAX, lower=self.PMIN,
                                         T=1, K=1,
                                         y0='(q0 / (Hdam ** 0.5))',
                                         info="Gate position"
                                         )
        self.WO = Washout(u='GATE_y',
                          K=self.TrRtemp,
                          T=self.Tr,
                          info='Washout feedback with T_r'
                          )

        self.Psum = Algeb(info='summation of power input to servo',
                          unit='p.u.',
                          tex_name='P_{sum}',
                          v_str='0',
                          e_str='ue * (pref + paux - (Rperm * GATE_y + WO_y ) - wd) - Psum',
                          )

        self.LAG = Lag(u=self.Psum,
                       K=1,
                       T=self.Tp,
                       info='lag block with T_p, outputs velocity',
                       )

        self.trhead = Algeb(info='turbine head',
                            unit='p.u.',
                            tex_name="trhead",
                            e_str='q_y**2 / GATE_y**2 - trhead',
                            v_str='Hdam',
                            )

        self.q = Integrator(u='Hdam - trhead',
                            T=self.Tw, K=1,
                            y0='q0',
                            check_init=False,
                            info="turbine flow (q)"
                            )

        self.pout.e_str = 'ue * (At * trhead * (q_y - qNL) - Dturb * wd * GATE_y) - pout'


class HYGOV4(HYGOV4Data, HYGOV4Model):
    """
    HYGOV4 turbine governor model.

    Implements the PSS/E HYGOV4 model with the following ignored:

    - input deadband DB1
    - valve position deadband DB2
    - nonlinear function bewteen GV and P_{GV}
    """

    def __init__(self, system, config):
        HYGOV4Data.__init__(self)
        HYGOV4Model.__init__(self, system, config)
