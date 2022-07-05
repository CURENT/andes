"""
HYGOV4 hydro governor model
"""

from cmath import inf
from andes.core import Algeb, ConstService, NumParam, State
from andes.core.block import  Integrator, Lag, AntiWindupRate
from andes.core.service import VarService
from andes.models.governor.tgbase import TGBase, TGBaseData 

class HYGOV4Data(TGBaseData):
    """
    HYGOV4 Data
    """ 

    def __init__(self):
        super().__init__()
        self.Rperm = NumParam(info = 'Speed Regulation Gain (mach. base default)',
                              tex_name = 'Rperm',
                              default = 0.5,
                              unit = 'p.u.',
                              ipower = True,
                              )
        self.Rtemp = NumParam(info = 'Temporary Droop (Rtemp < Rperm)',
                              tex_name = 'Rtemp',
                              default = 1,
                              unit = 'p.u.',
                              ipower = True,
                              )
        self.UO = NumParam(info = 'Maximum Gate opening velocity',
                           tex_name = 'U_{O}',
                           default = 1,
                           unit = 'p.u.',
                           power = True,
                           ) 
        self.UC = NumParam(info = 'Maximum Gate closing velocity',
                           tex_name = 'U_{C}',
                           default = 1,
                           unit = 'p.u.',
                           power = True,
                           )
        self.PMAX = NumParam(info = 'Maximum Gate opening',
                             tex_name = 'PMAX',
                             default = 1,
                             unit = 'p.u.',
                             power = True,
                             ) 
        self.PMIN = NumParam(info = 'Minimum Gate opening',
                             tex_name = 'PMIN',
                             default = 0,
                             unit = 'p.u.',
                             power = True,
                             )
        self.Tp = NumParam(info = 'Pilot servo time constant',
                           default = 0.05,
                           tex_name = 'T_p')
        self.Tg = NumParam(info = 'Gate servo time constant',
                           default = 0.05,
                           tex_name = 'T_g')
        self.Tr = NumParam(info = 'Dashpot time constant',
                           default = 0.05,
                           tex_name = 'T_r')
        self.Tw = NumParam(info = 'Water inertia time constant',
                           default = 1,
                           tex_name = 'T_w')
        self.At = NumParam(info = 'Turbine gain',
                           default = 1,
                           tex_name = 'A_t')
        self.Dturb = NumParam(info = 'Turbine Damping Factor',
                              default = 0.0,
                              tex_name ='D_{turb}',
                              power = True,
                              )
        self.qNL = NumParam(info = 'No-Load flow at nominal head',
                            default = 0.1,
                            tex_name = 'q_NL',
                            power = True,
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

    The input lead-lag filter is ignored.

    The ``g`` signal (LAG) is initialized to the initial value of
    ``q`` (Integrator) to simplify the initializaiton equations.
    """

    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        self.tr = ConstService(v_str='r * Tr',
                               tex_name='r*Tr',
                               )
        self.R = ConstService(v_str='Rtemp + Rperm',
                               tex_name='Rtemp + Rperm',
                               )
        self.gr = ConstService(v_str = '1/r',
                               tex_name = '1/r'
                               )
        self.q0 = ConstService(v_str='tm0 / At + qNL',
                               tex_name='q_0',
                               )
        self.pref = Algeb(info = 'Reference power input',
                          tex_name= 'P_{ref}',
                          v_str = 'R * q0',
                          e_str = 'R * q0 - pref'
                          )
        self.wd = Algeb(info = 'Generator speed deviation',
                        unit = 'p.u.',
                        tex_name = r'\omega_{dev}',
                        v_str = 0,
                        e_str = 'ue - (omega - wref) - wd',
                        )
        self.rg = Algeb(info = 'input to LAGTR',
                        unit = 'p.u.',
                        tex_name = 'rg',
                        v_str = 0,
                        e_str = '(Rtemp * g) - rg',
                        )
        self.LAGTR = Lag(u = self.rg,
                     K = 1,
                     T = self.Tr,
                     info = 'lag block with T_r',
                     )
        self.up = Algeb(info = 'input to LAGTP',
                        unit = 'p.u.',
                        tex_name = 'up',
                        v_str = 0,
                        e_str = '(pref + paux - R + LAGTR_y) - up',
                        )
        self.LAGTP = Lag(u = self.up,
                     K = 1,
                     T = self.Tp,
                     info = 'lag block with T_p, velocity',
                     )

        self.k = ConstService(v_str='u/Tg',
                                 tex_name='1/T_g',
                                 )

        self.gtpos = State(info='State in gate position (c)',
                           unit='rad',
                           v_str='q0',
                           tex_name=r'\delta',
                           ##t_const = self.Tg,
                           e_str='LGTP_y * k'
                           )
        
        self.g_lim = AntiWindupRate(u=self.gtpos, lower=self.PMIN, upper=self.PMAX,
                                     rate_lower=self.UO, rate_upper=self.UC,
                                     tex_name='lim_{g}',
                                     info='gate velocity limiter',
                                     )
        self.g = Algeb(info='gate',
                        unit='p.u.',
                        tex_name="g",
                        v_str='q0',
                        e_str='gtpos - g',
                        )

        self.h = Algeb(info='turbine head',
                       unit='p.u.',
                       tex_name="h",
                       e_str='q_y**2 / g**2 - h',
                       v_str='1',
                       )
        self.q = Integrator(u="1 - q_y**2 / g**2",
                            T=self.Tw, K=1,
                            y0='q0',
                            check_init=False,
                            info="turbine flow (q)"
                            )

        self.pout.e_str = 'ue * (At * h * (q_y - qNL) - Dt * wd * g) - pout'

class HYGOV4(HYGOV4Data, HYGOV4Model):
    """
    HYGOV4 turbine governor model.

    Implements the PSS/E HYGOV4 model without deadband.

    """

    def __init__(self, system, config):
        HYGOV4Data.__init__(self)
        HYGOV4Model.__init__(self, system, config)

