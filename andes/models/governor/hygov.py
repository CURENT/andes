from andes.core import Algeb, ConstService, NumParam
from andes.core.block import Integrator, Lag
from andes.models.governor.tgbase import TGBase, TGBaseData


class HYGOVData(TGBaseData):
    def __init__(self):
        super().__init__()
        # Is ipower still approriate?
        self.R = NumParam(info='Speed regulation gain (mach. base default)',
                          tex_name='R',
                          default=0.05,
                          unit='p.u.',
                          ipower=True,
                          )
        # Is power=True correct here?
        self.r = NumParam(info='Temporary droop (R<r)',
                          tex_name='r',
                          default=1,
                          unit='p.u.',
                          power=True,
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
        # TODO: default value
        self.Tw = NumParam(info='Water inertia time constant constant',
                           default=1,
                           tex_name='T_w')
        # TODO: default value
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


class HYGOVModel(TGBase):
    """
    Implement HYGOV model.

    The input lead-lag filter is ignored.
    """

    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        self.gain = ConstService(v_str='ue/R',
                                 tex_name='G',
                                 )
        self.gainr = ConstService(v_str='1/r',
                                  tex_name='g',
                                  )
        # TODO: equations here can be inapproriate
        self.pref = Algeb(info='Reference power input',
                          tex_name='P_{ref}',
                          v_str='tm0 * R',
                          e_str='pref0 * R - pref',
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
                        v_str='ue * tm0',
                        e_str='ue*(-wd + pref + paux - R * dg) - pd')

        self.LG = Lag(u=self.pd,
                      K=1,
                      T=self.Tf,
                      info='filter after speed deviation (e)',
                      )

        # TODO: check y0
        self.INT = Integrator(u=self.LG_y,
                              T="r*Tr", K=1,
                              y0=0,
                              check_init=False,
                              )

        # TODO: check v_str
        self.dg = Algeb(info='desired gate (c)',
                        unit='p.u.',
                        tex_name="dg",
                        v_str='ue * tm0',
                        e_str='INT_y + gainr * LG_y - dg'
                        )

        self.LAG = Lag(u=self.dg,
                       K=1,
                       T=self.Tg,
                       info='gate opening (g)',
                       )

        # TODO: check v_str
        self.h = Algeb(info='turbine head',
                       unit='p.u.',
                       tex_name="h",
                       v_str='ue * tm0',
                       e_str='q_y / LAG_y * q_y / LAG_y - h'
                       )

        # TODO: check y0
        self.q = Integrator(u="1 - h",
                            T=self.Tw, K=1,
                            y0=0,
                            check_init=False,
                            info="turbine flow (q)"
                            )

        self.pout.e_str = 'ue * (At * (q_y - qNL) - Dt * wd * LAG_y) - pout'


class HYGOV(HYGOVData, HYGOVModel):
    """
    HYGOV turbine governor model.

    Implements the PSS/E HYGOV model without deadband.
    """

    def __init__(self, system, config):
        HYGOVData.__init__(self)
        HYGOVModel.__init__(self, system, config)
