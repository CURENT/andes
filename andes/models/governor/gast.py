"""
Governor GAST.
"""

from andes.core import Algeb, ConstService, LagAntiWindup, NumParam
from andes.core.block import Lag, LVGate
from andes.models.governor.tgbase import TGBase, TGBaseData


class GASTData(TGBaseData):
    """
    Data for governor GAST.
    """

    def __init__(self):
        super().__init__()
        self.R = NumParam(info='Speed regulation gain (mach. base default)',
                          tex_name='R',
                          default=0.05,
                          unit='p.u.',
                          ipower=True,
                          )
        self.VMAX = NumParam(info='Maximum valve position',
                             tex_name='V_{max}',
                             unit='p.u.',
                             default=1.2,
                             power=True,
                             )
        self.VMIN = NumParam(info='Minimum valve position',
                             tex_name='V_{min}',
                             unit='p.u.',
                             default=0.0,
                             power=True,
                             )

        self.KT = NumParam(info='Temperature limiter gain',
                           default=5,
                           tex_name='K_T')

        self.AT = NumParam(info='Ambient temperature load limit',
                           default=1,
                           tex_name='A_T',
                           power=True,
                           )

        self.T1 = NumParam(info='Valve time constant',
                           default=0.1,
                           tex_name='T_1')
        self.T2 = NumParam(info='Lead-lag lead time constant',
                           default=0.2,
                           tex_name='T_2')
        self.T3 = NumParam(info='Lead-lag lag time constant',
                           default=10.0,
                           tex_name='T_3')
        self.Dt = NumParam(info='Turbine damping coefficient',
                           default=0.0,
                           tex_name='D_t',
                           power=True,
                           )


class GASTModel(TGBase):
    """
    Implement GAST model.
    """

    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        self.gain = ConstService(v_str='ue/R',
                                 tex_name='G',
                                 )

        self.pref = Algeb(info='Reference power input',
                          tex_name='P_{ref}',
                          v_str='tm0 * R',
                          e_str='pref0 * R - pref',
                          )

        self.wd = Algeb(info='Generator under speed',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        v_str='0',
                        e_str='ue * (omega - wref) - wd',
                        )
        self.pd = Algeb(info='Pref plus under speed times gain',
                        unit='p.u.',
                        tex_name="P_d",
                        v_str='ue * tm0',
                        e_str='ue*(- wd + pref + paux) * gain - pd')

        self.v9 = Algeb(tex_name=r'V_{9}',
                        info='V_9 for LVGate input',
                        v_str='ue * (AT + KT * (AT - tm0))',
                        e_str='ue * (AT + KT * (AT - LG3_y)) - v9',
                        )

        self.LVG = LVGate(u1=self.pd,
                          u2=self.v9,
                          info='LVGate',
                          )

        self.LAG = LagAntiWindup(u=self.LVG_y,
                                 K=1,
                                 T=self.T1,
                                 lower=self.VMIN,
                                 upper=self.VMAX,
                                 )

        self.LG2 = Lag(u=self.LAG_y, T=self.T2, K=1,
                       info='Lag T2')

        self.LG3 = Lag(u=self.LG2_y, T=self.T3, K=1,
                       info='Lag T3')

        self.pout.e_str = 'ue * (LG2_y - Dt * wd) - pout'


class GAST(GASTData, GASTModel):
    """
    GAST turbine governor model.

    Reference:

    [1] Neplan, TURBINE-GOVERNOR GAST, [Online],

    Available:

    https://www.neplan.ch/wp-content/uploads/2015/08/Nep_TURBINES_GOV.pdf
    """

    def __init__(self, system, config):
        GASTData.__init__(self)
        GASTModel.__init__(self, system, config)
