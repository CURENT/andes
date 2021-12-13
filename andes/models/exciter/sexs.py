from andes.core.block import LagAntiWindup, LeadLag
from andes.core.param import NumParam
from andes.core.service import ConstService, PostInitService
from andes.core.var import Algeb
from andes.models.exciter.excbase import ExcBase, ExcBaseData


class SEXSData(ExcBaseData):
    """Data class for Simplified Excitation System model (SEXS)"""

    def __init__(self):
        ExcBaseData.__init__(self)
        self.TATB = NumParam(default=0.4,
                             tex_name='T_A/T_B',
                             info='Time constant TA/TB',
                             vrange=(0.05, 1),
                             )
        self.TB = NumParam(default=5,
                           tex_name='T_B',
                           info='Time constant TB in LL',
                           vrange=(5, 20),
                           )
        self.K = NumParam(default=20,
                          tex_name='K',
                          info='Gain',
                          non_zero=True,
                          vrange=(20, 100),
                          )
        # 5 <= K * TA / TB <= 15
        self.TE = NumParam(default='1',
                           tex_name='T_E',
                           info='AW Lag time constant',
                           vrange=(0, 0.5),
                           )
        self.EMIN = NumParam(default=-99,
                             tex_name='E_{MIN}',
                             info='lower limit',
                             )
        self.EMAX = NumParam(default=99,
                             tex_name='E_{MAX}',
                             info='upper limit',
                             vrange=(3, 6),
                             )


class SEXSModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        self.TA = ConstService(v_str='TATB * TB')

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='v + vf0 / K',
                          e_str='vref0 - vref'
                          )

        self.vref0 = PostInitService(info='Constant vref',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        )
        self.vi.e_str = '(vref - v) - vi'
        self.vi.v_str = 'v + vf0 / K- v'

        self.LL = LeadLag(u=self.vi, T1=self.TA, T2=self.TB, zero_out=True)

        self.LAW = LagAntiWindup(u=self.LL_y,
                                 T=self.TE,
                                 K=self.K,
                                 lower=self.EMIN,
                                 upper=self.EMAX,
                                 )

        self.vout.e_str = 'ue * LAW_y - vout'


class SEXS(SEXSData, SEXSModel):
    """Simplified Excitation System"""

    def __init__(self, system, config):
        SEXSData.__init__(self)
        SEXSModel.__init__(self, system, config)
