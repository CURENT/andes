from andes.core.param import NumParam
from andes.core.var import Algeb
from andes.core.service import ConstService
from andes.core.block import LeadLag, Washout, Lag
from andes.core.discrete import HardLimiter

from andes.models.exciter.excbase import ExcBase, ExcBaseData


class EXST1Data(ExcBaseData):
    """Parameters for EXST1."""

    def __init__(self):
        ExcBaseData.__init__(self)

        self.TR = NumParam(default=0.01,
                           info='Measurement delay',
                           tex_name='T_R',
                           )
        self.VIMAX = NumParam(default=0.2,
                              info='Max. input voltage',
                              tex_name='V_{IMAX}',
                              )

        self.VIMIN = NumParam(default=0,
                              info='Min. input voltage',
                              tex_name='V_{IMIN}',
                              )
        self.TC = NumParam(default=1,
                           info='LL numerator',
                           tex_name='T_C',
                           )
        self.TB = NumParam(default=1,
                           info='LL denominator',
                           tex_name='T_B',
                           )
        self.KA = NumParam(default=80,
                           info='Regulator gain',
                           tex_name='K_A',
                           )
        self.TA = NumParam(default=0.05,
                           info='Regulator delay',
                           tex_name='T_A',
                           )
        self.VRMAX = NumParam(default=8,
                              info='Max. regulator output',
                              tex_name='V_{RMAX}',
                              )

        self.VRMIN = NumParam(default=-3,
                              info='Min. regulator output',
                              tex_name='V_{RMIN}',
                              )
        self.KC = NumParam(default=0.2,
                           info='Coef. for Ifd',
                           tex_name='K_C',
                           )
        self.KF = NumParam(default=0.1,
                           info='Feedback gain',
                           tex_name='K_F',
                           )
        self.TF = NumParam(default=1.0,
                           info='Feedback delay',
                           tex_name='T_F',
                           non_negative=True,
                           non_zero=True,
                           )


class EXST1Model(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name='V_{ref0}',
                                  v_str='v + vf0 / KA',
                                  )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='vref0',
                          e_str='vref0 - vref'
                          )

        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        )
        self.vi.v_str = 'vf0 / KA'
        self.vi.e_str = '(vref - LG_y - WF_y) - vi'

        self.LG = Lag(u=self.v, T=self.TR, K=1,
                      info='Sensing delay',
                      )

        self.HLI = HardLimiter(u=self.vi, lower=self.VIMIN, upper=self.VIMAX,
                               info='Hard limiter on input',
                               )

        self.vl = Algeb(info='Input after limiter',
                        tex_name='V_l',
                        v_str='HLI_zi*vi + HLI_zu*VIMAX + HLI_zl*VIMIN',
                        e_str='HLI_zi*vi + HLI_zu*VIMAX + HLI_zl*VIMIN - vl',
                        )

        self.LL = LeadLag(u=self.vl, T1=self.TC, T2=self.TB, info='Lead-lag compensator', zero_out=True)

        self.LR = Lag(u=self.LL_y, T=self.TA, K=self.KA, info='Regulator')

        self.WF = Washout(u=self.LR_y, T=self.TF, K=self.KF, info='Stablizing circuit feedback')

        # the following uses `XadIfd` for `IIFD` in the PSS/E manual
        self.vfmax = Algeb(info='Upper bound of output limiter',
                           tex_name='V_{fmax}',
                           v_str='VRMAX - KC * XadIfd',
                           e_str='VRMAX - KC * XadIfd - vfmax',
                           )
        self.vfmin = Algeb(info='Lower bound of output limiter',
                           tex_name='V_{fmin}',
                           v_str='VRMIN - KC * XadIfd',
                           e_str='VRMIN - KC * XadIfd - vfmin',
                           )

        self.HLR = HardLimiter(u=self.WF_y, lower=self.vfmin, upper=self.vfmax,
                               info='Hard limiter on regulator output')

        self.vout.e_str = 'LR_y*HLR_zi + vfmin*HLR_zl + vfmax*HLR_zu - vout'


class EXST1(EXST1Data, EXST1Model):
    """
    EXST1-type static excitation system.
    """

    def __init__(self, system, config):
        EXST1Data.__init__(self)
        EXST1Model.__init__(self, system, config)
