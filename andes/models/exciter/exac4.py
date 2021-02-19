from andes.models.exciter.excbase import ExcBase, ExcBaseData

from andes.core.param import NumParam
from andes.core.var import Algeb
from andes.core.block import LeadLag, Lag
from andes.core.service import ConstService
from andes.core.discrete import HardLimiter


class EXAC4Data(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u.',
                           )
        self.VIMAX = NumParam(default=5.0,
                              info='Max. input voltage',
                              tex_name='V_{IMAX}',
                              vrange=(0, 5),
                              )
        self.VIMIN = NumParam(default=-0.1,
                              info='Min. input voltage',
                              tex_name='V_{IMIN}',
                              vrange=(-1, 0),
                              )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=1,
                           unit='p.u.',
                           )
        self.TB = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_B',
                           default=1,
                           unit='p.u.',
                           )
        self.KA = NumParam(default=80,
                           info='Regulator gain',
                           tex_name='K_A',
                           )
        self.TA = NumParam(info='Lag time constant in regulator',
                           tex_name='T_A',
                           default=0.04,
                           unit='p.u.',
                           )
        self.VRMAX = NumParam(info='Maximum excitation limit',
                              tex_name='V_{RMAX}',
                              default=8,
                              unit='p.u.',
                              vrange=(0.5, 10),
                              )
        self.VRMIN = NumParam(info='Minimum excitation limit',
                              tex_name='V_{RMIN}',
                              default=0,
                              unit='p.u.',
                              vrange=(-10, 0.5),
                              )
        self.KC = NumParam(default=0.0,
                           tex_name='K_C',
                           info='Reactive power compensation gain',
                           )


class EXAC4Model(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name='V_{ref0}',
                                  v_str='v + vf0 / KA',
                                  )

        self.LG = Lag(u=self.v, T=self.TR, K=1,
                      info='Sensing delay',
                      )
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        )
        self.vi.v_str = 'vf0 / KA'
        self.vi.e_str = '(vref0 - LG_y) - vi'

        self.HLI = HardLimiter(u=self.vi, lower=self.VIMIN, upper=self.VIMAX,
                               info='Hard limiter on input',
                               )

        self.LL = LeadLag(u='vi * HLI_zi + VIMIN * HLI_zl + VIMAX * HLI_zu',
                          T1=self.TC,
                          T2=self.TB,
                          info='Lead-lag compensator',
                          zero_out=True,
                          )

        self.LR = Lag(u=self.LL_y, T=self.TA, K=self.KA, info='Regulator')

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

        self.HLR = HardLimiter(u=self.LR_y, lower=self.vfmin, upper=self.vfmax,
                               info='Hard limiter on regulator output')

        self.vout.e_str = 'LR_y*HLR_zi + vfmin*HLR_zl + vfmax*HLR_zu - vout'


class EXAC4(EXAC4Data, EXAC4Model):
    """
    IEEE Type AC4 excitation system model.
    """

    def __init__(self, system, config):
        EXAC4Data.__init__(self)
        EXAC4Model.__init__(self, system, config)
