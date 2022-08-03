"""
EXAC2 exciter.
"""

from andes.core.block import GainLimiter, LVGate
from andes.core.param import NumParam
from andes.core.service import PostInitService
from andes.core.var import Algeb
from andes.models.exciter.exac1 import EXAC1Data, EXAC1Model


class EXAC2Data(EXAC1Data):
    """
    EXAC2 parameters.
    """

    def __init__(self):
        EXAC1Data.__init__(self)
        self.VAMAX = NumParam(info='Maximum KA block output',
                              tex_name='V_{RMAX}',
                              default=8,
                              unit='p.u.',
                              vrange=(0.5, 10),
                              )
        self.VAMIN = NumParam(info='Minimum KA block output',
                              tex_name='V_{RMIN}',
                              default=0,
                              unit='p.u.',
                              vrange=(-10, 0.5),
                              )
        self.VLR = NumParam(info='low voltage constant',
                            tex_name='V_{LR}',
                            default=0.0,
                            unit='p.u.',
                            )
        self.KL = NumParam(info='gain for low voltage',
                           tex_name='K_L',
                           default=1.0,
                           unit='p.u.',
                           )
        self.KH = NumParam(info='gain for high voltage',
                           tex_name='K_H',
                           default=1.0,
                           unit='p.u.',
                           )
        self.KB = NumParam(info='gain KB for regulator',
                           tex_name='K_B',
                           default=1.0,
                           unit='p.u.',
                           non_negative=True,
                           )


class EXAC2Model(EXAC1Model):
    """
    EXAC2 implementation.
    """

    def __init__(self, system, config):
        EXAC1Model.__init__(self, system, config)

        self.VHA = Algeb(v_str='LA_y - KH * VFE',
                         e_str='LA_y - KH * VFE - VHA',
                         )
        self.VL = Algeb(v_str='(VLRx - VFE) * KL',
                        e_str='(VLRx - VFE) * KL - VL')

        self.LA.lower = self.VAMIN
        self.LA.upper = self.VAMAX

        self.LVG = LVGate(u1=self.VHA, u2=self.VL)

        self.VR = GainLimiter(u='LVG_y', K=self.KB, R=1,
                              upper=self.VRMAX, lower=self.VRMIN,
                              )

        self.INT.u = 'ue* (VR_y - VFE)'  # this won't propagate to its `y` variable. Need the next line
        self.INT_y.e_str = 'ue* (VR_y - VFE)'

        self.vref.v_str = '(VFE * KL + VFE / KB) / KA + v'

        self.VLRx = Algeb(v_str='Indicator(VFE / KL / KB + VFE > VLR) * (VFE / KL / KB + VFE) + '
                          'Indicator(VFE / KL / KB + VFE < VLR) * VLR ',
                          e_str='VLR0 - VLRx'
                          )

        self.VLR0 = PostInitService(v_str='VLRx')


class EXAC2(EXAC1Data, EXAC1Model):
    """
    EXAC2 model.

    Ref: https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Exciter%20EXAC2.htm

    Notes
    -----

    ``VLR`` is an input parameter, but to initialize the LVGate, an internal
    ``VLRx`` will be computed as a contant upon initialization. The constant
    ``VLRx`` will be used in the place of ``VLR`` in the block diagram.
    """

    def __init__(self, system, config):
        EXAC2Data.__init__(self)
        EXAC2Model.__init__(self, system, config)
