"""
Grid-forming renewable energy model with VSM control.
"""

from andes.core.var import Algeb, ExtAlgeb
from andes.core.service import DeviceFinder, ConstService
from andes.core.param import IdxParam, NumParam
from andes.core.block import Integrator, GainLimiter
from andes.models.renewable.regf1 import REGF1Data, REGF1Model, REGFInnerPIModel, REGFOuterPIModel


class REGF2Data(REGF1Data):
    """
    REGF2 model data.
    """

    def __init__(self):
        REGF1Data.__init__(self)

        self.mf = NumParam(default=0.15, info='VSM inertia constant',
                           tex_name='M_f'
                           )
        self.dd = NumParam(default=0.11, info='VSM damping factor',
                           tex_name='d_d')
        self.pll = IdxParam(info='PLL device idx (optional)', default=None, model='PLL')


class REGF2Primary:
    """
    Primary frequency and voltage controllers based on VSM.
    """

    def __init__(self) -> None:

        self.pllidx = DeviceFinder(self.pll,
                                   link=self.bus,
                                   idx_name='bus',
                                   default_model='PLL2',
                                   auto_find=True,
                                   auto_add=True,
                                   )

        self.plldw = ExtAlgeb(model='PLL',
                              src='PI_y',
                              indexer=self.pllidx,
                              allow_none=True,
                              tex_name=r'\Delta \omega_{PLL}',
                              info='PLL measured freq. deviation',
                              )
        self.wref = Algeb(v_str='1', e_str='1 - wref',
                          tex_name=r'\omega_{ref}', info='speed ref', unit='pu')

        self.Tint = ConstService(v_str='mf * wdrp', tex_name='T_{int}')

        self.INTw = Integrator(u='(plldw * dd * wdrp) + wref + wdrp * (PIplim_y - Psen_y) - INTw_y',
                               T=self.Tint, K=1, y0='wref',
                               )

        self.dw = GainLimiter(u='w0 * (INTw_y - wref)', K=1, R=1,
                              lower=self.dwmin,
                              upper=self.dwmax,
                              )

        self.vref2 = Algeb(tex_name=r'v_{ref2}',
                           info='voltage reference after droop',
                           e_str='(u * PIqlim_y - Qsen_y) * Qdrp + vref - vref2',
                           v_str='u * vref')


class REGF2(REGF2Data, REGF1Model, REGF2Primary,
            REGFOuterPIModel, REGFInnerPIModel):
    """
    Grid-forming inverter with VSM control.

    Implementation of EPRI Memorandum

    D. Ramasubramanian, "PROPOSAL FOR SUITE OF GENERIC GRID FORMING (GFM) POSITIVE SEQUENCE MODELS"
    """

    def __init__(self, system, config):
        REGF2Data.__init__(self)

        REGF1Model.__init__(self, system, config)
        REGF2Primary.__init__(self)
        REGFOuterPIModel.__init__(self)
        REGFInnerPIModel.__init__(self)
