"""
Grid-forming renewable energy model with dispatchable virtual oscillator control
(dVOC).
"""

from andes.core.var import Algeb, State
from andes.core.service import ConstService
from andes.core.block import GainLimiter
from andes.models.renewable.regf1 import REGF1Data, REGF1Model, REGFInnerPIModel, REGFOuterPIModel


class REGF3Primary:
    """
    Primary frequency and voltage controllers based on dVOC.
    """

    def __init__(self) -> None:

        self.Kdvoc = ConstService(
            v_str='wdrp * (4 * 100000000) / (100000000 - (2*(100-100*Qdrp)**2 ) - 10000)**2',
            info='Kdvoc parameter',
        )

        self.wref = Algeb(v_str='1', e_str='1 - wref',
                          tex_name=r'\omega_{ref}', info='speed ref', unit='pu')

        self.Vref = Algeb(v_str='vref', e_str='vref - Vref')

        self.dw = GainLimiter(u='w0 * (PIplim_y - Psen_y) * wdrp / vref2 / vref2', K=1, R=1,
                              lower=self.dwmin,
                              upper=self.dwmax,
                              )

        derv = 'wdrp / vref2 * (PIqlim_y - Qsen_y) + (vref2 * Kdvoc * (Vref + vref2) * (Vref - vref2)) - derv'
        self.derv = Algeb(tex_name=r'dV',
                          info='input to voltage integrator',
                          e_str=derv,
                          v_str='0')

        self.vref2 = State(e_str='w0 * derv', v_str='vd')


class REGF3(REGF1Data, REGF1Model, REGF3Primary,
            REGFOuterPIModel, REGFInnerPIModel):
    """
    Grid-forming inverter with dVOC.

    Implementation of EPRI Memorandum

    D. Ramasubramanian, "PROPOSAL FOR SUITE OF GENERIC GRID FORMING (GFM) POSITIVE SEQUENCE MODELS"
    """

    def __init__(self, system, config):
        REGF1Data.__init__(self)

        REGF1Model.__init__(self, system, config)
        REGF3Primary.__init__(self)
        REGFOuterPIModel.__init__(self)
        REGFInnerPIModel.__init__(self)
