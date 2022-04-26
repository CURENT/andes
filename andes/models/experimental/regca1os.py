"""
Module for renewable energy generator (converter) model A with operator
splitting.
"""

from andes.core.service import VarService

from andes.models.renewable.regca1 import REGCA1Data, REGCA1Model


class REGCA1OSModel(REGCA1Model):
    """
    REGCA1OS implementation.
    """

    def __init__(self, system, config):
        REGCA1Model.__init__(self, system, config)
        delattr(self, 'LVG')
        delattr(self, 'LVG_y')
        delattr(self, 'Ipout')

        self.blocks.pop('LVG')
        self.algebs.pop('LVG_y')
        self.algebs.pop('Ipout')

        self.LVG_y = VarService(
            v_str='Piecewise((0, v <= Lvpnt0), \
                             ((v - Lvpnt0) * kLVG, v <= Lvpnt1), \
                             (1, v > Lvpnt1), \
                             (0, True))')
        self.Ipout = VarService(
            v_str='S0_y * LVG_y'
        )


class REGCA1OS(REGCA1Data, REGCA1OSModel):
    """
    Renewable energy generator model type A with operator splitting.
    """

    def __init__(self, system, config):
        REGCA1Data.__init__(self)
        REGCA1OSModel.__init__(self, system, config)
