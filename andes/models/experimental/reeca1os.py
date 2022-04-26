"""
REECA1 with operator splitting.
"""

from andes.core.service import VarService
from andes.models.renewable.reeca1 import REECA1Data, REECA1Model


class REECA1OSModel(REECA1Model):
    """
    REECA1 with operator splitting.
    """

    def __init__(self, system, config):
        super().__init__(system, config)

        # --- split `Verr`` ---
        delattr(self, 'Verr')
        self.algebs.pop('Verr')
        self.Verr = VarService(v_str='Vref0 - s0_y')
        self.dbV.u = self.Verr
        self.dbV.db.u = self.Verr

        # --- split `Iqinj` ---
        delattr(self, 'Iqinj')
        self.algebs.pop('Iqinj')
        # Gain after dbB
        Iqv = "(dbV_y * Kqv)"
        Iqinj = f'{Iqv} * Volt_dip + ' \
                f'(1 - Volt_dip) * fThld * ({Iqv} * nThld + Iqfrz * pThld)'

        self.Iqinj = VarService(v_str=Iqinj)


class REECA1OS(REECA1Data, REECA1OSModel):
    """
    REECA1 with operator splitting.
    """

    def __init__(self, system, config):
        REECA1Data.__init__(self)
        REECA1OSModel.__init__(self, system, config)
