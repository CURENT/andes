"""
Round-rotor generator model.
"""

import logging

from andes.core.service import VarService
from andes.models.synchronous.genbase import GENBase, Flux0
from andes.models.synchronous.genrou import GENROUData, GENROUModel

logger = logging.getLogger(__name__)


class GENROUOSModel(GENROUModel):
    def __init__(self):
        GENROUModel.__init__(self)
        delattr(self, 'psi2q')
        delattr(self, 'psi2d')
        delattr(self, 'psi2')

        self.algebs.pop('psi2q')
        self.algebs.pop('psi2d')
        self.algebs.pop('psi2')

        self.psi2q = VarService(tex_name=r"\psi_{aq}", info='q-axis air gap flux',
                                v_str='gq1*e1d + (1-gq1)*e2q',
                                )

        self.psi2d = VarService(tex_name=r"\psi_{ad}", info='d-axis air gap flux',
                                v_str='gd1*e1q + gd2*(xd1-xl)*e2d',
                                )

        self.psi2 = VarService(tex_name=r"\psi_a", info='air gap flux magnitude',
                               v_str='sqrt(psi2d **2 + psi2q ** 2)',
                               )
        # fix the reference to `psi2`
        self.SL.u = self.psi2


class GENROUOS(GENROUData, GENBase, GENROUOSModel, Flux0):
    """
    Round rotor generator with quadratic saturation. The model implements
    operator splitting for the flux linkage equations.
    """

    def __init__(self, system, config):
        GENROUData.__init__(self)
        GENBase.__init__(self, system, config)
        Flux0.__init__(self)
        GENROUOSModel.__init__(self)
