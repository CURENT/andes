"""
ESDC1A model.
"""

from andes.core.service import ConstService

from andes.models.exciter.esdc2a import ESDC2AData, ESDC2AModel


class ESDC1AModel(ESDC2AModel):
    """
    Implementation of ESDC1A.
    """

    def __init__(self, system, config):
        ESDC2AModel.__init__(self, system, config)

        self.services.pop("VRU")
        self.services.pop("VRL")
        self.services_var.pop("VRU")
        self.services_var.pop("VRL")
        self.services_var_seq.pop("VRU")
        self.services_var_seq.pop("VRL")

        delattr(self, 'VRU')
        delattr(self, 'VRL')

        self.VRU = ConstService(v_str='VRMAXc',
                                tex_name='V_{RMAX}',
                                )
        self.VRL = ConstService(v_str='VRMIN',
                                tex_name='V_{RMIN}',
                                )


class ESDC1A(ESDC2AData, ESDC1AModel):
    """
    ESDC1A model.

    This model derives from ESDC2A and changes the regular limits to "VRMAX" and
    "VRMIN".

    """

    def __init__(self, system, config):
        ESDC2AData.__init__(self)
        ESDC1AModel.__init__(self, system, config)
