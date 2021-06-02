from andes.core.block import Lag

from andes.models.renewable.regcvsg import REGCVSGData, REGCVSGModelBase


class REGCVSG2Model(REGCVSGModelBase):
    """
    REGCVSG2 model with lag transfer functions replacing PI controllers.
    """
    def __init__(self, system, config):
        REGCVSGModelBase.__init__(system, config)


class REGCVSG2(REGCVSGData, REGCVSG2Model):
    """
    Voltage-controlled VSC with VSG control.

    PI controllers are replaced with lag transfer functions.
    """

    def __init__(self, system, config):
        REGCVSGData.__init__(self)
        REGCVSG2Model.__init__(self, system, config)
