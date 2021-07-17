from andes.core.param import NumParam
from andes.core.block import Lag

from andes.models.renewable.regcvsg import REGCVSGData, VSGOuterPIData
from andes.models.renewable.regcvsg import REGCVSGModelBase, VSGOuterPIModel


class VSGInnerLagData:
    def __init__(self) -> None:
        self.Tiq = NumParam(default=0.01, tex_name='T_{Iq}')
        self.Tid = NumParam(default=0.01, tex_name='T_{Id}')


class VSGInnerLagModel:
    """
    REGCVSG2 model with lag transfer functions replacing PI controllers.
    """
    def __init__(self):
        self.LGId = Lag(u=self.PIdv_y, T=self.Tid, K=-1)  # Id
        self.LGIq = Lag(u=self.PIqv_y, T=self.Tiq, K=-1)  # Iq

        self.Id.e_str = 'LGId_y - Id'
        self.Iq.e_str = 'LGIq_y - Iq'


class REGCVSG2(REGCVSGData, VSGOuterPIData, VSGInnerLagData,
               REGCVSGModelBase, VSGOuterPIModel, VSGInnerLagModel):
    """
    Voltage-controlled VSC with VSG control.

    PI controllers are replaced with lag transfer functions.
    """

    def __init__(self, system, config):
        REGCVSGData.__init__(self)
        VSGOuterPIData.__init__(self)
        VSGInnerLagData.__init__(self)

        REGCVSGModelBase.__init__(self, system, config)
        VSGOuterPIModel.__init__(self)
        VSGInnerLagModel.__init__(self)
