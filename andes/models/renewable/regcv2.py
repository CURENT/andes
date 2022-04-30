from andes.core.block import Lag
from andes.core.param import NumParam
from andes.models.renewable.regcv1 import (REGCV1Data, REGCV1ModelBase,
                                           VSGOuterPIData, VSGOuterPIModel,)


class VSGInnerLagData:
    def __init__(self) -> None:
        self.Tiq = NumParam(default=0.01, tex_name='T_{Iq}')
        self.Tid = NumParam(default=0.01, tex_name='T_{Id}')


class VSGInnerLagModel:
    """
    REGCV2 model with lag transfer functions replacing PI controllers.
    """

    def __init__(self):
        self.LGId = Lag(u=self.PIvd_y, T=self.Tid, K=1)  # Id
        self.LGIq = Lag(u=self.PIvq_y, T=self.Tiq, K=1)  # Iq

        self.Id.e_str = 'LGId_y - Id'
        self.Iq.e_str = 'LGIq_y - Iq'


class REGCV2(REGCV1Data, VSGOuterPIData, VSGInnerLagData,
             REGCV1ModelBase, VSGOuterPIModel, VSGInnerLagModel):
    """
    Voltage-controlled VSC with VSG control.

    The inner-loop current PI controllers are replaced with lag transfer
    functions.

    Notes
    -----
    To avoid small-signal stability issues, one take extreme care in setting the
    PI control gains ``Kpvd``, ``Kivd``, ``Kpvq``, and ``Kivq``, and the emulated
    inertia ``M`` and damping ``D``.

    """

    def __init__(self, system, config):
        REGCV1Data.__init__(self)
        VSGOuterPIData.__init__(self)
        VSGInnerLagData.__init__(self)

        REGCV1ModelBase.__init__(self, system, config)
        VSGOuterPIModel.__init__(self, vderr='vd-vref2')
        VSGInnerLagModel.__init__(self)
