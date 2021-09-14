from andes.core.param import NumParam
from andes.core.var import Algeb

from andes.core.service import ConstService, VarService, FlagValue

from andes.core.block import LagAntiWindup, LeadLag, Washout, Lag, HVGate
from andes.core.block import LessThan
from andes.core.block import Integrator

from andes.models.exciter.excbase import ExcBase, ExcBaseData
from andes.models.exciter.saturation import ExcQuadSat


class ESAC1AData(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)


class ESAC1AModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)


class ESAC1A(ESAC1AData, ESAC1AModel):
    """
    Static exciter type 3A model
    """

    def __init__(self, system, config):
        ESAC1AData.__init__(self)
        ESAC1AModel.__init__(self, system, config)
