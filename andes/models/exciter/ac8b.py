from andes.core.param import NumParam
from andes.core.var import Algeb

from andes.core.service import ConstService, VarService, FlagValue

from andes.core.block import LagAntiWindup, LeadLag, Washout, Lag, HVGate
from andes.core.block import LessThan, LVGate, AntiWindup, IntegratorAntiWindup
from andes.core.block import Integrator

from andes.models.exciter.excbase import ExcBase, ExcBaseData
from andes.models.exciter.saturation import ExcQuadSat


class AC8BData(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)






class AC8BModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)




class AC8B(AC8BData, AC8BModel):
    """
    Exciter AC8B model.

    Reference:

    [1] PowerWorld, Exciter AC8B, [Online],

    [2] NEPLAN, Exciters Models, [Online],

    Available:

    https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Exciter%20AC8B.htm

    https://www.neplan.ch/wp-content/uploads/2015/08/Nep_EXCITERS1.pdf
    """

    def __init__(self, system, config):
        AC8BData.__init__(self)
        AC8BModel.__init__(self, system, config)

