from andes.core.param import RefParam
from andes.core.model import Model, ModelData


class AreaData(ModelData):
    def __init__(self):
        super().__init__()
        self.Bus = RefParam()
        self.AcTopology = RefParam()


class Area(AreaData, Model):
    def __init__(self, system, config):
        AreaData.__init__(self)
        Model.__init__(self, system, config)
