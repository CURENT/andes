from andes.core.param import RefParam
from andes.core.model import Model, ModelData
from andes.core.var import ExtAlgeb, Algeb  # NOQA


class AreaData(ModelData):
    def __init__(self):
        super().__init__()
        self.Bus = RefParam()
        self.AcTopology = RefParam()


class Area(AreaData, Model):
    def __init__(self, system, config):
        AreaData.__init__(self)
        Model.__init__(self, system, config)
        self.flags.update({'pflow': True})

        self.a = ExtAlgeb(model='AcTopology', src='a', indexer=self.AcTopology)
        self.v = ExtAlgeb(model='AcTopology', src='v', indexer=self.AcTopology)
        # self.a_times_v = Algeb(v_setter=True)
        # self.a_times_v.e_str = 'a * v'
