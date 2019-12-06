import numpy as np
from andes.core.param import RefParam, ExtParam
from andes.core.model import Model, ModelData
from andes.core.var import ExtAlgeb, Algeb  # NOQA
from andes.core.service import ServiceReduce, ServiceRepeat


class AreaData(ModelData):
    def __init__(self):
        super().__init__()


class Area(AreaData, Model):
    def __init__(self, system, config):
        AreaData.__init__(self)
        Model.__init__(self, system, config)
        self.flags.update({'pflow': True,
                           'tds': True})

        self.Bus = RefParam(export=False)
        self.AcTopology = RefParam(export=False)

        # --------------------Experiment Zone--------------------
        self.Vn = ExtParam(model='Bus', src='Vn', indexer=self.AcTopology, export=False)
        self.Vn_sum = ServiceReduce(u=self.Vn, fun=np.sum, ref=self.Bus)
        self.Vn_sum_rep = ServiceRepeat(u=self.Vn_sum, ref=self.Bus)

        self.a = ExtAlgeb(model='AcTopology', src='a', indexer=self.AcTopology)
        self.v = ExtAlgeb(model='AcTopology', src='v', indexer=self.AcTopology)

        # self.time = Algeb(e_str='time - dae_t', v_setter=True)
