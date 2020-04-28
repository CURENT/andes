from andes.core.param import ExtParam
from andes.core.model import Model, ModelData
from andes.core.var import ExtAlgeb, Algeb  # NOQA
from andes.core.service import NumReduce, NumRepeat, BackRef
from andes.shared import np


class AreaData(ModelData):
    def __init__(self):
        super().__init__()


class Area(AreaData, Model):
    def __init__(self, system, config):
        AreaData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Collection'
        self.flags.update({'pflow': True,
                           'tds': True})

        self.Bus = BackRef()
        self.ACTopology = BackRef()

        # --------------------Experiment Zone--------------------
        self.Vn = ExtParam(model='Bus', src='Vn', indexer=self.ACTopology, export=False)
        self.Vn_sum = NumReduce(u=self.Vn, fun=np.sum, ref=self.Bus)
        self.Vn_sum_rep = NumRepeat(u=self.Vn_sum, ref=self.Bus)

        self.a = ExtAlgeb(model='ACTopology', src='a', indexer=self.ACTopology,
                          info='Bus voltage angle')
        self.v = ExtAlgeb(model='ACTopology', src='v', indexer=self.ACTopology,
                          info='Bus voltage magnitude')

        # self.time = Algeb(e_str='time - dae_t', v_setter=True)
