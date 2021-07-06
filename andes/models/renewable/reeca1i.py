"""
REECA1 model with inertia emulation.
"""

from andes.core.param import NumParam, IdxParam
from andes.core.var import ExtAlgeb
from andes.core.service import DeviceFinder

from andes.models.renewable.reeca1 import REECA1Data, REECA1Model


class REECA1IData(REECA1Data):
    def __init__(self):
        REECA1Data.__init__(self)
        self.Kwd = NumParam(default=0.0,
                            info='gain for speed derivative',
                            tex_name='K_{wd}',
                            )
        self.busroc = IdxParam(info='Optional BusROCOF device idx',
                               model='BusROCOF',
                               default=None,
                               )


class REECA1IModel(REECA1Model):
    def __init__(self, system, config):
        REECA1Model.__init__(self, system, config)
        self.busrocof = DeviceFinder(self.busroc,
                                     link=self.bus,
                                     idx_name='bus',
                                     )

        self.df = ExtAlgeb(model='FreqMeasurement',
                           src='Wf_y',
                           indexer=self.busrocof,
                           export=False,
                           info='Bus ROCOF',
                           unit='p.u.',
                           )

        self.Pref.e_str += '- Kwd * df'


class REECA1I(REECA1IData, REECA1IModel):
    def __init__(self, system, config):
        REECA1IData.__init__(self)
        REECA1IModel.__init__(self, system, config)
