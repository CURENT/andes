"""
REECA1 model with inertia emulation.
"""

from andes.core.param import NumParam, IdxParam
from andes.core.var import ExtAlgeb
from andes.core.service import DeviceFinder

from andes.models.renewable.reeca1 import REECA1Data, REECA1Model


class REECA1EData(REECA1Data):
    """
    Data for REECA1E.
    """
    def __init__(self):
        REECA1Data.__init__(self)
        self.Kdf = NumParam(default=0.0,
                            info='gain for frequency derivative',
                            tex_name='K_{df}',
                            )
        self.busroc = IdxParam(info='Optional BusROCOF device idx',
                               model='BusROCOF',
                               default=None,
                               )


class REECA1EModel(REECA1Model):
    """
    Model for REECA1E.
    """
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

        self.Pref.e_str += '- Kdf * df'


class REECA1E(REECA1EData, REECA1EModel):
    """
    REGCA1 with inertia emulation.

    Bus ROCOF obtained from `BusROCOF` devices.
    """
    def __init__(self, system, config):
        REECA1EData.__init__(self)
        REECA1EModel.__init__(self, system, config)
