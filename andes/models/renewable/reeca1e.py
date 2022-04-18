"""
REECA1 model with inertia emulation.
"""

from andes.core.param import IdxParam, NumParam
from andes.core.service import DeviceFinder
from andes.core.var import ExtAlgeb, ExtState
from andes.models.renewable.reeca1 import REECA1Data, REECA1Model


class REECA1EData(REECA1Data):
    """
    Data for REECA1E.
    """

    def __init__(self):
        REECA1Data.__init__(self)
        self.Kf = NumParam(default=0.0,
                           info='gain for frequency deviation',
                           tex_name='K_{df}',
                           )

        self.Kdf = NumParam(default=0.0,
                            info='gain for rate-of-change of frequency',
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
                                     default_model='BusROCOF',
                                     )

        self.df = ExtAlgeb(model='FreqMeasurement',
                           src='WO_y',
                           indexer=self.busrocof,
                           export=False,
                           info='Bus frequency deviation',
                           )

        self.dfdt = ExtAlgeb(model='FreqMeasurement',
                             src='Wf_y',
                             indexer=self.busrocof,
                             export=False,
                             info='Bus ROCOF',
                             unit='p.u.',
                             )

        self.Pref.e_str += '- Kdf * dfdt - Kf * df'


class REECA1E(REECA1EData, REECA1EModel):
    """
    REGCA1 with inertia emulation and primary frequency droop.
    Measurements are based on frequency measurement model.

    Bus ROCOF obtained from ``BusROCOF`` devices.
    """

    def __init__(self, system, config):
        REECA1EData.__init__(self)
        REECA1EModel.__init__(self, system, config)


class REECA1GData(REECA1Data):
    """
    Data for REECA1G.
    """

    def __init__(self):
        REECA1Data.__init__(self)
        self.Kf = NumParam(default=0.0,
                           info='gain for frequency deviation',
                           tex_name='K_{df}',
                           )

        self.sg = IdxParam(info='synchronous gen idx',
                           model='Synchronous',
                           default=None,
                           mandatory=True,
                           )


class REECA1GModel(REECA1Model):
    """
    Model for REECA1G. See docstring for REECA1G.
    """

    def __init__(self, system, config):
        REECA1Model.__init__(self, system, config)

        self.omega = ExtState(model='SynGen',
                              src='omega',
                              indexer=self.sg,
                              export=False,
                              info='generator speed',
                              unit='pu',
                              )

        self.Pref.e_str += '- Kf * (omega - 1)'


class REECA1G(REECA1GData, REECA1GModel):
    """
    REECA1G is a variant of REECA1E.

    REECA1G uses speed from synchronous generators.

    The application of this model is limited because it is uncommon
    to connect a SynGen on the same bus as a RenGen.
    """
    def __init__(self, system, config):
        REECA1GData.__init__(self)
        REECA1GModel.__init__(self, system, config)
