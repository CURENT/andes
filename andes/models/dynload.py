"""
Module for dynamic loads.
"""

from andes.core.model import ModelData, Model  # NOQA
from andes.core.param import IdxParam  # NOQA


class PQRandData(ModelData):
    """
    Data for random PQ model.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.pq = IdxParam(mandatory=True,
                           info='idx of static PQ to replace',
                           )

        # Timestamp-dependent stuff
        # use `Le(dae_t, t_max) * Ge(dae_t, t_min) * (Load Scale Equation)`
