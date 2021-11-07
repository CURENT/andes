"""
Bus rate-of-change of frequency measurement based on BusFreq.
"""

from andes.core import NumParam, Washout
from andes.models.measurement import BusFreq


class BusROCOF(BusFreq):
    """
    Bus frequency and ROCOF measurement.

    The ROCOF output variable is ``Wf_y``.
    """

    def __init__(self, system, config):
        BusFreq.__init__(self, system, config)
        self.Tr = NumParam(default=0.1,
                           info="frequency washout time constant",
                           tex_name='T_r')

        self.Wf = Washout(u=self.f,
                          K=1,
                          T=self.Tr,
                          info='frequency washout yielding ROCOF',
                          )
