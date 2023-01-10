"""
Module for open-loop PID Controllers.
"""

import logging

from andes.core.model import ModelData, Model
from andes.core.param import NumParam, IdxParam
from andes.core.block import PIDController
from andes.core.var import ExtAlgeb
from andes.core.service import ConstService

logger = logging.getLogger(__name__)


class OLPIData(ModelData):
    """
    Data for open-loop PI controller..
    """

    def __init__(self):
        super().__init__()
        self.gov = IdxParam(model='TurbineGov',
                            info='Turbine governor idx',
                            mandatory=True,
                            )
        self.kP = NumParam(info='PID proportional coeff.',
                           tex_name='k_P',
                           default=1,
                           )
        self.kI = NumParam(info='PID integrative coeff.',
                           tex_name='k_I',
                           default=1,
                           )
        self.kD = NumParam(info='PID derivative coeff.',
                           tex_name='k_D',
                           default=0,
                           )
        self.tD = NumParam(info='PID derivative time constant coeff.',
                           tex_name='t_D',
                           default=0,
                           )


class OLPIModel(Model):
    """
    Implementation for open-loop PI controller.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'Experimental'
        self.flags.tds = True

        self.pout = ExtAlgeb(model='TurbineGov', src='pout', indexer=self.gov,
                             tex_name='P_{out}',
                             info='Turbine governor output',
                             is_input=True,
                             )
        self.pout0 = ConstService(v_str='pout',
                                  tex_name='P_{out0}',
                                  info='initial turbine governor output',
                                  )
        self.PID = PIDController(u=self.pout, kp=self.kP, ki=self.kI,
                                 kd=self.kD, Td=self.tD,
                                 tex_name='PID', info='PID', name='PID',
                                 ref=self.pout0,
                                 )


class OLPI(OLPIData, OLPIModel):
    r"""
    Open-loop PI controller that takes Turbine Governor output power as input.

    ```
        ┌────────────────────┐
        │      ki     skd    │
    u -> │kp + ─── + ───────  │ -> y
        │      s    1 + sTd  │
        └────────────────────┘
    ```
    """

    def __init__(self, system, config):
        OLPIData.__init__(self)
        OLPIModel.__init__(self, system, config)
