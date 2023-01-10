"""
Module for PI Controllers.
"""

import logging

from andes.core.model import ModelData, Model
from andes.core.param import NumParam, IdxParam
from andes.core.block import PIController
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
        self.kP = NumParam(info='PI proportional coeff.',
                           tex_name='k_P',
                           default=1,
                           )
        self.kI = NumParam(info='PI integrative coeff.',
                           tex_name='k_I',
                           default=1,
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
                             tex_name=r'\tau_m', info='Turbine governor output',
                             is_input=True,
                             )
        self.pout0 = ConstService(v_str='pout',
                               tex_name='P_{out0}',
                               info='initial turbine governor output',
                               )
        self.PI = PIController(u=self.pout, kp=self.kP, ki=self.kI,
                               tex_name='PI', info='PI', name='PI',
                               ref=self.pout0,
                               )


class OLPI(OLPIData, OLPIModel):
    r"""
    Open-loop PI controller that takes Turbine Governor output as input.

    ```
        ┌─────────┐
        │      ki │
    u -> │kp + ─── │ -> y
        │      s  │
        └─────────┘
    ```
    """

    def __init__(self, system, config):
        OLPIData.__init__(self)
        OLPIModel.__init__(self, system, config)
