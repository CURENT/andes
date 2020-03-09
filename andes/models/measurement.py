"""
Measurement device classes
"""

from andes.core.param import IdxParam, NumParam
from andes.core.model import Model, ModelData
from andes.core.var import ExtAlgeb, Algeb, State
from andes.core.block import Washout
from andes.core.service import ConstService, ExtService
import logging
logger = logging.getLogger(__name__)


class BusFreq(ModelData, Model):
    """
    Bus frequency measurement.
    """
    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.flags['tds'] = True
        self.group = 'FreqMeasurement'

        # Parameters
        self.bus = IdxParam(info="bus idx", mandatory=True)
        self.Tf = NumParam(default=0.1, info="angle washout time constant", unit="sec",
                           tex_name='T_f')
        self.Tw = NumParam(default=1.0, info="lag time constant for frequency", unit="sec",
                           tex_name='T_w')
        self.fn = NumParam(default=60.0, info="nominal frequency",
                           tex_name='f_n')

        # Variables
        self.iwn = ConstService(v_str='u / (2 * pi * fn)', tex_name=r'1/\omega_n')
        self.a0 = ExtService(src='a',
                             model='Bus',
                             indexer=self.bus,
                             tex_name=r'\theta_0',
                             info='initial phase angle',
                             )
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          )
        self.ad = Algeb(info='angle deviation',
                        tex_name=r'\theta_d',
                        v_str='0',
                        e_str='(a - a0) - ad',
                        )
        self.WO = Washout(u=self.ad,
                          K=self.iwn,
                          T=self.Tf,
                          info='angle washout yielding frequency',
                          )
        self.w = State(info='frequency after output delay',
                       tex_name=r'\omega',
                       v_str='1',
                       e_str='(1 + WO_y - w) / Tw',
                       )


class BusROCOF(BusFreq):
    """
    Bus frequency and ROCOF measurement.

    The ROCOF output variable is ``Wf_y``.
    """
    def __init__(self, system, config):
        BusFreq.__init__(self, system, config)
        self.Tr = NumParam(default=0.1, info="frequency washout time constant", tex_name='T_r')

        self.Wf = Washout(u=self.w,
                          K=1,
                          T=self.Tr,
                          info='frequency washout yielding ROCOF',
                          )
