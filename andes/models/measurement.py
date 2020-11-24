"""
Measurement device classes
"""

from andes.core.param import IdxParam, NumParam
from andes.core.model import Model, ModelData
from andes.core.var import ExtAlgeb, Algeb, State  # NOQA
from andes.core.block import Washout, Lag
from andes.core.service import ConstService, ExtService
import logging
logger = logging.getLogger(__name__)


class BusFreq(ModelData, Model):
    """
    Bus frequency measurement.

    Bus frequency output variable is `f`.
    """
    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'FreqMeasurement'

        # Parameters
        self.bus = IdxParam(info="bus idx", mandatory=True)
        self.Tf = NumParam(default=0.02, info="input digital filter time const", unit="sec",
                           tex_name='T_f')
        self.Tw = NumParam(default=0.02, info="washout time const", unit="sec",
                           tex_name='T_w')
        self.fn = NumParam(default=60.0, info="nominal frequency", unit='Hz',
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
        self.L = Lag(u='(a-a0)',
                     T=self.Tf,
                     K=1,
                     info='digital filter',
                     )
        # the output `WO_y` is the frequency deviation in p.u.
        self.WO = Washout(u=self.L_y,
                          K=self.iwn,
                          T=self.Tw,
                          info='angle washout',
                          )
        self.WO_y.info = 'frequency deviation'
        self.WO_y.unit = 'p.u. (Hz)'

        self.f = Algeb(info='frequency output',
                       unit='p.u. (Hz)',
                       tex_name='f',
                       v_str='1',
                       e_str='1 + WO_y - f',
                       )


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


class PMUData(ModelData):
    """
    Phasor measurement unit data.
    """
    def __init__(self):
        ModelData.__init__(self)
        self.bus = IdxParam(info="bus idx", mandatory=True)

        self.Ta = NumParam(default=0.1, tex_name='T_a', info='angle filter time constant')
        self.Tv = NumParam(default=0.1, tex_name='T_v', info='voltage filter time constant')


class PMU(PMUData, Model):
    """
    Simple phasor measurement unit model.

    This model tracks the bus voltage magnitude and phase angle, each using
    a low-pass filter.
    """
    def __init__(self, system, config):
        PMUData.__init__(self)
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'PhasorMeasurement'

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          )

        self.am = State(tex_name=r'\theta_m', info='phase angle measurement',
                        unit='rad.', e_str='a - am', t_const=self.Ta, v_str='a',
                        )

        self.vm = State(tex_name='V_m', info='voltage magnitude measurement',
                        unit='p.u.(kV)', e_str='v - vm', t_const=self.Tv, v_str='v',
                        )
