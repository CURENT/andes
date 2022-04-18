"""
Power system stabilizer base models.
"""

import logging

import numpy as np

from andes.core import (ModelData, IdxParam, Model, ExtParam,
                        ExtState, ExtService, ExtAlgeb, Algeb)
from andes.core.service import Replace, DataSelect, DeviceFinder

logger = logging.getLogger(__name__)


class PSSBaseData(ModelData):
    def __init__(self):
        super().__init__()
        self.avr = IdxParam(info='Exciter idx', mandatory=True, model='Exciter')


class PSSBase(Model):
    """
    PSS base model.
    """

    def __init__(self, system, config):
        super().__init__(system, config)
        self.group = 'PSS'
        self.flags.update({'tds': True})

        self.VCUr = Replace(self.VCU, lambda x: np.equal(x, 0.0), 999)
        self.VCLr = Replace(self.VCL, lambda x: np.equal(x, 0.0), -999)

        # retrieve indices of connected generator, bus, and bus freq
        self.syn = ExtParam(model='Exciter', src='syn', indexer=self.avr, export=False,
                            info='Retrieved generator idx', vtype=str)

        self.bus = ExtParam(model='SynGen', src='bus', indexer=self.syn, export=False,
                            info='Retrieved bus idx', vtype=str, default=None,
                            )

        self.buss = DataSelect(self.busr, self.bus, info='selected bus (bus or busr)')

        self.busfreq = DeviceFinder(self.busf, link=self.buss, idx_name='bus',
                                    default_model='BusFreq')

        # from SynGen
        self.Sn = ExtParam(model='SynGen', src='Sn', indexer=self.syn, tex_name='S_n',
                           info='Generator power base', export=False)

        self.omega = ExtState(model='SynGen', src='omega', indexer=self.syn,
                              tex_name=r'\omega', info='Generator speed', unit='p.u.',
                              is_input=True,
                              )

        self.tm0 = ExtService(model='SynGen', src='tm', indexer=self.syn,
                              tex_name=r'\tau_{m0}', info='Initial mechanical input',
                              )
        self.tm = ExtAlgeb(model='SynGen', src='tm', indexer=self.syn,
                           tex_name=r'\tau_m', info='Generator mechanical input',
                           is_input=True,
                           )
        self.te = ExtAlgeb(model='SynGen', src='te', indexer=self.syn,
                           tex_name=r'\tau_e', info='Generator electrical output',
                           is_input=True,
                           )
        # from Bus
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.buss, tex_name=r'V',
                          info='Bus (or busr, if given) terminal voltage',
                          is_input=True,
                          )
        self.v0 = ExtService(model='Bus', src='v', indexer=self.buss, tex_name="V_0",
                             info='Initial bus voltage',
                             )

        # from BusFreq
        self.f = ExtAlgeb(model='FreqMeasurement', src='f', indexer=self.busfreq, export=False,
                          info='Bus frequency',
                          is_input=True,
                          )

        # from Exciter
        self.vi = ExtAlgeb(model='Exciter', src='vi', indexer=self.avr, tex_name='v_i',
                           info='Exciter input voltage',
                           e_str='u * vsout',
                           ename='Vi',
                           tex_ename='V_i',
                           is_input=True,
                           )

        self.vsout = Algeb(info='PSS output voltage to exciter',
                           tex_name='v_{sout}',
                           is_output=True,
                           )  # `self.vsout.e_str` to be provided by specific models
