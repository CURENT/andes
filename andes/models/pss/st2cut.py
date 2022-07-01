import logging
from collections import OrderedDict

import numpy as np

from andes.core import (NumParam, IdxParam, ExtAlgeb, ExtService,
                        Switcher, Algeb, Lag, LeadLag, ConstService, Limiter)
from andes.core.block import WashoutOrLag, GainLimiter
from andes.core.discrete import Derivative
from andes.core.service import DataSelect, DeviceFinder
from andes.models.pss.pssbase import PSSBaseData, PSSBase

logger = logging.getLogger(__name__)


class ST2CUTData(PSSBaseData):
    def __init__(self):
        PSSBaseData.__init__(self)
        self.MODE = NumParam(info='Input signal 1', mandatory=True)
        self.busr = IdxParam(info='Remote bus 1')
        self.busf = IdxParam(info='BusFreq idx for signal 1 mode 2',
                             model='BusFreq', )

        self.MODE2 = NumParam(info='Input signal 2')
        self.busr2 = IdxParam(info='Remote bus 2')
        self.busf2 = IdxParam(info='BusFreq idx for signal 2 mode 2', model='BusFreq')

        self.K1 = NumParam(default=1, tex_name='K_1',
                           info='Transducer 1 gain',
                           vrange=(0, 10),
                           )
        self.K2 = NumParam(default=1, tex_name='K_2',
                           info='Transducer 2 gain',
                           vrange=(0, 10),
                           )
        self.T1 = NumParam(default=1, tex_name='T_1',
                           info='Transducer 1 time const.',
                           vrange=(0, 10),
                           )
        self.T2 = NumParam(default=1, tex_name='T_2',
                           info='Transducer 2 time const.',
                           vrange=(0, 10),
                           )

        self.T3 = NumParam(default=1, tex_name='T_3',
                           info='Washout int. time const.',
                           vrange=(0, 10),
                           )
        self.T4 = NumParam(default=0.2, tex_name='T_4',
                           info='Washout delay time const.',
                           vrange=(0.05, 10),
                           )

        self.T5 = NumParam(default=1, tex_name='T_5',
                           info='Leadlag 1 time const. (1)',
                           vrange=(0, 10),
                           )
        self.T6 = NumParam(default=0.5, tex_name='T_6',
                           info='Leadlag 1 time const. (2)',
                           vrange=(0, 2),
                           )

        self.T7 = NumParam(default=1, tex_name='T_7',
                           info='Leadlag 2 time const. (1)',
                           vrange=(0, 10),
                           )
        self.T8 = NumParam(default=1, tex_name='T_8',
                           info='Leadlag 2 time const. (2)',
                           vrange=(0, 10),
                           )

        self.T9 = NumParam(default=1, tex_name='T_9',
                           info='Leadlag 3 time const. (1)',
                           vrange=(0, 2),
                           )
        self.T10 = NumParam(default=0.2, tex_name='T_{10}',
                            info='Leadlag 3 time const. (2)',
                            vrange=(0, 2),
                            )

        self.LSMAX = NumParam(default=0.3, tex_name='L_{SMAX}', vrange=(0, 0.3), info='Max. output limit')
        self.LSMIN = NumParam(default=-0.3, tex_name='L_{SMIN}', vrange=(-0.3, 0), info='Min. output limit')

        self.VCU = NumParam(default=999, tex_name='V_{CU}', vrange=(1, 1.2),
                            unit='p.u.', info='Upper enabling bus voltage')

        self.VCL = NumParam(default=-999, tex_name='V_{CL}', vrange=(-0.1, 1),
                            unit='p.u.', info='Upper enabling bus voltage')


class ST2CUTModel(PSSBase):
    def __init__(self, system, config):
        PSSBase.__init__(self, system, config)

        # ALL THE FOLLOWING IS FOR INPUT 2
        # retrieve indices of bus and bus freq
        self.buss2 = DataSelect(self.busr2, self.bus, info='selected bus (bus or busr)')

        self.busfreq2 = DeviceFinder(self.busf2, link=self.buss2, idx_name='bus',
                                     default_model='BusFreq', info='bus frequency idx')

        # from Bus
        self.v2 = ExtAlgeb(model='Bus', src='v', indexer=self.buss2, tex_name=r'V',
                           info='Bus (or busr2, if given) terminal voltage',
                           )

        # from BusFreq 2
        self.f2 = ExtAlgeb(model='FreqMeasurement', src='f', indexer=self.busfreq2, export=False,
                           info='Bus frequency 2')

        # Config
        self.config.add(OrderedDict([('freq_model', 'BusFreq')]))
        self.config.add_extra('_help', {'freq_model': 'default freq. measurement model'})
        self.config.add_extra('_alt', {'freq_model': ('BusFreq',)})

        self.busf.model = self.config.freq_model
        self.busf2.model = self.config.freq_model

        # input signal switch
        self.dv = Derivative(self.v)
        self.dv2 = Derivative(self.v2)

        self.SnSb = ExtService(model='SynGen', src='M', indexer=self.syn, attr='pu_coeff',
                               info='Machine base to sys base factor for power',
                               tex_name='(Sb/Sn)')

        self.SW = Switcher(u=self.MODE,
                           options=[0, 1, 2, 3, 4, 5, 6, np.nan],
                           )
        self.SW2 = Switcher(u=self.MODE2,
                            options=[0, 1, 2, 3, 4, 5, 6, np.nan],
                            )

        # Input signals
        self.sig = Algeb(tex_name='S_{ig}',
                         info='Input signal',
                         )
        self.sig.v_str = 'SW_s1*(omega-1) + SW_s2*0 + SW_s3*(tm0/SnSb) + ' \
                         'SW_s4*(tm-tm0) + SW_s5*v + SW_s6*0'
        self.sig.e_str = 'SW_s1*(omega-1) + SW_s2*(f-1) + SW_s3*(te/SnSb) + ' \
                         'SW_s4*(tm-tm0) + SW_s5*v + SW_s6*dv_v - sig'

        self.sig2 = Algeb(tex_name='S_{ig2}',
                          info='Input signal 2',
                          )
        self.sig2.v_str = 'SW2_s1*(omega-1) + SW2_s2*0 + SW2_s3*(tm0/SnSb) + ' \
                          'SW2_s4*(tm-tm0) + SW2_s5*v2 + SW2_s6*0'
        self.sig2.e_str = 'SW2_s1*(omega-1) + SW2_s2*(f2-1) + SW2_s3*(te/SnSb) + ' \
                          'SW2_s4*(tm-tm0) + SW2_s5*v2 + SW2_s6*dv2_v - sig2'

        self.L1 = Lag(u=self.sig,
                      K=self.K1,
                      T=self.T1,
                      info='Transducer 1',
                      )
        self.L2 = Lag(u=self.sig2,
                      K=self.K2,
                      T=self.T2,
                      info='Transducer 2',
                      )
        self.IN = Algeb(tex_name='I_N',
                        info='Sum of inputs',
                        v_str='L1_y + L2_y',
                        e_str='L1_y + L2_y - IN',
                        )

        self.WO = WashoutOrLag(u=self.IN,
                               K=self.T3,
                               T=self.T4,
                               )

        self.LL1 = LeadLag(u=self.WO_y,
                           T1=self.T5,
                           T2=self.T6,
                           zero_out=True,
                           )

        self.LL2 = LeadLag(u=self.LL1_y,
                           T1=self.T7,
                           T2=self.T8,
                           zero_out=True,
                           )

        self.LL3 = LeadLag(u=self.LL2_y,
                           T1=self.T9,
                           T2=self.T10,
                           zero_out=True,
                           )

        self.VSS = GainLimiter(u=self.LL3_y,
                               K=1,
                               R=1,
                               lower=self.LSMIN,
                               upper=self.LSMAX
                               )

        self.VOU = ConstService(v_str='VCUr + v0')
        self.VOL = ConstService(v_str='VCLr + v0')

        self.OLIM = Limiter(u=self.v, lower=self.VOL, upper=self.VOU, info='output limiter')

        self.vsout.e_str = 'OLIM_zi * VSS_y - vsout'


class ST2CUT(ST2CUTData, ST2CUTModel):
    """
    ST2CUT stabilizer model. Automatically adds frequency measurement devices if not provided.

    Input signals (MODE and MODE2):

    0 - Disable input signal
    1 (s1) - Rotor speed deviation (p.u.),
    2 (s2) - Bus frequency deviation (*) (p.u.),
    3 (s3) - Generator P electrical in Gen MVABase (p.u.),
    4 (s4) - Generator accelerating power (p.u.),
    5 (s5) - Bus voltage (p.u.),
    6 (s6) - Derivative of p.u. bus voltage.

    (*) Due to the frequency measurement implementation difference,
    mode 2 is likely to yield different results across software.

    Blocks are named `LL1`, `LL2`, `LL3`, `LL4` in sequence.
    Two limiters are named `VSS_lim` and `OLIM` in sequence.
    """

    def __init__(self, system, config):
        ST2CUTData.__init__(self)
        ST2CUTModel.__init__(self, system, config)
