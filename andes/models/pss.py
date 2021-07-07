"""
Power system stabilizer models.
"""
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.var import Algeb, ExtAlgeb, ExtState
from andes.core.block import Lag2ndOrd, LeadLag2ndOrd, LeadLag, WashoutOrLag, Gain, Lag, GainLimiter
from andes.core.service import ExtService, DataSelect, DeviceFinder, Replace, ConstService
from andes.core.discrete import Switcher, Limiter, Derivative
from andes.core.model import ModelData, Model
from collections import OrderedDict
import numpy as np

import logging

logger = logging.getLogger(__name__)


class PSSBaseData(ModelData):
    def __init__(self):
        super().__init__()
        self.avr = IdxParam(info='Exciter idx', mandatory=True, model='Exciter')


class IEEESTData(PSSBaseData):
    def __init__(self):
        super().__init__()
        self.MODE = NumParam(info='Input signal', mandatory=True)

        self.busr = IdxParam(info='Optional remote bus idx', model='Bus', default=None)
        self.busf = IdxParam(info='BusFreq idx for mode 2', model='BusFreq', default=None)

        self.A1 = NumParam(default=1, tex_name='A_1', info='filter time const. (pole)')
        self.A2 = NumParam(default=1, tex_name='A_2', info='filter time const. (pole)')
        self.A3 = NumParam(default=1, tex_name='A_3', info='filter time const. (pole)')
        self.A4 = NumParam(default=1, tex_name='A_4', info='filter time const. (pole)')
        self.A5 = NumParam(default=1, tex_name='A_5', info='filter time const. (zero)')
        self.A6 = NumParam(default=1, tex_name='A_6', info='filter time const. (zero)')

        self.T1 = NumParam(default=1, tex_name='T_1', vrange=(0, 10), info='first leadlag time const. (zero)')
        self.T2 = NumParam(default=1, tex_name='T_2', vrange=(0, 10), info='first leadlag time const. (pole)')
        self.T3 = NumParam(default=1, tex_name='T_3', vrange=(0, 10), info='second leadlag time const. (pole)')
        self.T4 = NumParam(default=1, tex_name='T_4', vrange=(0, 10), info='second leadlag time const. (pole)')
        self.T5 = NumParam(default=1, tex_name='T_5', vrange=(0, 10), info='washout time const. (zero)')
        self.T6 = NumParam(default=1, tex_name='T_6', vrange=(0.04, 2), info='washout time const. (pole)')

        self.KS = NumParam(default=1, tex_name='K_S', info='Gain before washout')
        self.LSMAX = NumParam(default=0.3, tex_name='L_{SMAX}', vrange=(0, 0.3), info='Max. output limit')
        self.LSMIN = NumParam(default=-0.3, tex_name='L_{SMIN}', vrange=(-0.3, 0), info='Min. output limit')

        self.VCU = NumParam(default=999, tex_name='V_{CU}', vrange=(1, 1.2),
                            unit='p.u.', info='Upper enabling bus voltage')
        self.VCL = NumParam(default=-999, tex_name='V_{CL}', vrange=(0., 1),
                            unit='p.u.', info='Upper enabling bus voltage')


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

        self.busfreq = DeviceFinder(self.busf, link=self.buss, idx_name='bus')

        # from SynGen
        self.Sn = ExtParam(model='SynGen', src='Sn', indexer=self.syn, tex_name='S_n',
                           info='Generator power base', export=False)

        self.omega = ExtState(model='SynGen', src='omega', indexer=self.syn,
                              tex_name=r'\omega', info='Generator speed', unit='p.u.',
                              )

        self.tm0 = ExtService(model='SynGen', src='tm', indexer=self.syn,
                              tex_name=r'\tau_{m0}', info='Initial mechanical input',
                              )
        self.tm = ExtAlgeb(model='SynGen', src='tm', indexer=self.syn,
                           tex_name=r'\tau_m', info='Generator mechanical input',
                           )
        self.te = ExtAlgeb(model='SynGen', src='te', indexer=self.syn,
                           tex_name=r'\tau_e', info='Generator electrical output',
                           )
        # from Bus
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.buss, tex_name=r'V',
                          info='Bus (or busr, if given) terminal voltage',
                          )
        self.v0 = ExtService(model='Bus', src='v', indexer=self.buss, tex_name="V_0",
                             info='Initial bus voltage',
                             )

        # from BusFreq
        self.f = ExtAlgeb(model='FreqMeasurement', src='f', indexer=self.busfreq, export=False,
                          info='Bus frequency')

        # from Exciter
        self.vi = ExtAlgeb(model='Exciter', src='vi', indexer=self.avr, tex_name='v_i',
                           info='Exciter input voltage',
                           e_str='u * vsout')

        self.vsout = Algeb(info='PSS output voltage to exciter',
                           tex_name='v_{sout}',
                           )  # `self.vsout.e_str` to be provided by specific models


class IEEESTModel(PSSBase):
    """
    IEEEST Stabilizer equation.
    """

    def __init__(self, system, config):
        PSSBase.__init__(self, system, config)

        self.config.add(OrderedDict([('freq_model', 'BusFreq')]))
        self.config.add_extra('_help', {'freq_model': 'default freq. measurement model'})
        self.config.add_extra('_alt', {'freq_model': ('BusFreq',)})

        self.busf.model = self.config.freq_model

        self.dv = Derivative(self.v, tex_name='dV/dt', info='Finite difference of bus voltage')

        self.SnSb = ExtService(model='SynGen', src='M', indexer=self.syn, attr='pu_coeff',
                               info='Machine base to sys base factor for power',
                               tex_name='(Sb/Sn)')

        self.SW = Switcher(u=self.MODE,
                           options=[0, 1, 2, 3, 4, 5, 6],
                           )

        self.sig = Algeb(tex_name='S_{ig}',
                         info='Input signal',
                         )

        self.sig.v_str = 'SW_s1*(omega-1) + SW_s2*0 + SW_s3*(tm0/SnSb) + ' \
                         'SW_s4*(tm-tm0) + SW_s5*v + SW_s6*0'

        self.sig.e_str = 'SW_s1*(omega-1) + SW_s2*(f-1) + SW_s3*(te/SnSb) + ' \
                         'SW_s4*(tm-tm0) + SW_s5*v + SW_s6*dv_v - sig'

        self.F1 = Lag2ndOrd(u=self.sig, K=1, T1=self.A1, T2=self.A2)

        self.F2 = LeadLag2ndOrd(u=self.F1_y, T1=self.A3, T2=self.A4, T3=self.A5, T4=self.A6, zero_out=True)

        self.LL1 = LeadLag(u=self.F2_y, T1=self.T1, T2=self.T2, zero_out=True)

        self.LL2 = LeadLag(u=self.LL1_y, T1=self.T3, T2=self.T4, zero_out=True)

        self.Vks = Gain(u=self.LL2_y, K=self.KS)

        self.WO = WashoutOrLag(u=self.Vks_y, T=self.T6, K=self.T5, name='WO', zero_out=True)  # WO_y == Vss

        self.VLIM = Limiter(u=self.WO_y, lower=self.LSMIN, upper=self.LSMAX, info='Vss limiter')

        self.Vss = Algeb(tex_name='V_{ss}', info='Voltage output before output limiter',
                         e_str='VLIM_zi * WO_y + VLIM_zu * LSMAX + VLIM_zl * LSMIN - Vss')

        self.OLIM = Limiter(u=self.v, lower=self.VCLr, upper=self.VCUr, info='output limiter')

        self.vsout.e_str = 'OLIM_zi * Vss - vsout'


class IEEEST(IEEESTData, IEEESTModel):
    """
    IEEEST stabilizer model. Automatically adds frequency measurement devices if not provided.

    Input signals (MODE):

    1 - Rotor speed deviation (p.u.),
    2 - Bus frequency deviation (*) (p.u.),
    3 - Generator P electrical in Gen MVABase (p.u.),
    4 - Generator accelerating power (p.u.),
    5 - Bus voltage (p.u.),
    6 - Derivative of p.u. bus voltage.

    (*) Due to the frequency measurement implementation difference,
    mode 2 is likely to yield different results across software.

    Blocks are named `F1`, `F2`, `LL1`, `LL2` and `WO` in sequence.
    Two limiters are named `VLIM` and `OLIM` in sequence.
    """

    def __init__(self, system, config):
        IEEESTData.__init__(self)
        IEEESTModel.__init__(self, system, config)


class ST2CUTData(PSSBaseData):
    def __init__(self):
        PSSBaseData.__init__(self)
        self.MODE = NumParam(info='Input signal 1', mandatory=True)
        self.busr = NumParam(info='Remote bus 1')
        self.busf = IdxParam(info='BusFreq idx for signal 1 mode 2',
                             model='BusFreq', )

        self.MODE2 = NumParam(info='Input signal 2')
        self.busr2 = NumParam(info='Remote bus 2')
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

        self.busfreq2 = DeviceFinder(self.busf2, link=self.buss2, idx_name='bus')

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
