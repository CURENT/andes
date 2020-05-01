"""
Power system stabilizer models.
"""
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.var import Algeb, ExtAlgeb, ExtState
from andes.core.block import Lag2ndOrd, LeadLag2ndOrd, LeadLag, WashoutOrLag, Gain
from andes.core.service import ExtService, OptionalSelect, DeviceFinder
from andes.core.discrete import Switcher, Limiter, Derivative
from andes.core.model import ModelData, Model
from collections import OrderedDict

import logging

logger = logging.getLogger(__name__)


class IEEESTData(ModelData):
    def __init__(self):
        super(IEEESTData, self).__init__()

        self.avr = IdxParam(info='Exciter idx', mandatory=True)
        self.MODE = NumParam(info='Input signal', mandatory=True)

        self.busr = IdxParam(info='Optional remote bus idx', model='Bus')
        self.busf = IdxParam(info='BusFreq idx for mode 2', model='BusFreq')

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

        self.VCU = NumParam(default=999, tex_name='V_{CU}', vrange=(1, 1.2), non_zero=True,
                            unit='p.u.', info='Upper enabling bus voltage')
        self.VCL = NumParam(default=-999, tex_name='V_{CL}', vrange=(0.8, 1), non_zero=True,
                            unit='p.u.', info='Upper enabling bus voltage')


class PSSBase(Model):
    """
    PSS base model
    """
    def __init__(self, system, config):
        super().__init__(system, config)
        self.group = 'PSS'
        self.flags.update({'tds': True})

        # retrieve indices of connected generator, bus, and bus freq
        self.syn = ExtParam(model='Exciter', src='syn', indexer=self.avr, export=False,
                            info='Retrieved generator idx')
        self.bus = ExtParam(model='SynGen', src='bus', indexer=self.syn, export=False,
                            info='Retrieved bus idx')

        self.buss = OptionalSelect(self.busr, self.bus, info='selected bus (bus or busr)')

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

        # from BusFreq
        self.f = ExtState(model='FreqMeasurement', src='f', indexer=self.busfreq, export=False,
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
        self.config.add_extra('_alt', {'freq_model': ('BusFreq', )})

        self.busf.model = self.config.freq_model

        self.dv = Derivative(self.v)

        self.SnSb = ExtService(model='SynGen', src='M', indexer=self.syn, attr='pu_coeff',
                               info='Machine base to sys base factor for power',
                               tex_name='(Sn/Sb)')

        self.SW = Switcher(u=self.MODE,
                           options=[1, 2, 3, 4, 5, 6],
                           )

        self.signal = Algeb(tex_name='S_{in}',
                            info='Input signal',
                            )

        self.signal.v_str = 'SW_s0*(omega-1) + SW_s1*0 + SW_s2*(tm0/SnSb) + ' \
                            'SW_s3*(tm-tm0) + SW_s4*v + SW_s5*0'

        self.signal.e_str = 'SW_s0*(omega-1) + SW_s1*(f-1) + SW_s2*(te/SnSb) + ' \
                            'SW_s3*(tm-tm0) + SW_s4*v + SW_s5*dv_v - signal'

        self.F1 = Lag2ndOrd(u=self.signal, K=1, T1=self.A1, T2=self.A2)

        self.F2 = LeadLag2ndOrd(u=self.F1_y, T1=self.A3, T2=self.A4, T3=self.A5, T4=self.A6)

        self.LL1 = LeadLag(u=self.F2_y, T1=self.T1, T2=self.T2)

        self.LL2 = LeadLag(u=self.LL1_y, T1=self.T3, T2=self.T4)

        self.Vks = Gain(u=self.LL2_y, K=self.KS)

        self.WO = WashoutOrLag(u=self.Vks_y, T=self.T6, K=self.T5, name='WO', zero_out=True)  # WO_y == Vss

        self.VLIM = Limiter(u=self.WO_y, lower=self.LSMIN, upper=self.LSMAX, info='Vss limiter')

        self.Vss = Algeb(tex_name='V_{ss}', info='Voltage output before output limiter',
                         e_str='VLIM_zi * WO_y + VLIM_zu * LSMAX + VLIM_zl * LSMIN - Vss')

        self.OLIM = Limiter(u=self.v, lower=self.VCL, upper=self.VCU, info='output limiter')

        self.vsout.e_str = 'OLIM_zi * Vss - vsout'


class IEEEST(IEEESTData, IEEESTModel):
    """
    IEEEST stabilizer model. Automatically adds frequency measurement devices if not provided.

    Input signals (MODE):

    1 (s0) - Rotor speed deviation (p.u.),
    2 (s1) - Bus frequency deviation (p.u.),
    3 (s2) - Generator P electrical in Gen MVABase (p.u.),
    4 (s3) - Generator accelerating power (p.u.),
    5 (s4) - Bus voltage (p.u.),
    6 (s5) - Derivative of p.u. bus voltage.

    Blocks are named `F1`, `F2`, `LL1`, `LL2` and `WO` in sequence.
    Two limiters are named `VLIM` and `OLIM` in sequence.
    """

    def __init__(self, system, config):
        IEEESTData.__init__(self)
        IEEESTModel.__init__(self, system, config)
