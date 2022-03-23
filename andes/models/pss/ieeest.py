import logging
from collections import OrderedDict

from andes.core import NumParam, IdxParam, ExtService, Switcher, Algeb, LeadLag, Limiter
from andes.core.block import Lag2ndOrd, LeadLag2ndOrd, Gain, WashoutOrLag
from andes.core.discrete import Derivative
from andes.models.pss.pssbase import PSSBaseData, PSSBase

logger = logging.getLogger(__name__)


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

    1. Rotor speed deviation (p.u.),
    2. Bus frequency deviation (p.u.) (*),
    3. Generator P electrical in Gen MVABase (p.u.),
    4. Generator accelerating power (p.u.),
    5. Bus voltage (p.u.),
    6. Derivative of p.u. bus voltage.

    (*) Due to the frequency measurement implementation difference,
    mode 2 is likely to yield different results across software.

    .. note::

        Blocks are named `F1`, `F2`, `LL1`, `LL2` and `WO` in sequence.
        Two limiters are named `VLIM` and `OLIM` in sequence.
    """

    def __init__(self, system, config):
        IEEESTData.__init__(self)
        IEEESTModel.__init__(self, system, config)
