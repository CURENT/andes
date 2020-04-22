"""
Power system stabilizer models
"""
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.var import Algeb, ExtAlgeb, ExtState
from andes.core.block import Lag2ndOrd, LeadLag2ndOrd, LeadLag, Washout
from andes.core.service import ConstService, ExtService
from andes.core.discrete import Switcher, Limiter
from andes.core.model import ModelData, Model

import logging

logger = logging.getLogger(__name__)


class IEEESTData(ModelData):
    def __init__(self):
        super(IEEESTData, self).__init__()

        self.avr = IdxParam(info='Exciter idx', mandatory=True)
        self.MODE = NumParam(info='Input signal selection', mandatory=True)
        self.BUSR = IdxParam(info='Remote bus idx (local if empty)')

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

        self.syn = ExtParam(model='Exciter', src='syn', indexer=self.avr, export=False,
                            info='Retrieved generator idx')
        self.bus = ExtParam(model='SynGen', src='bus', indexer=self.syn, export=False,
                            info='Retrieved bus idx')
        self.Sn = ExtParam(model='SynGen', src='Sn', indexer=self.syn, tex_name='S_n',
                           info='Generator power base', export=False)

        # from SynGen
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
        self.vf = ExtAlgeb(model='SynGen', src='vf', indexer=self.syn, tex_name='v_f',
                           info='Generator excitation voltage',
                           e_str='u * vsout')

        # from Bus  #TODO: implement the optional BUSR
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name=r'V',
                          info='Bus (or BUSR, if given) terminal voltage',
                          )

        # from Exciter
        self.vsout = Algeb(info='PSS output voltage to exciter',
                           tex_name='v_{sout}',
                           )  # `e_str` to be provided by specific models


class IEEESTModel(PSSBase):
    """
    IEEEST Stabilizer equation.
    """

    def __init__(self, system, config):
        super(IEEESTModel, self).__init__(system, config)

        self.KST5 = ConstService(v_str='KS * T5', tex_name='KS*T5')
        self.SW = Switcher(u=self.MODE,
                           options=[1, 2, 3, 4, 5, 6],
                           )

        self.signal = Algeb(tex_name='S_{in}',
                            info='Input signal',
                            )
        # input signals:
        # 1 (s0) - Rotor speed deviation (p.u.)
        # 2 (s1) - Bus frequency deviation (p.u.)                    # TODO: calculate freq without reimpl.
        # 3 (s2) - Generator electrical power in Gen MVABase (p.u.)  # TODO: allow using system.config.mva
        # 4 (s3) - Generator accelerating power (p.u.)
        # 5 (s4) - Bus voltage (p.u.)
        # 6 (s5) - Derivative of p.u. bus voltage                    # TODO: memory block for calc. of derivative

        self.signal.e_str = 'SW_s0 * (1-omega) + SW_s1 * 0 + SW_s2 * te + ' \
                            'SW_s3 * (tm-tm0) + SW_s4 *v + SW_s5 * 0 - signal'

        self.F1 = Lag2ndOrd(u=self.signal, K=1, T1=self.A1, T2=self.A2)

        self.F2 = LeadLag2ndOrd(u=self.F1_y, T1=self.A3, T2=self.A4, T3=self.A5, T4=self.A6)

        self.LL1 = LeadLag(u=self.F2_y, T1=self.T1, T2=self.T2)

        self.LL2 = LeadLag(u=self.LL1_y, T1=self.T3, T2=self.T4)

        self.WO = Washout(u=self.LL2_y, T=self.T6, K=self.KST5)  # WO_y == Vss

        self.VLIM = Limiter(u=self.WO_y, lower=self.LSMIN, upper=self.LSMAX, info='Vss limiter')

        self.Vss = Algeb(tex_name='V_{ss}', info='Voltage output before output limiter',
                         e_str='VLIM_zi * WO_y + VLIM_zu * LSMAX + VLIM_zl * LSMIN - Vss')

        self.OLIM = Limiter(u=self.v, lower=self.VCL, upper=self.VCU, info='output limiter')

        # TODO: allow ignoring VCU or VCL when zero

        self.vsout.e_str = 'OLIM_zi * Vss - vsout'


class IEEEST(IEEESTData, IEEESTModel):
    """
    IEEEST stabilizer model.

    Blocks are named "F1", "F2", "LL1", "LL2" and "WO" in sequence.
    Two limiters are named "VLIM" and "OLIM" in sequence.
    """

    def __init__(self, system, config):
        IEEESTData.__init__(self)
        IEEESTModel.__init__(self, system, config)
