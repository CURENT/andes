from collections import OrderedDict

from andes.core.param import NumParam
from andes.core.var import Algeb, ExtAlgeb

from andes.core.service import PostInitService, ConstService, VarService
from andes.core.discrete import Switcher, Limiter
from andes.core.block import LagAntiWindup, Lag, HVGate, LVGate
from andes.core.block import Piecewise, PIDTrackAW, LeadLag, Washout

from andes.models.exciter.excbase import ExcBase, ExcBaseData, ExcVsum, ExcACSat
from andes.core.common import dummify

class ESST1AData(ExcBaseData):
    """
    ESST1A data.
    """

    def __init__(self):
        ExcBaseData.__init__(self)
        self.kP = NumParam(info='PID proportional coeff.',
                           tex_name='k_P',
                           default=10,
                           vrange=(10, 500),
                           )
        self.kI = NumParam(info='PID integrative coeff.',
                           tex_name='k_I',
                           default=10,
                           vrange=(10, 500),
                           )
        self.kD = NumParam(info='PID direvative coeff.',
                           tex_name='k_D',
                           default=10,
                           vrange=(10, 500),
                           )
        self.Td = NumParam(info='PID direvative time constant.',
                           tex_name='T_d',
                           default=0.2,
                           vrange=(0, 0.5),
                           )

        self.VPMAX = NumParam(info='PID maximum limit',
                              tex_name='V_{PMAX}',
                              default=999,
                              unit='p.u.')
        self.VPMIN = NumParam(info='PID minimum limit',
                              tex_name='V_{PMIN}',
                              default=-999,
                              unit='p.u.')


        # TODO: check default value for VFEMAX
        self.VFEMAX = NumParam(info='Maximum VFE',
                               tex_name=r'V_{FEMAX}',
                               default=999,
                               unit='p.u.')

        # TODO: check default value for VEMIN
        self.VEMIN = NumParam(info='Minimum excitation output',
                              tex_name=r'V_{EMIN}',
                              default=-999,
                              unit='p.u.')

        self.TE = NumParam(info='Exciter integrator time constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u.',
                           )

        self.E1 = NumParam(info='First saturation point',
                           tex_name='E_1',
                           default=0.,
                           unit='p.u.',
                           )
        self.SE1 = NumParam(info='Value at first saturation point',
                            tex_name='S_{E1}',
                            default=0.,
                            unit='p.u.',
                            )
        self.E2 = NumParam(info='Second saturation point',
                           tex_name='E_2',
                           default=1.,
                           unit='p.u.',
                           )
        self.SE2 = NumParam(info='Value at second saturation point',
                            tex_name='S_{E2}',
                            default=1.,
                            unit='p.u.',
                            )

        self.KE = NumParam(info='Gain added to saturation',
                           tex_name='K_E',
                           default=1,
                           unit='p.u.',
                           )
        self.KD = NumParam(default=0,
                           info='Ifd feedback gain',
                           tex_name='K_D',
                           vrange=(0, 1),
                           )

        # -- cut line
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           )

        self.VIMAX = NumParam(default=0.8,
                              info='Max. input voltage',
                              tex_name='V_{IMAX}',
                              )
        self.VIMIN = NumParam(default=-0.1,
                              info='Min. input voltage',
                              tex_name='V_{IMIN}',
                              )
        self.TB = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_B',
                           default=1,
                           )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=1,
                           )

        self.VAMAX = NumParam(info='V_A upper limit',
                              tex_name='V_{AMAX}',
                              default=999,
                              unit='p.u.')
        self.VAMIN = NumParam(info='V_A lower limit',
                              tex_name='V_{AMIN}',
                              default=-999,
                              unit='p.u.')
        self.TB1 = NumParam(info='Lag time constant in lead-lag 1',
                            tex_name=r'T_{B1}',
                            default=1,
                            )
        self.TC1 = NumParam(info='Lead time constant in lead-lag 1',
                            tex_name=r'T_{C1}',
                            default=1,
                            )

        self.KA = NumParam(default=80,
                           info='Regulator gain',
                           tex_name='K_A',
                           )
        self.TA = NumParam(info='Lag time constant in regulator',
                           tex_name='T_A',
                           default=0.04,
                           )

        self.KLR = NumParam(default=1,
                            info='Exciter output current limiter gain',
                            tex_name=r'K_LR',
                            )

        self.VRMAX = NumParam(info='Maximum excitation limit',
                              tex_name='V_{RMAX}',
                              default=7.3,
                              unit='p.u.',)
        self.VRMIN = NumParam(info='Minimum excitation limit',
                              tex_name='V_{RMIN}',
                              default=1,
                              unit='p.u.',)

        self.KF = NumParam(default=0.1,
                           info='Feedback gain',
                           tex_name='K_F',
                           )
        self.TF = NumParam(info='Feedback washout time constant',
                           tex_name='T_{F1}',
                           default=1,
                           )

        self.KC = NumParam(info='Rectifier loading factor proportional to commutating reactance',
                           tex_name='K_C',
                           default=0.1,
                           )

        self.UELc = NumParam(info='Alternate UEL inputs, input code 1-3',
                             tex_name='UEL',
                             default=1,
                             )
        self.VOSc = NumParam(info='Alternate Stabilizer inputs, input code 1-2',
                             tex_name='VOS',
                             default=1,
                             )

class ESST1AModel(ExcBase, ExcVsum, ExcACSat):
    """
    Implementation of the ESST1A model.
    """

    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        self.flags.nr_iter = True
        
        self.ul = ConstService('9999')
        self.ll = ConstService('-9999')

        self.SWUEL = Switcher(u=self.UELc, options=[0, 1, 2, 3], tex_name='SW_{UEL}', cache=True)
        self.SWVOS = Switcher(u=self.VOSc, options=[0, 1, 2], tex_name='SW_{VOS}', cache=True)

        # control block begin
        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )
        self.VOTHSG0 = ConstService(v_str='0', info='VOTHSG initial value.')
        self.VOTHSG = Algeb(tex_name='VOTHSG', info='VOTHSG',
                            v_str='VOTHSG0',
                            e_str='VOTHSG0 - VOTHSG',
                            )
        # TODO: check v_str
        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name='V_{ref0}',
                                #   v_str='v + vfe0 / KA',
                                  v_str='1',
                                  )

        ExcVsum.__init__(self)
        self.UEL0.v_str = '-999'
        self.OEL0.v_str = '999'

        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='ue * (-LG_y + vref - WF_y + SWUEL_s1 * UEL + SWVOS_s1 * VOTHSG + Vs - vi)',
                        v_str='-v + vref',
                        diag_eps=True,
                        )

        self.vil = Limiter(u=self.vi, lower=self.VIMIN, upper=self.VIMAX,
                           info='Hard limiter before V_I')

        # TODO: check v_str
        self.VI = Algeb(tex_name='V_I',
                        info='V_I',
                        v_str='1',
                        e_str='ue * (vil_zi * vi + vil_zl * VIMIN + vil_zu * VIMAX - VI)',
                        diag_eps=True,
                        )

        self.UEL2 = Algeb(tex_name='UEL_2',
                          info='UEL_2 as HVG1 u1',
                          v_str='SWUEL_s2 * UEL + (1 - SWUEL_s2) * ll',
                          e_str='ue * (SWUEL_s2 * UEL + (1 - SWUEL_s2) * ll - UEL2)',
                          )
        self.HVG1 = HVGate(u1=self.UEL2,
                           u2=self.VI,
                           info='HVGate after V_I',
                           )

        self.LL = LeadLag(u=self.HVG1_y,
                          T1=self.TC,
                          T2=self.TB,
                          info='Lead-lag compensator',
                          zero_out=True,
                          )

        self.LL1 = LeadLag(u=self.LL_y,
                           T1=self.TC1,
                           T2=self.TB1,
                           info='Lead-lag compensator 1',
                           zero_out=True,
                           )

        self.LA = LagAntiWindup(u=self.LL1_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VAMAX,
                                lower=self.VAMIN,
                                info='V_A, Anti-windup lag',
                                )  # LA_y is VA

        self.ILR0 = ConstService(v_str='0', tex_name='I_{LR0}', info='ILR initial value')
        self.ILR = Algeb(info='exciter output current limit reference',
                         tex_name='I_{LR}}',
                         v_str='ILR0',
                         e_str='ILR0 - ILR',
                         )

        self.zero = ConstService('0')
        self.HLI = Limiter(u='XadIfd - ILR',
                           lower=self.zero, upper=self.ul, no_upper=True,
                           info='Hard limiter for excitation current')

        self.UEL3 = Algeb(tex_name='UEL_3',
                          info='UEL_3 as HVG u1',
                          v_str='SWUEL_s3 * UEL + (1 - SWUEL_s3) * ll',
                          e_str='ue * (SWUEL_s3 * UEL + (1 - SWUEL_s3) * ll - UEL3)',
                          )
        self.HVGu2 = Algeb(tex_name=r'HVG_{u2}',
                           info='HVG u2',
                           v_str='SWVOS_s2 * VOTHSG + LA_y - (1 - HLI_zl) * KLR * (ILR - XadIfd)',
                           e_str='ue * (SWVOS_s2 * VOTHSG + LA_y - (1 - HLI_zl) * KLR * (ILR - XadIfd) - HVGu2)',
                           )
        self.HVG = HVGate(u1=self.UEL3,
                          u2=self.HVGu2,
                          info='HVGate for under excitation',
                          )

        self.LVG = LVGate(u1=self.HVG_y,
                          u2=self.OEL,
                          info='HVGate for over excitation',
                          )

        # vd, vq, Id, Iq from SynGen
        self.vd = ExtAlgeb(src='vd',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'V_d',
                           info='d-axis machine voltage',
                           )
        self.vq = ExtAlgeb(src='vq',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'V_q',
                           info='q-axis machine voltage',
                           )

        self.efdu = VarService(info='Output exciter voltage upper bound',
                               tex_name='efd_{u}',
                               v_str='Abs(vd + 1j*vq) * VRMAX - KC * XadIfd',
                               )
        self.efdl = VarService(info='Output exciter voltage lower bound',
                               tex_name='efd_{l}',
                               v_str='Abs(vd + 1j*vq) * VRMIN'
                               )

        self.vol = Limiter(u=self.LVG_y, info='vout limiter',
                           lower=self.efdu, upper=self.efdl,
                           )

        self.WF = Washout(u=self.LVG_y,
                          T=self.TF,
                          K=self.KF,
                          info='V_F, Stablizing circuit feedback',
                          )

        self.vout.e_str = 'ue * (vol_zi * LVG_y + vol_zl * efdl + vol_zu * efdu  - vout)'


class ESST1A(ESST1AData, ESST1AModel):
    """
    Exciter ESST1A model.
    Reference:
    [1] PowerWorld, Exciter ESST1A, [Online],
    [2] NEPLAN, Exciters Models, [Online],
    Available:
    https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Exciter%20ESST1A.htm
    https://www.neplan.ch/wp-content/uploads/2015/08/Nep_EXCITERS1.pdf
    """
    def __init__(self, system, config):
        ESST1AData.__init__(self)
        ESST1AModel.__init__(self, system, config)
