"""
ESST1A model.
"""

from andes.core.block import (GainLimiter, HVGate, Lag, LagAntiWindup, LeadLag,
                              LVGate, Washout,)
from andes.core.discrete import Switcher
from andes.core.param import NumParam
from andes.core.service import ConstService, PostInitService, VarService
from andes.core.var import Algeb, ExtAlgeb
from andes.models.exciter.excbase import ExcBase, ExcBaseData, ExcVsum


class ESST1AData(ExcBaseData):
    """
    ESST1A data.
    """

    def __init__(self):
        ExcBaseData.__init__(self)
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           )

        self.VIMAX = NumParam(default=0.8,
                              info='Max. input voltage',
                              tex_name=r'V_{IMAX}',
                              )
        self.VIMIN = NumParam(default=-0.1,
                              info='Min. input voltage',
                              tex_name=r'V_{IMIN}',
                              )
        self.TB = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_B',
                           default=1,
                           )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=1,
                           )

        self.TB1 = NumParam(info='Lag time constant in lead-lag 1',
                            tex_name=r'T_{B1}',
                            default=1,
                            )
        self.TC1 = NumParam(info='Lead time constant in lead-lag 1',
                            tex_name=r'T_{C1}',
                            default=1,
                            )

        self.VAMAX = NumParam(info='V_A upper limit',
                              tex_name=r'V_{AMAX}',
                              default=999,
                              unit='p.u.')
        self.VAMIN = NumParam(info='V_A lower limit',
                              tex_name=r'V_{AMIN}',
                              default=-999,
                              unit='p.u.')

        self.KA = NumParam(default=80,
                           info='Regulator gain',
                           tex_name='K_A',
                           )
        self.TA = NumParam(info='Lag time constant in regulator',
                           tex_name='T_A',
                           default=0.04,
                           )

        self.ILR = NumParam(default=1,
                            info='Exciter output current limite reference',
                            tex_name=r'I_{LR}',
                            )
        self.KLR = NumParam(default=1,
                            info='Exciter output current limiter gain',
                            tex_name=r'K_{LR}',
                            )

        self.VRMAX = NumParam(info='Maximum voltage regulator output limit',
                              tex_name=r'V_{RMAX}',
                              default=7.3,
                              unit='p.u.',)
        self.VRMIN = NumParam(info='Minimum voltage regulator output limit',
                              tex_name=r'V_{RMIN}',
                              default=-7.3,
                              unit='p.u.',)

        self.KF = NumParam(default=0.1,
                           info='Feedback gain',
                           tex_name='K_F',
                           )
        self.TF = NumParam(info='Feedback washout time constant',
                           tex_name='T_{F}',
                           default=1,
                           )

        self.KC = NumParam(info='Rectifier loading factor proportional to commutating reactance',
                           tex_name='K_C',
                           default=0.1,
                           )

        self.UELc = NumParam(info='Alternate UEL inputs, input code 1-3',
                             tex_name='UEL_c',
                             default=1,
                             )
        self.VOSc = NumParam(info='Alternate Stabilizer inputs, input code 1-2',
                             tex_name='VOS_c',
                             default=1,
                             )


class ESST1AModel(ExcBase, ExcVsum):
    """
    Implementation of the ESST1A model.
    """

    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        self.flags.nr_iter = True

        ExcVsum.__init__(self)
        self.UEL0.v_str = '-999'
        self.OEL0.v_str = '999'

        self.ulim = ConstService('9999')
        self.llim = ConstService('-9999')

        self.SWUEL = Switcher(u=self.UELc, options=[0, 1, 2, 3], tex_name='SW_{UEL}', cache=True)
        self.SWVOS = Switcher(u=self.VOSc, options=[0, 1, 2], tex_name='SW_{VOS}', cache=True)

        # control block begin
        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )
        self.SG0 = ConstService(v_str='0', info='SG initial value.')
        self.SG = Algeb(tex_name='SG', info='SG',
                        v_str='SG0',
                        e_str='SG0 - SG',
                        )

        self.zero = ConstService('0')
        self.LR = GainLimiter(u='XadIfd - ILR',
                              K=self.KLR, R=1,
                              upper=self.ulim, lower=self.zero,
                              no_upper=True,
                              info='Exciter output current gain limiter',
                              )

        self.VA0 = PostInitService(tex_name='V_{A0}',
                                   v_str='vf0 - SWVOS_s2 * SG + LR_y',
                                   info='VA (LA_y) initial value')

        self.vref.v_str = 'ue * (v + (vf0 - SWVOS_s2 * SG + LR_y) / KA - SWVOS_s1 * SG - SWUEL_s1 * UEL)'
        self.vref.v_iter = 'ue * (v + (vf0 - SWVOS_s2 * SG + LR_y) / KA - SWVOS_s1 * SG - SWUEL_s1 * UEL)'

        self.vref0 = PostInitService(info='Initial reference voltage input',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='ue * (-LG_y + vref - WF_y + SWUEL_s1 * UEL + SWVOS_s1 * SG + Vs) - vi',
                        v_iter='ue * (-LG_y + vref - WF_y + SWUEL_s1 * UEL + SWVOS_s1 * SG + Vs)',
                        v_str='ue * (-LG_y + vref - WF_y + SWUEL_s1 * UEL + SWVOS_s1 * SG + Vs)',
                        )

        self.vil = GainLimiter(u=self.vi,
                               K=1, R=1,
                               upper=self.VIMAX, lower=self.VIMIN,
                               info='Exciter voltage input limiter',
                               )

        self.UEL2 = Algeb(tex_name='UEL_2',
                          info='UEL_2 as HVG1 u1',
                          v_str='ue * (SWUEL_s2 * UEL + (1 - SWUEL_s2) * llim)',
                          e_str='ue * (SWUEL_s2 * UEL + (1 - SWUEL_s2) * llim) - UEL2',
                          )
        self.HVG1 = HVGate(u1=self.UEL2,
                           u2=self.vil_y,
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

        self.vas = Algeb(tex_name=r'V_{As}',
                         info='V_A after subtraction, as HVG u2',
                         v_str='ue * (SWVOS_s2 * SG + LA_y - LR_y)',
                         v_iter='ue * (SWVOS_s2 * SG + LA_y - LR_y)',
                         e_str='ue * (SWVOS_s2 * SG + LA_y - LR_y) - vas',
                         )

        self.UEL3 = Algeb(tex_name='UEL_3',
                          info='UEL_3 as HVG u1',
                          v_str='ue * (SWUEL_s3 * UEL + (1 - SWUEL_s3) * llim)',
                          e_str='ue * (SWUEL_s3 * UEL + (1 - SWUEL_s3) * llim) - UEL3',
                          )
        self.HVG = HVGate(u1=self.UEL3,
                          u2=self.vas,
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
                               tex_name=r'efd_{u}',
                               v_str='Abs(vd + 1j*vq) * VRMAX - KC * XadIfd',
                               sequential=False,
                               )
        self.efdl = VarService(info='Output exciter voltage lower bound',
                               tex_name=r'efd_{l}',
                               v_str='Abs(vd + 1j*vq) * VRMIN',
                               sequential=False,
                               )

        self.vol = GainLimiter(u=self.LVG_y,
                               K=1, R=1,
                               upper=self.efdu,
                               lower=self.efdl,
                               info='Exciter output limiter',
                               )

        self.WF = Washout(u=self.LVG_y,
                          T=self.TF,
                          K=self.KF,
                          info='V_F, Stablizing circuit feedback',
                          )

        self.vout.e_str = 'ue * vol_y  - vout'


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
