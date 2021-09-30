from andes.core.param import NumParam
from andes.core.var import Algeb

from andes.core.service import ConstService, VarService
from andes.core.discrete import LessThan, Limiter, Switcher
from andes.core.block import LagAntiWindup, LeadLag, Washout, Lag, HVGate
from andes.core.block import Piecewise, Integrator

from andes.models.exciter.excbase import ExcBase, ExcBaseData, ExcVsum
from andes.models.exciter.saturation import ExcQuadSat


class ESST1AData(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)


        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u.',
                           )

        self.TB = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_B',
                           default=1,
                           unit='p.u.',
                           )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=1,
                           unit='p.u.',
                           )

        self.TB1 = NumParam(info='Lag time constant in lead-lag 1',
                            tex_name=r'T_{B1}',
                            default=1,
                            unit='p.u.',
                            )
        self.TC1 = NumParam(info='Lead time constant in lead-lag 1',
                            tex_name=r'T_{C1}',
                            default=1,
                            unit='p.u.',
                            )

        self.KA = NumParam(default=80,
                           info='Regulator gain',
                           tex_name='K_A',
                           )
        self.TA = NumParam(info='Lag time constant in regulator',
                           tex_name='T_A',
                           default=0.04,
                           unit='p.u.',
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
                           unit='p.u.',
                           non_negative=True,
                           non_zero=True,
                           )

        self.KC = NumParam(info='Rectifier loading factor proportional to commutating reactance',
                           tex_name='K_C',
                           default=0.1,
                           )

        self.UELcode = NumParam(info='Alternate UEL inputs, input code 1-3',
                                tex_name='UEL',
                                default=1,
                                )
        self.VOScode = NumParam(info='Alternate Stabilizer inputs, input code 1-2',
                                tex_name='VOS',
                                default=1,
                                )


class ESST1AModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        ExcVsum.__init__(self)

        self.ul = ConstService('9999')
        self.ll = ConstService('-9999')

        self.SWUEL = Switcher(u=self.UELcode, options=(0, 1, 2, 3), tex_name='SW_{UEL}', cache=True)
        self.SWVOS = Switcher(u=self.VOScode, options=(0, 1, 2), tex_name='SW_{VOS}', cache=True)

        # control block begin
        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )

        self.VOTHSG0 = ConstService(v_str='0', info='VOTHSG initial value.')
        self.VOTHSG = Algeb(tex_name='VOTHSG', info='VOTHSG',
                            v_str='VOTHSG0',
                            e_str='VOTHSG0 - VOTHSG',
                            )

        # input excitation voltages;
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='ue * (-LG_y + vref - WF_y + SWUEL_s1 * UEL + SWVOS_s1 * VOTHSG + Vs - vi)',
                        v_str='-v + vref',
                        diag_eps=True,
                        )
        self.HL = Limiter(u=self.HVG_y, lower=self.VIMIN, upper=self.VIMAX,
                           info='Hard limiter befor V_I')

        self.VI = Algeb(tex_name='V_I',
                        info='V_I',
                        v_str='1', # ?
                        e_str='ue * (HL_zi * vi + HL_zl * VIMIN + HL_zu * VIMAX - VI)',
                        diag_eps=True,
                        )

        self.HVG = HVGate(u1='SWUEL_s2 * UEL + (1 - SWUEL_s2) * ul',
                          u2=self.VI,
                          info='HVGate after V_I',
                          )

        self.LL = LeadLag(u=self.vi,
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

        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VAMAX,
                                lower=self.VAMIN,
                                info='V_A, Anti-windup lag',
                                )  # LA_y == VR

        # TODO: check values
        self.UEL0.v_str = '-999'
        self.OEL0.v_str = '999'

        self.vin = Algeb(info='input voltage',
                         tex_name='v_{in}',
                         v_str='VFE',
                         e_str='ue * (1-LVC_zl) * OEL + LVC_zl * HVG_y - vin',
                         diag_eps=True,
                         )

        self.ILR0 = ConstService(v_str='0', tex_name='I_{LR0}', info='ILR initial value')

        self.ILR = Algeb(info='exciter output current limit reference',
                         tex_name='I_{LR}}',
                         v_str='VFE',
                         e_str='ILR0 - ILR',
                         )

        self.zero = ConstService('0')
        self.HLI = Limiter(u=self.HVG_y, lower=self.zero,
                           upper=self.ul, no_lower=True,
                           info='Hard limiter for excitation current')

        self.HVG = HVGate(u1='SWUEL_s3 * UEL + (1 - SWUEL_s3) * ll',
                          u2='SWVOS_s2 * VOTHSG + LA_y - (1 - HLI_zl) * KLR * (ILR - XadIfd)',
                          info='HVGate for under excitation',
                          )

        self.LVC = Limiter(u=self.HVG_y, lower=self.OEL, upper=self.ul,
                           info='LVGate for over excitation', no_warn=True)

        self.LVG = Algeb(info='LVGate ouput',
                         tex_name='LVG_{y}',
                         v_str='VFE',
                         e_str='(1-LVC_zl) * OEL + LVC_zl * HVG_y - LVG',
                         )

        # TODO: lower and upper
        self.VOL = Limiter(u=self.LVG, lower=self.VRMIN, upper=self.VRMAX,
                           info='EFD limiter')

        self.WF = Washout(u=self.LVG,
                          T=self.TF,
                          K=self.KF,
                          info='V_F, Stablizing circuit feedback',
                          )

        self.LVG = Algeb(info='LVGate ouput',
                         tex_name='LVG_{y}',
                         v_str='VFE',
                         e_str='(1-LVC_zl) * OEL + LVC_zl * HVG_y - LVG',
                         )

        # TODO: should I use magnitude?
        self.efdu = Algeb(info='Output exciter voltage upper limit',
                           tex_name='LVG_{y}',
                           v_str='VFE',
                           e_str='Abs(vd + 1j*vq) * VRMAX - KC * XadIfd - efdu',
                           )

        self.efdl = Algeb(info='Output exciter voltage lower limit',
                          tex_name='LVG_{y}',
                          v_str='VFE',
                          e_str='Abs(vd + 1j*vq) * VRMIN - efdl',
                          )

        self.HLV = Limiter(u=self.LVG_y, lower=self.efdu, upper=self.efdl,
                           info='Hardlimiter for output excitation voltage')

        self.vout.e_str = 'HLV_zi * LVG + HLV_zu * efdu + HLV_zl * efdl - vout',



class ESST1A(ESST1AData, ESST1AModel):
    """
    Exciter ESST1A.
    """

    def __init__(self, system, config):
        ESST1AData.__init__(self)
        ESST1AModel.__init__(self, system, config)
