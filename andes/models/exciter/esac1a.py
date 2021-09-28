from andes.core.param import NumParam
from andes.core.var import Algeb

from andes.core.service import ConstService
from andes.core.discrete import LessThan, HardLimiter
from andes.core.block import LagAntiWindup, LeadLag, Washout, Lag, HVGate
from andes.core.block import LVGate, IntegratorAntiWindup
from andes.core.block import Piecewise

from andes.models.exciter.excbase import ExcBase, ExcBaseData, ExcVsum
from andes.models.exciter.saturation import ExcQuadSat


class ESAC1AData(ExcBaseData):
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
        self.VAMAX = NumParam(info='V_A upper limit',
                              tex_name='V_{AMAX}',
                              default=999,
                              unit='p.u.')
        self.VAMIN = NumParam(info='V_A lower limit',
                              tex_name='V_{AMIN}',
                              default=-999,
                              unit='p.u.')
        self.KA = NumParam(default=80,
                           info='Regulator gain',
                           tex_name='K_A',
                           )
        self.TA = NumParam(info='Lag time constant in regulator',
                           tex_name='T_A',
                           default=0.04,
                           unit='p.u.',
                           )
        self.VRMAX = NumParam(info='Max. exc. limit (0-unlimited)',
                              tex_name='V_{RMAX}',
                              default=7.3,
                              unit='p.u.')
        self.VRMIN = NumParam(info='Min. excitation limit',
                              tex_name='V_{RMIN}',
                              default=-7.3,
                              unit='p.u.')
        self.TE = NumParam(info='Integrator time constant',
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
        self.KC = NumParam(info='Rectifier loading factor proportional to commutating reactance',
                           tex_name='K_C',
                           default=0.1,
                           )
        self.KD = NumParam(info='Ifd feedback gain',
                           tex_name='K_D',
                           default=0,
                           )
        self.KE = NumParam(info='Gain added to saturation',
                           tex_name='K_E',
                           default=1,
                           )
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

        self.Switch = NumParam(info='Switch that PSS/E did not implement',
                               tex_name='S_w',
                               default=0,
                               unit='bool',
                               )


class ESAC1AModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        ExcVsum.__init__(self)

        # TODO: check values
        self.UEL0.v_str = '-999'
        self.OEL0.v_str = '999'

        # Assume FEX is in (0, 0.433) at initial
        self.VE0 = ConstService(info='Initial VE',
                                tex_name=r'V_{E0}',
                                v_str='vf0 + 0.577 * KC * XadIfd')

        self.VR0 = ConstService(info='Initial VR',
                                tex_name=r'V_{R0}',
                                v_str='VFE0 + VE0')

        self.vb0 = ConstService(info='Initial vb',
                                tex_name='V_{b0}',
                                v_str='VR0 / KA')

        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name=r'V_{ref0}',
                                  v_str='v + vb0')

        # control block begin
        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )

        # input excitation voltages;
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='ue * (-LG_y + vref + UEL + OEL + Vs - vi)',
                        v_str='-v + vref',
                        diag_eps=True,
                        )

        self.LL = LeadLag(u=self.vi, T1=self.TC, T2=self.TB,
                          info='V_A, Lead-lag compensator',
                          zero_out=True,
                          )  # LL_y == VA

        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VAMAX,
                                lower=self.VAMIN,
                                info='V_R, Anti-windup lag',
                                )  # LA_y == VR

        # self.HVG = HVGate(u1=self.UEL,
        #                   u2=self.LA_y,
        #                   info='HVGate for under excitation',
        #                   )

        # LVGate results in initialization bug
        # self.LVG = LVGate(u1=self.HVG_y,
        #                   u2=self.OEL,
        #                   info='V_R, LVGate for under excitation',
        #                   )
        # self.LVG = LessThan(u=self.HVG_y, bound=self.OEL, equal=False, enable=True, cache=False)

        # self.HLR = HardLimiter(u=self.LA_y,
        #                        lower=self.VRMIN,
        #                        upper=self.VRMAX,
        #                        info='V_R input limiter',
        #                        )

        # self.VR = Algeb(info='V_R after limiter',
        #                 tex_name='V_{R}',
        #                 v_str='HLR_zi*LA_y + HLR_zl*VRMIN + HLR_zu*VRMAX',
        #                 e_str='HLR_zi*LA_y + HLR_zl*VRMIN + HLR_zu*VRMAX - VR'
        #                 )

        self.zero = ConstService('0')
        self.large = ConstService('999')

        # self.VEin = Algeb(info='before INT',
        #                   tex_name='V_{Ein}',
        #                   v_str='vf0 + 0.577 * KC * vf0',
        #                   e_str='u * (VR - VFE - VEin)'
        #                   )
        self.INT = IntegratorAntiWindup(u=self.LA_y,
                                        T=self.TE,
                                        K=1,
                                        y0='vf0 + 0.577 * KC * XadIfd',
                                        lower=self.zero,
                                        upper=self.large,
                                        info='V_E, Integrator Anti-Windup',
                                        )

        self.SAT = ExcQuadSat(self.E1, self.SE1, self.E2, self.SE2,
                              info='Field voltage saturation',
                              )

        self.SL = LessThan(u=self.INT_y, bound=self.SAT_A, equal=False, enable=True, cache=False)

        # SL_z0 indicates saturation
        self.Se = Algeb(tex_name=r"V_{E}*S_e(|V_{E}|)", info='saturation output',
                        v_str='Indicator(INT_y > SAT_A) * SAT_B * (INT_y - SAT_A) ** 2',
                        e_str='SL_z0 * (INT_y - SAT_A) ** 2 * SAT_B - Se',
                        )

        self.VFE0 = ConstService(info='Initial VFE', tex_name=r'V_{FE0}',
                                 v_str='INT_y * KE + Se + XadIfd * KD',
                                 )

        self.IN = Algeb(tex_name='I_N',
                        info='Input to FEX',
                        v_str='safe_div(KC * XadIfd, INT_y)',
                        e_str='ue * (KC * XadIfd + IN * INT_y)',
                        diag_eps=True,
                        )

        # TODO: check funs
        self.FEX = Piecewise(u=self.IN,
                             points=(0, 0.433, 0.75, 1),
                             funs=('1', '1 - 0.577*IN', 'sqrt(0.75 - IN ** 2)', '1.732*(1 - IN)', 0),
                             info='Piecewise function FEX',
                             )

        # TODO: check if e_str needs modification
        self.VFE = Algeb(info='Combined saturation feedback',
                         tex_name=r'V_{FE}',
                         unit='p.u.',
                         v_str='INT_y * KE + Se + KD * XadIfd',
                         e_str='(INT_y * KE + Se + KD * XadIfd) - VFE'
                         )

        self.WO = Washout(u=self.VFE,
                          T=self.TF,
                          K=self.KF,
                          info='V_F, Washout, Feedback to input'
                          )

        self.vout.e_str = 'FEX_y * INT_y - vout'


class ESAC1A(ESAC1AData, ESAC1AModel):
    """
    Exciter ESAC1A.
    """

    def __init__(self, system, config):
        ESAC1AData.__init__(self)
        ESAC1AModel.__init__(self, system, config)
