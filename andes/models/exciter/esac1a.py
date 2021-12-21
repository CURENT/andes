from andes.core.block import (HVGate, Lag, LagAntiWindup, LeadLag, LVGate,
                              Piecewise, Washout,)
from andes.core.param import NumParam
from andes.core.service import PostInitService, ConstService
from andes.core.var import Algeb
from andes.models.exciter.excbase import (ExcACSat, ExcBase, ExcBaseData,
                                          ExcVsum,)


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
                           non_negative=True,
                           )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=1,
                           unit='p.u.',
                           non_negative=True,
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
                           unit='p.u.',
                           non_negative=True,
                           )
        self.VRMAX = NumParam(info='Max. exc. limit (0-unlimited)',
                              tex_name=r'V_{RMAX}',
                              default=7.3,
                              unit='p.u.')
        self.VRMIN = NumParam(info='Min. excitation limit',
                              tex_name=r'V_{RMIN}',
                              default=-7.3,
                              unit='p.u.')
        self.TE = NumParam(info='Integrator time constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u.',
                           non_negative=True,
                           )
        self.E1 = NumParam(info='First saturation point',
                           tex_name='E_1',
                           default=0.,
                           unit='p.u.',
                           )
        self.SE1 = NumParam(info='Value at first saturation point',
                            tex_name=r'S_{E1}',
                            default=0.,
                            unit='p.u.',
                            )
        self.E2 = NumParam(info='Second saturation point',
                           tex_name='E_2',
                           default=1.,
                           unit='p.u.',
                           )
        self.SE2 = NumParam(info='Value at second saturation point',
                            tex_name=r'S_{E2}',
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


class ESAC1AModel(ExcBase, ExcVsum, ExcACSat):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        ExcVsum.__init__(self)

        self.UEL0.v_str = '-999'
        self.OEL0.v_str = '999'

        self.flags.nr_iter = True

        # NOTE: e_str `KC*XadIfd / INT_y - IN` causes numerical inaccuracies
        self.IN = Algeb(tex_name='I_N',
                        info='Input to FEX',
                        v_str='1',
                        v_iter='KC * XadIfd - INT_y * IN',
                        e_str='ue * (KC * XadIfd - INT_y * IN)',
                        diag_eps=True,
                        )

        self.FEX = Piecewise(u=self.IN,
                             points=(0, 0.433, 0.75, 1),
                             funs=('1', '1 - 0.577*IN', 'sqrt(0.75 - IN ** 2)', '1.732*(1 - IN)', 0),
                             info='Piecewise function FEX',
                             )
        self.FEX.y.v_str = '1'
        self.FEX.y.v_iter = self.FEX.y.e_str

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

        self.VAMAXu = ConstService('VAMAX * ue + (1-ue) * 999')
        self.VAMINu = ConstService('VAMIN * ue + (1-ue) * -999')

        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VAMAXu,
                                lower=self.VAMINu,
                                info='V_A, Anti-windup lag',
                                )  # LA_y == VA

        self.HVG = HVGate(u1=self.UEL,
                          u2=self.LA_y,
                          info='HVGate for under excitation',
                          )

        self.LVG = LVGate(u1=self.HVG_y,
                          u2=self.OEL,
                          info='HVGate for under excitation',
                          )

        self.INTin = 'ue * (LVG_y - VFE)'

        ExcACSat.__init__(self)

        self.vref.v_str = 'v + VFE / KA'

        self.vref0 = PostInitService(info='Initial reference voltage input',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        self.WF = Washout(u=self.VFE,
                          T=self.TF,
                          K=self.KF,
                          info='Stablizing circuit feedback',
                          )

        self.vout.e_str = 'ue * FEX_y * INT_y - vout'


class ESAC1A(ESAC1AData, ESAC1AModel):
    """
    Exciter ESAC1A.
    """

    def __init__(self, system, config):
        ESAC1AData.__init__(self)
        ESAC1AModel.__init__(self, system, config)
