from andes.core.param import NumParam
from andes.core.var import Algeb

from andes.core.service import ConstService

from andes.core.block import LagAntiWindup, LeadLag, Washout, Lag, HVGate
from andes.core.block import LVGate, AntiWindup, IntegratorAntiWindup

from andes.models.exciter.excbase import ExcBase, ExcBaseData
# from andes.models.exciter.saturation import ExcQuadSat


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

        # control block begin
        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )

        # TODO: add self.vref

        # TODO: modify v_str for VF
        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='-LG_y + vref + VF - vi',
                        v_str='-v + vref',
                        )

        # TODO: check info
        self.LL = LeadLag(u=self.vi, T1=self.TC, T2=self.TB,
                          info='Regulator',
                          zero_out=True,
                          )  # LL_y == VA ???

        # TODO: replace upper and lower
        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VRU,
                                lower=self.VRL,
                                info='Anti-windup lag',
                                )  # LA_y == VR

        # TODO: check if eqn is correct
        self.UEL = Algeb(info='Interface var for under exc. limiter',
                         tex_name='U_{EL}',
                         v_str='0',
                         e_str='0 - UEL'
                         )

        # TODO: check info
        self.HG = HVGate(u1=self.UEL,
                         u2=self.LA_y,
                         info='HVGate for under excitation',
                         )

        # TODO: check if eqn is correct, seems wrong
        self.OEL = Algeb(info='Interface var for under exc. limiter',
                         tex_name='O_{EL}',
                         v_str='0',
                         e_str='0 - OEL'
                         )
        # TODO: check info
        self.LG = LVGate(u1=self.HG_y,
                         u2=self.OEL,
                         info='LVGate for under excitation',
                         )

        self.AW = AntiWindup(u=self.LG_y,
                             lower=self.VRMIN,
                             upper=self.VRMAX,
                             tex_name=r'lim_{LG}',
                             )

        # TODO: check y0
        # TODO: add self.vf0
        self.large = ConstService('999')
        self.IAW = IntegratorAntiWindup(u='ue * (AW_y - VFE)',
                                        T=self.TE,
                                        K=1,
                                        y0=self.vf0,
                                        lower=self.zero,
                                        upper=self.large,
                                        info='Integrator Anti-Windup',
                                        )

        # TODO: add VFE

        self.VF = Washout(u=self.INT_y,
                          T=self.TF,
                          K=self.KF,
                          info='Feedback to input'
                          )

        # TODO: add saturation VX

        # TODO: add FEX


class ESAC1A(ESAC1AData, ESAC1AModel):
    """
    Static exciter type 3A model
    """

    def __init__(self, system, config):
        ESAC1AData.__init__(self)
        ESAC1AModel.__init__(self, system, config)
