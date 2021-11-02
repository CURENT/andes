from andes.core.block import (HVGate, Integrator, Lag, LagAntiWindup, LeadLag,
                              LessThan, Washout,)
from andes.core.param import NumParam
from andes.core.service import (ConstService, FlagValue, PostInitService,
                                VarService,)
from andes.core.var import Algeb
from andes.models.exciter.excbase import ExcBase, ExcBaseData
from andes.models.exciter.saturation import ExcQuadSat


class ESDC2AData(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
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
        self.VRMAX = NumParam(info='Max. exc. limit (0-unlimited)',
                              tex_name='V_{RMAX}',
                              default=7.3,
                              unit='p.u.')
        self.VRMIN = NumParam(info='Min. excitation limit',
                              tex_name='V_{RMIN}',
                              default=-7.3,
                              unit='p.u.')
        self.KE = NumParam(info='Saturation feedback gain',
                           tex_name='K_E',
                           default=1,
                           unit='p.u.',
                           )
        self.TE = NumParam(info='Integrator time constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u.',
                           )
        self.KF = NumParam(default=0.1,
                           info='Feedback gain',
                           tex_name='K_F',
                           )
        self.TF1 = NumParam(info='Feedback washout time constant',
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
                           default=0.,
                           unit='p.u.',
                           )
        self.SE2 = NumParam(info='Value at second saturation point',
                            tex_name='S_{E2}',
                            default=0.,
                            unit='p.u.',
                            )


class ESDC2AModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        # Set VRMAX to 999 when VRMAX = 0
        self._zVRM = FlagValue(self.VRMAX, value=0,
                               tex_name='z_{VRMAX}',
                               )
        self.VRMAXc = ConstService(v_str='VRMAX + 999*(1-_zVRM)',
                                   info='Set VRMAX=999 when zero',
                                   )
        self.LG = Lag(u=self.v, T=self.TR, K=1,
                      info='Transducer delay',
                      )

        self.SAT = ExcQuadSat(self.E1, self.SE1, self.E2, self.SE2,
                              info='Field voltage saturation',
                              )

        # NOTE: Se0 below is not divided by `vf0` or `INT_y`
        # `Se0` is the variable `Vx` in Powerworld's diagram
        self.Se0 = ConstService(
            tex_name='S_{e0}',
            v_str='Indicator(vf0>SAT_A) * SAT_B*(SAT_A-vf0) ** 2',
        )

        self.vfe0 = ConstService(v_str='vf0*KE + Se0',
                                 tex_name='V_{FE0}',
                                 )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='v + vfe0 / KA',
                          e_str='vref0 - vref'
                          )
        self.vref0 = PostInitService(info='Const reference voltage',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        v_str='vfe0 / KA',
                        e_str='(vref - v - WF_y) - vi',
                        )

        self.LL = LeadLag(u=self.vi,
                          T1=self.TC,
                          T2=self.TB,
                          info='Lead-lag compensator',
                          zero_out=True,
                          )

        self.UEL = Algeb(info='Interface var for under exc. limiter',
                         tex_name='U_{EL}',
                         v_str='0',
                         e_str='0 - UEL'
                         )

        self.HG = HVGate(u1=self.UEL,
                         u2=self.LL_y,
                         info='HVGate for under excitation',
                         )

        self.VRU = VarService(v_str='VRMAXc * v',
                              tex_name='V_T V_{RMAX}',
                              )
        self.VRL = VarService(v_str='VRMIN * v',
                              tex_name='V_T V_{RMIN}',
                              )

        # TODO: WARNING: HVGate is temporarily skipped
        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VRU,
                                lower=self.VRL,
                                info='Anti-windup lag',
                                )  # LA_y == VR

        self.SL = LessThan(u=self.vout,
                           bound=self.SAT_A,
                           equal=False,
                           enable=True,
                           cache=False,
                           )

        self.Se = Algeb(tex_name=r"V_{out}*S_e(|V_{out}|)", info='saturation output',
                        v_str='Se0',
                        e_str='SL_z0 * (INT_y - SAT_A) ** 2 * SAT_B - Se',
                        )

        self.VFE = Algeb(info='Combined saturation feedback',
                         tex_name='V_{FE}',
                         unit='p.u.',
                         v_str='vfe0',
                         e_str='(INT_y*KE + Se) - VFE'
                         )

        self.INT = Integrator(u='ue * (LA_y - VFE)',
                              T=self.TE,
                              K=1,
                              y0=self.vf0,
                              info='Integrator',
                              )

        self.WF = Washout(u=self.INT_y,
                          T=self.TF1,
                          K=self.KF,
                          info='Feedback to input'
                          )

        self.vout.e_str = 'INT_y - vout'


class ESDC2A(ESDC2AData, ESDC2AModel):
    """
    ESDC2A model.

    This model is implemented as described in the PSS/E manual,
    except that the HVGate is not in use.
    Due to the HVGate and saturation function, the results
    are close to but different from TSAT.
    """

    def __init__(self, system, config):
        ESDC2AData.__init__(self)
        ESDC2AModel.__init__(self, system, config)
