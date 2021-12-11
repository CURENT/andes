"""
Module for EXDC2 exciter.
"""

from andes.core.block import Lag, LagAntiWindup, LeadLag, LessThan, Washout
from andes.core.param import NumParam
from andes.core.service import ConstService, PostInitService
from andes.core.var import Algeb, State
from andes.models.exciter.excbase import ExcBase, ExcBaseData
from andes.models.exciter.saturation import ExcQuadSat


class EXDC2Data(ExcBaseData):
    def __init__(self):
        super().__init__()
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u.',
                           )
        self.TA = NumParam(info='Lag time constant in anti-windup lag',
                           tex_name='T_A',
                           default=0.04,
                           unit='p.u.',
                           )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=1,
                           unit='p.u.',
                           )
        self.TB = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_B',
                           default=1,
                           unit='p.u.',
                           )
        self.TE = NumParam(info='Exciter integrator time constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u.',
                           )
        self.TF1 = NumParam(info='Feedback washout time constant',
                            tex_name='T_{F1}',
                            default=1,
                            unit='p.u.',
                            non_zero=True
                            )
        self.KF1 = NumParam(info='Feedback washout gain',
                            tex_name='K_{F1}',
                            default=0.03,
                            unit='p.u.',
                            )
        self.KA = NumParam(info='Gain in anti-windup lag TF',
                           tex_name='K_A',
                           default=40,
                           unit='p.u.',
                           )
        self.KE = NumParam(info='Gain added to saturation',
                           tex_name='K_E',
                           default=1,
                           unit='p.u.',
                           )
        self.VRMAX = NumParam(info='Maximum excitation limit',
                              tex_name='V_{RMAX}',
                              default=7.3,
                              unit='p.u.')
        self.VRMIN = NumParam(info='Minimum excitation limit',
                              tex_name='V_{RMIN}',
                              default=-7.3,
                              unit='p.u.')
        self.E1 = NumParam(info='First saturation point',
                           tex_name='E_1',
                           default=0.0,
                           unit='p.u.',
                           )
        self.SE1 = NumParam(info='Value at first saturation point',
                            tex_name='S_{E1}',
                            default=0.0,
                            unit='p.u.',
                            )
        # the defaults for `E2` and `SE2` has been changed to 1
        # so that when E1=SE1=0, E2=SE2=1, saturation is disabled.
        # This will be patched later to allow all to be 0

        self.E2 = NumParam(info='Second saturation point',
                           tex_name='E_2',
                           default=1.0,
                           unit='p.u.',
                           )
        self.SE2 = NumParam(info='Value at second saturation point',
                            tex_name='S_{E2}',
                            default=1.0,
                            unit='p.u.',
                            )


class EXDC2Model(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        self.SAT = ExcQuadSat(self.E1, self.SE1, self.E2, self.SE2,
                              info='Field voltage saturation',
                              )

        # calculate `Se0` ahead of time in order to calculate `vr0`
        # The term `1-ug` is to prevent division by zero when generator is off
        self.Se0 = ConstService(info='Initial saturation output',
                                tex_name='S_{e0}',
                                v_str='Indicator(vf0>SAT_A) * SAT_B * (SAT_A - vf0) ** 2 / (vf0 + 1 - ug)',
                                )
        self.vr0 = ConstService(info='Initial vr',
                                tex_name='V_{r0}',
                                v_str='(KE + Se0) * vf0')
        self.vb0 = ConstService(info='Initial vb',
                                tex_name='V_{b0}',
                                v_str='vr0 / KA')

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='v + vb0',
                          e_str='vref0 - vref'
                          )
        self.vref0 = PostInitService(info='Constant v ref',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        self.SL = LessThan(u=self.vout,
                           bound=self.SAT_A,
                           equal=False,
                           enable=True,
                           cache=False,
                           )

        self.Se = Algeb(tex_name=r"S_e(|V_{out}|)", info='saturation output',
                        v_str='Se0',
                        e_str='SL_z0 * (vp - SAT_A) ** 2 * SAT_B - Se * vp',
                        diag_eps=True,
                        )

        self.vp = State(info='Voltage after saturation feedback, before speed term',
                        tex_name='V_p',
                        unit='p.u.',
                        v_str='vf0',
                        e_str='ue * (LA_y - KE*vp - Se*vp)',
                        t_const=self.TE,
                        )

        self.LS = Lag(u=self.v, T=self.TR, K=1.0, info='Sensing lag TF')

        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        )
        self.vi.v_str = 'vb0'
        self.vi.e_str = '(vref - LS_y - W_y) - vi'

        self.LL = LeadLag(u=self.vi,
                          T1=self.TC,
                          T2=self.TB,
                          info='Lead-lag for internal delays',
                          zero_out=True,
                          )
        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VRMAX,
                                lower=self.VRMIN,
                                info='Anti-windup lag',
                                )
        self.W = Washout(u=self.vp,
                         T=self.TF1,
                         K=self.KF1,
                         info='Signal conditioner'
                         )
        self.vout.e_str = 'ue * omega * vp - vout'


class EXDC2(EXDC2Data, EXDC2Model):
    """
    EXDC2 model.
    """

    def __init__(self, system, config):
        EXDC2Data.__init__(self)
        EXDC2Model.__init__(self, system, config)
