"""
EXAC1 exciter.
"""

from andes.core.block import (Integrator, Lag, LagAntiWindup, LeadLag,
                              Piecewise, Washout,)
from andes.core.discrete import LessThan
from andes.core.param import NumParam
from andes.core.service import PostInitService
from andes.core.var import Algeb
from andes.models.exciter.excbase import ExcBase, ExcBaseData
from andes.models.exciter.saturation import ExcQuadSat


class EXAC1Data(ExcBaseData):
    """
    EXAC1 parameters.
    """

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
        self.VRMAX = NumParam(info='Maximum regulator output',
                              tex_name='V_{RMAX}',
                              default=8,
                              unit='p.u.',
                              vrange=(0.5, 10),
                              )
        self.VRMIN = NumParam(info='Minimum regulator output',
                              tex_name='V_{RMIN}',
                              default=0,
                              unit='p.u.',
                              vrange=(-10, 0.5),
                              )
        self.TE = NumParam(info='Exciter integrator time constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u.',
                           non_negative=True,
                           )
        self.KF = NumParam(default=0.1,
                           info='Feedback gain',
                           tex_name='K_F',
                           )
        self.TF = NumParam(default=1.0,
                           info='Feedback delay',
                           tex_name='T_F',
                           non_negative=True,
                           non_zero=True,
                           )
        self.KC = NumParam(default=0.1,
                           info='Rectifier loading factor proportional to commutating reactance',
                           tex_name='K_C',
                           vrange=(0, 1),
                           )
        self.KD = NumParam(default=0,
                           info='Ifd feedback gain',
                           tex_name='K_C',
                           vrange=(0, 1),
                           )
        self.KE = NumParam(info='Saturation feedback gain',
                           tex_name='K_E',
                           default=1,
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


class EXAC1Model(ExcBase):
    """
    EXAC1 implementation.

    The model contains an algebraic loop that will be iteratively initialized.
    The algebraic loop contains variables ``IN``, ``FEX_y`` and ``INT_y``.

    The input to the integrator ``VFE`` is calculated using the solved ``INT_y``
    and the saturation coefficients.
    """

    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        self.flags.nr_iter = True

        self.SAT = ExcQuadSat(self.E1, self.SE1, self.E2, self.SE2,
                              info='Field voltage saturation',
                              )

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

        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )

        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='ue * (-v + vref - WF_y - vi)',
                        v_str='-v + vref',
                        diag_eps=True,
                        )

        self.LL = LeadLag(u=self.vi, T1=self.TC, T2=self.TB,
                          info='Regulator',
                          zero_out=True,
                          )

        # LA_y is VR
        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.TA,
                                K=self.KA,
                                lower=self.VRMIN,
                                upper=self.VRMAX,
                                info='Lag AW on VR',
                                )

        self.INT = Integrator(u='ue * (LA_y - VFE)',
                              T=self.TE,
                              K=1,
                              y0=0,
                              info='Integrator',
                              )
        self.INT.y.v_str = 0.1
        self.INT.y.v_iter = 'INT_y * FEX_y - vf0'

        self.SL = LessThan(u=self.INT_y, bound=self.SAT_A, equal=False, enable=True, cache=False)

        self.Se = Algeb(tex_name=r"V_{out}*S_e(|V_{out}|)", info='saturation output',
                        v_str='Indicator(INT_y > SAT_A) * SAT_B * (INT_y - SAT_A) ** 2',
                        e_str='ue * (SL_z0 * (INT_y - SAT_A) ** 2 * SAT_B - Se)',
                        diag_eps=True,
                        )

        self.VFE = Algeb(info='Combined saturation feedback',
                         tex_name='V_{FE}',
                         unit='p.u.',
                         v_str='INT_y * KE + Se + XadIfd * KD',
                         e_str='ue * (INT_y * KE + Se + XadIfd * KD - VFE)',
                         diag_eps=True,
                         )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='v + VFE / KA',
                          e_str='vref0 - vref',
                          )

        self.vref0 = PostInitService(info='Initial reference voltage input',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        self.WF = Washout(u=self.VFE,
                          T=self.TF,
                          K=self.KF,
                          info='Stablizing circuit feedback',
                          )

        self.vout.e_str = 'ue * (INT_y * FEX_y - vout)'


class EXAC1(EXAC1Data, EXAC1Model):
    """
    EXAC1 model.

    Ref: https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Exciter%20EXAC1.htm
    """

    def __init__(self, system, config):
        EXAC1Data.__init__(self)
        EXAC1Model.__init__(self, system, config)
