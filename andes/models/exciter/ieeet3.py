from andes.core.param import NumParam
from andes.core.var import Algeb

from andes.core.service import ConstService, FlagValue

from andes.core.block import LagAntiWindup, Washout, Lag
from andes.core.block import LessThan
from andes.core.block import Integrator

from andes.models.exciter.excbase import ExcBase, ExcBaseData
from andes.models.exciter.saturation import ExcQuadSat


class IEEET3Data(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)

        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.02,
                           unit='p.u.',
                           )
        self.KA = NumParam(info='Regulator gain',
                           tex_name='K_A',
                           default=5.0,
                           unit='p.u.',
                           )
        self.TA = NumParam(info='Lag time constant in anti-windup lag',
                           tex_name='T_A',
                           default=0.04,
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

        ## info may need change
        self.KE = NumParam(info='Gain added to saturation',
                           tex_name='K_E',
                           default=1,
                           unit='p.u.',
                           )
        ## info may need change
        self.TE = NumParam(info='Exciter integrator time constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u.',
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

        self.KP = NumParam(info='Potential circuit gain coeff.',
                           tex_name='K_P',
                           default=4,
                           vrange=(1, 10),
                           )
        self.KI = NumParam(info='Potential circuit gain coeff.',
                           tex_name='K_I',
                           default=0.1,
                           vrange=(0, 1.1),
                           )


class IEEET3Model(ExcBase):
    """
    IEEE Type 3 exciter model.
    """

    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        ## Check if this is approriate for ieeet3
        # Set VRMAX to 999 when VRMAX = 0
        self._zVRM = FlagValue(self.VRMAX, value=0,
                               tex_name='z_{VRMAX}',
                               )
        self.VRMAXc = ConstService(v_str='VRMAX + 999*(1-_zVRM)',
                                   info='Set VRMAX=999 when zero',
                                   )

        ## It seems that IEEET3 does not consider saturation
        # Saturation
        self.SAT = ExcQuadSat(self.E1, self.SE1, self.E2, self.SE2,
                              info='Field voltage saturation',
                              )

        self.Se0 = ConstService(info='Initial saturation output',
                                tex_name='S_{e0}',
                                v_str='Indicator(vf0>SAT_A) * SAT_B * (SAT_A - vf0) ** 2',
                                )

        ## Checkout what these eqns mean
        self.vr0 = ConstService(info='Initial vr',
                                tex_name='V_{r0}',
                                v_str='KE * vf0 + Se0')
        self.vb0 = ConstService(info='Initial vb',
                                tex_name='V_{b0}',
                                v_str='vr0 / KA')
        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name='V_{ref0}',
                                  v_str='v + vb0',
                                  )
        self.vfe0 = ConstService(v_str='vf0 * KE + Se0',
                                 tex_name='V_{FE0}',
                                 )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='vref0',
                          e_str='vref0 - vref'
                          )

        self.LG = Lag(u=self.v, T=self.TR, K=1,
                      info='Sensing delay',
                      )

        ## modify vi eqn by IEEET1
        # NOTE: for offline exciters, `vi` equation ignores ext. voltage changes
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='ue * (-LG_y + vref - vi)',
                        v_str='-v + vref',
                        diag_eps=True,
                        )

        ## upper may need modify
        self.LA = LagAntiWindup(u='ue * (vi - WF_y)',
                                T=self.TA,
                                K=self.KA,
                                upper=self.VRMAXc,
                                lower=self.VRMIN,
                                info='Anti-windup lag',
                                )

        ## may not need
        self.VFE = Algeb(info='Combined saturation feedback',
                         tex_name='V_{FE}',
                         unit='p.u.',
                         v_str='vfe0',
                         e_str='ue * (INT_y * KE + Se - VFE)',
                         diag_eps=True,
                         )

        self.INT = Integrator(u='ue * (LA_y - VFE)',
                              T=self.TE,
                              K=1,
                              y0=self.vf0,
                              info='Integrator',
                              )

        self.SL = LessThan(u=self.vout, bound=self.SAT_A, equal=False, enable=True, cache=False)

        self.Se = Algeb(tex_name=r"S_e(|V_{out}|)", info='saturation output',
                        v_str='Se0',
                        e_str='SL_z0 * (INT_y - SAT_A) ** 2 * SAT_B - Se',
                        )

        self.WF = Washout(u=self.vout, T=self.TF, K=self.KF, info='Stablizing circuit feedback')

        self.vout.e_str = 'ue * (INT_y - vout)'


class IEEET3(IEEET3Data, IEEET3Model):
    '''
    Reference:
    [1] PowerWorld, Exciter IEEET3, [Online],
    Available:

    https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Exciter%20IEEET3.htm
    '''
    def __init__(self, system, config):
        IEEET3Data.__init__(self)
        IEEET3Model.__init__(self, system, config)
