from andes.core.param import NumParam
from andes.core.var import Algeb

from andes.core.service import ConstService, PostInitService
from andes.core.discrete import LessThan, Limiter
from andes.core.block import LagAntiWindup, LeadLag, Washout, Lag, HVGate
from andes.core.block import Piecewise, Integrator

from andes.models.exciter.excbase import ExcBase, ExcBaseData, ExcVsum
from andes.models.exciter.saturation import ExcQuadSat


class ESST1AData(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)

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



class ESST1AModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        ExcVsum.__init__(self)




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
                         )

        self.ILR0 = ConstService(v_str='0', tex_name='I_{LR0}', info='ILR initial value')

        self.ILR = Algeb(info='exciter output current limit reference',
                         tex_name='I_{LR}}',
                         v_str='VFE',
                         e_str='ILR0 - ILR',
                         )

        self.HVG = HVGate(u1=self.UEL,
                          u2='LA_y - KLR * (ILR - XadIfd)',
                          info='HVGate for under excitation',
                          )

        self.ubd = ConstService('9999')
        self.LVC = Limiter(u=self.HVG_y, lower=self.OEL, upper=self.ubd,
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

        self.vout.e_str = 'VOL_zi * LVG + VOL_zu * VRMAX + VOL_zl * VRMIN - VR',



class ESST1A(ESST1AData, ESST1AModel):
    """
    Exciter ESST1A.
    """

    def __init__(self, system, config):
        ESST1AData.__init__(self)
        ESST1AModel.__init__(self, system, config)
