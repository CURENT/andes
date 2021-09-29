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


        self.vin = Algeb(info='input voltage',
                         tex_name='v_{in}',
                         v_str='VFE',
                         e_str='ue * (1-LVC_zl) * OEL + LVC_zl * HVG_y - vin',
                         )

        self.HVG = HVGate(u1=self.UEL,
                          u2=self.LA_y,
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
