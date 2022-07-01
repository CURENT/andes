"""
Module for ESAC5A exciter model.
"""

from andes.core.block import Lag, LagAntiWindup, LeadLag, Washout
from andes.core.param import NumParam
from andes.core.service import PostInitService, ConstService
from andes.core.var import Algeb
from andes.models.exciter.excbase import ExcACSat, ExcBase, ExcBaseData, ExcVsum


class ESAC5AData(ExcBaseData):
    """
    ESAC5A data.
    """

    def __init__(self):
        ExcBaseData.__init__(self)
        self.TR = NumParam(info='Sensing Time Constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u',
                           )

        self.TA = NumParam(info='Voltage Regulator Time Constant',
                           tex_name='T_A',
                           default=0.04,
                           unit='p.u',
                           )

        self.KA = NumParam(info='Voltage Regulator Gain',
                           tex_name='K_A',
                           default=80,
                           )

        self.VRMIN = NumParam(info='V_R lower limit',
                              default=-7.3,
                              unit='p.u',
                              tex_name='V_Rmin'
                              )

        self.VRMAX = NumParam(info='V_R upper limit',
                              default=7.3,
                              unit='p.u',
                              tex_name='V_Rmax',
                              )

        self.TE = NumParam(info='Integrator Time Constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u',
                           non_negative=True
                           )

        self.KF = NumParam(info='Feedback Gain',
                           default=0.03,
                           tex_name='K_F'
                           )

        self.TF1 = NumParam(info='Lag Time Constant',
                            default=1.0,
                            unit='p.u',
                            tex_name='T_F_1',
                            )

        self.TF2 = NumParam(info='Lead-Lag Time Constant (pole)',
                            default=0.8,
                            unit='p.u',
                            tex_name='T_F_2',
                            )

        self.TF3 = NumParam(info='Lead-Lag Time Constant (zero)',
                            default=1,
                            unit='p.u',
                            tex_name='T_F_3',
                            )

        self.KE = NumParam(info='Exciter Feedback Gain',
                           tex_name='K_E',
                           default=1,
                           )

        self.E1 = NumParam(info='First saturation point',
                           default=0,
                           unit='p.u.',
                           tex_name='E_1',
                           )
        self.SE1 = NumParam(info='Value at first saturation point',
                            default=0,
                            unit='p.u.',
                            tex_name='S_E1'
                            )
        self.E2 = NumParam(info='Second saturation point',
                           default=1,
                           unit='p.u.',
                           tex_name='E_2'
                           )
        self.SE2 = NumParam(info='Value at second saturation point',
                            default=1,
                            unit='p.u.',
                            tex_name='S_E2'
                            )


class ESAC5AModel(ExcBase, ExcVsum, ExcACSat):
    """
    ESAC5A model implementation.
    """

    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        ExcVsum.__init__(self)

        self.LP = Lag(u=self.v, T=self.TR, K=1, info='Voltage transducer',)

        self.vi = Algeb(info='Total voltage input',
                        unit='pu',
                        e_str='ue * (-LP_y + vref + Vs - WF_y ) -vi ',
                        v_str='ue*(-v +vref)',
                        )

        self.VRMAXu = ConstService('VRMAX * ue + (1-ue) * 999')
        self.VRMINu = ConstService('VRMIN * ue + (1-ue) * -999')

        self.VR = LagAntiWindup(u=self.vi, T=self.TA, K=self.KA, upper=self.VRMAXu, lower=self.VRMINu,)

        self.LL = LeadLag(u=self.VR_y, T1=self.TF3, T2=self.TF2,)

        self.WF = Washout(u=self.LL_y, T=self.TF1, K=self.KF)

        self.INTin = 'ue * (VR_y - VFE)'

        ExcACSat.__init__(self)

        self.vref.v_str = 'v + VFE / KA'

        self.vref0 = PostInitService(info='Initial reference voltage input',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        self.VFE.v_str = "INT_y * KE + Se "
        self.VFE.e_str = "ue * (INT_y * KE + Se - VFE) "

        # disable iterative initialization of the integrator output
        self.INT.y.v_str = 'vf0'
        self.INT.y.v_iter = None

        self.vout.e_str = 'ue * INT_y - vout'


class ESAC5A (ESAC5AData, ESAC5AModel):
    """
    Exciter ESAC5A.
    """

    def __init__(self, system, config):
        ESAC5AData.__init__(self)
        ESAC5AModel.__init__(self, system, config)
