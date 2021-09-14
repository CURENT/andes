from andes.core.param import NumParam
from andes.core.var import Algeb

from andes.core.service import ConstService, VarService, FlagValue

from andes.core.block import LagAntiWindup, LeadLag, Washout, Lag, HVGate
from andes.core.block import LessThan
from andes.core.block import Integrator

from andes.models.exciter.excbase import ExcBase, ExcBaseData
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
        self.KA = NumParam(default=80,
                           info='Regulator gain',
                           tex_name='K_A',
                           )
        self.TA = NumParam(info='Lag time constant in regulator',
                           tex_name='T_A',
                           default=0.04,
                           unit='p.u.',
                           )



class ESAC1AModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        # control block begin
        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )

        # TODO: Question: where is the limiter in ESDC2A?
        # TODO: Question: the sequence of HG and LL in ESDC2A?
        # TODO: add vref

        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='-LG_y + vref - vi',
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



class ESAC1A(ESAC1AData, ESAC1AModel):
    """
    Static exciter type 3A model
    """

    def __init__(self, system, config):
        ESAC1AData.__init__(self)
        ESAC1AModel.__init__(self, system, config)
