from collections import OrderedDict

from andes.core.block import Lag, LagAntiWindup, PIDTrackAW, Piecewise
from andes.core.param import NumParam
from andes.core.service import PostInitService
from andes.core.var import Algeb
from andes.models.exciter.excbase import (ExcACSat, ExcBase, ExcBaseData,
                                          ExcVsum,)


class AC8BData(ExcBaseData):
    """
    AC8B data.
    """

    def __init__(self):
        ExcBaseData.__init__(self)
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u.',
                           )

        self.kP = NumParam(info='PID proportional coeff.',
                           tex_name='k_P',
                           default=10,
                           vrange=(10, 500),
                           )
        self.kI = NumParam(info='PID integrative coeff.',
                           tex_name='k_I',
                           default=10,
                           vrange=(10, 500),
                           )
        self.kD = NumParam(info='PID derivative coeff.',
                           tex_name='k_D',
                           default=10,
                           vrange=(10, 500),
                           )
        self.Td = NumParam(info='PID derivative time constant.',
                           tex_name='T_d',
                           default=0.2,
                           vrange=(0, 0.5),
                           )

        self.VPMAX = NumParam(info='PID maximum limit',
                              tex_name=r'V_{PMAX}',
                              default=999,
                              unit='p.u.')
        self.VPMIN = NumParam(info='PID minimum limit',
                              tex_name=r'V_{PMIN}',
                              default=-999,
                              unit='p.u.')

        self.VRMAX = NumParam(info='Maximum regulator limit',
                              tex_name=r'V_{RMAX}',
                              default=7.3,
                              unit='p.u.',
                              vrange=(1, 10))
        self.VRMIN = NumParam(info='Minimum regulator limit',
                              tex_name=r'V_{RMIN}',
                              default=1,
                              unit='p.u.',
                              vrange=(-1, 1.5))

        self.VFEMAX = NumParam(info='Exciter field current limit',
                               tex_name=r'V_{FEMAX}',
                               default=999,
                               unit='p.u.')

        self.VEMIN = NumParam(info='Minimum exciter voltage output',
                              tex_name=r'V_{EMIN}',
                              default=-999,
                              unit='p.u.')

        self.TA = NumParam(info='Lag time constant in anti-windup lag',
                           tex_name='T_A',
                           default=0.04,
                           unit='p.u.',
                           )
        self.KA = NumParam(info='Gain in anti-windup lag TF',
                           tex_name='K_A',
                           default=40,
                           unit='p.u.',
                           )
        self.TE = NumParam(info='Exciter integrator time constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u.',
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

        self.KE = NumParam(info='Gain added to saturation',
                           tex_name='K_E',
                           default=1,
                           unit='p.u.',
                           )
        self.KD = NumParam(default=0,
                           info='Ifd feedback gain',
                           tex_name='K_D',
                           vrange=(0, 1),
                           )

        self.KC = NumParam(default=0.1,
                           info='Rectifier loading factor proportional to commutating reactance',
                           tex_name='K_C',
                           vrange=(0, 1),
                           )


class AC8BModel(ExcBase, ExcVsum, ExcACSat):
    """
    Implementation of the AC8B model.
    """

    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        self.flags.nr_iter = True
        self.config.add(OrderedDict((('ks', 2),
                                     )))
        self.config.add_extra('_help',
                              ks='Tracking gain for PID controller',
                              )

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
        self.FEX.y.v_str = '0.5'
        self.FEX.y.v_iter = self.FEX.y.e_str

        # control block begin
        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )

        ExcVsum.__init__(self)

        self.vref.v_str = 'v'

        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='ue * (-LG_y + vref + UEL + OEL + Vs - vi)',
                        v_str='-v + vref',
                        diag_eps=True,
                        )

        # chekck y0
        self.PID = PIDTrackAW(u=self.vi, kp=self.kP, ki=self.kI,
                              ks=self.config.ks,
                              kd=self.kD, Td=self.Td, x0='VFE / KA',
                              lower=self.VPMIN, upper=self.VPMAX,
                              tex_name='PID', info='PID', name='PID',
                              )

        self.LA = LagAntiWindup(u=self.PID_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VRMAX,
                                lower=self.VRMIN,
                                info=r'V_{R}, Anti-windup lag',
                                )

        self.INTin = 'ue * (LA_y - VFE)'

        ExcACSat.__init__(self)

        self.vref0 = PostInitService(info='Initial reference voltage input',
                                     tex_name='V_{ref0}',
                                     v_str='v',
                                     )

        self.vout.e_str = 'ue * (FEX_y * INT_y - vout)'


class AC8B(AC8BData, AC8BModel):
    """
    Exciter AC8B model.

    Reference: [1]_, [2]_

    .. [1] Powerworld, Exciter AC8B, [Online], Available:
      https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Exciter%20AC8B.htm

    .. [2] NEPLAN, Exciters Models, [Online], Available:
      https://www.neplan.ch/wp-content/uploads/2015/08/Nep_EXCITERS1.pdf
    """
    def __init__(self, system, config):
        AC8BData.__init__(self)
        AC8BModel.__init__(self, system, config)
