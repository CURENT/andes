from andes.core.block import Lag, LagAntiWindup, LessThan, Washout, Piecewise
from andes.core.discrete import HardLimiter
from andes.core.param import NumParam
from andes.core.service import (ConstService, FlagValue, PostInitService,
                                VarService,)
from andes.core.var import Algeb, ExtAlgeb
from andes.models.exciter.excbase import ExcBase, ExcBaseData, ExcVsum


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
        self.VRMAX = NumParam(info='Maximum regulator limit',
                              tex_name=r'V_{RMAX}',
                              default=7.3,
                              unit='p.u.')
        self.VRMIN = NumParam(info='Minimum regulator limit',
                              tex_name=r'V_{RMIN}',
                              default=-7.3,
                              unit='p.u.')
        self.VBMAX = NumParam(info='VB upper limit',
                              tex_name='V_{BMAX}',
                              default=18,
                              unit='p.u.',
                              vrange=(0, 20),
                              )

        self.KE = NumParam(info='Exciter integrator constant',
                           tex_name='K_E',
                           default=1,
                           unit='p.u.',
                           )

        self.TE = NumParam(info='Exciter integrator time constant',
                           tex_name='T_E',
                           default=1,
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


class IEEET3Model(ExcBase, ExcVsum):
    """
    IEEE Type 3 exciter model.
    """

    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        # vd, vq, Id, Iq from SynGen
        self.vd = ExtAlgeb(src='vd',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'V_d',
                           info='d-axis machine voltage',
                           )
        self.vq = ExtAlgeb(src='vq',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'V_q',
                           info='q-axis machine voltage',
                           )
        self.Id = ExtAlgeb(src='Id',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'I_d',
                           info='d-axis machine current',
                           )
        self.Iq = ExtAlgeb(src='Iq',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'I_q',
                           info='q-axis machine current',
                           )
        self.VE = VarService(tex_name=r'V_{E}',
                             info=r'V_{E}',
                             v_str='Abs(KP * (vd + 1j*vq) + 1j*KI*(Id + 1j*Iq))',
                             )

        self.V40 = ConstService('sqrt(VE ** 2 - (0.78 * XadIfd) ** 2)')
        self.VR0 = ConstService(info='Initial VR',
                                tex_name='V_{R0}',
                                v_str='vf0 * KE - V40')

        self.vb0 = ConstService(info='Initial vb',
                                tex_name='V_{b0}',
                                v_str='VR0 / KA')

        # Set VRMAX to 999 when VRMAX = 0
        self._zVRM = FlagValue(self.VRMAX, value=0,
                               tex_name='z_{VRMAX}',
                               )
        self.VRMAXc = ConstService(v_str='VRMAX + 999*(1-_zVRM)',
                                   info='Set VRMAX=999 when zero',
                                   )

        self.LG = Lag(u=self.v, T=self.TR, K=1,
                      info='Sensing delay')

        ExcVsum.__init__(self)

        self.vref.v_str = 'v + vb0'

        self.vref0 = PostInitService(info='Constant vref',
                                     tex_name='V_{ref0}',
                                     v_str='vref')

        # NOTE: for offline exciters, `vi` equation ignores ext. voltage changes
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='ue * (-LG_y + vref + UEL + OEL + Vs - vi)',
                        v_str='vref - v',
                        diag_eps=True,
                        )

        self.LA3 = LagAntiWindup(u='ue * (vi - WF_y)',
                                 T=self.TA,
                                 K=self.KA,
                                 upper=self.VRMAXc,
                                 lower=self.VRMIN,
                                 info=r'V_{R}, Lag Anti-Windup',
                                 )  # LA3_y is V_R

        # FIXME: antiwindup out of limit is not warned of in initialization

        self.zeros = ConstService(v_str='0.0')

        self.LA1 = Lag('ue * (VB_y * HL_zi + VBMAX * HL_zu)',
                       T=self.TE, K=1, D=self.KE,
                       )

        self.WF = Washout(u=self.LA1_y, T=self.TF, K=self.KF,
                          info='V_F, stablizing circuit feedback, washout')

        self.SQE = Algeb(tex_name=r'SQE', info=r'Square of error after mul',
                         v_str='VE ** 2 - (0.78 * XadIfd) ** 2',
                         e_str='VE ** 2 - (0.78 * XadIfd) ** 2 - SQE',
                         )

        self.SL = LessThan(u=self.zeros, bound=self.SQE,
                           equal=False, enable=True, cache=False)

        self.VB = Piecewise(self.SQE, points=(0, ), funs=('ue * LA3_y', 'ue * (sqrt(SQE) + LA3_y)'))

        self.HL = HardLimiter(u=self.VB_y, lower=self.zeros, upper=self.VBMAX,
                              info='Hard limiter for VB',
                              )

        self.vout.e_str = 'ue * (LA1_y - vout)'


class IEEET3(IEEET3Data, IEEET3Model):
    """
    Exciter IEEET3.

    Reference:

    [1] PowerWorld, Exciter IEEET3, [Online],

    [2] NEPLAN, Exciters Models, [Online],

    Available:

    https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Exciter%20IEEET3.htm

    https://www.neplan.ch/wp-content/uploads/2015/08/Nep_EXCITERS1.pdf
    """

    def __init__(self, system, config):
        IEEET3Data.__init__(self)
        IEEET3Model.__init__(self, system, config)
