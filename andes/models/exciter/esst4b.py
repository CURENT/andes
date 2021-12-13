from collections import OrderedDict

from andes.core.block import (GainLimiter, Lag, LVGate, Piecewise,  # NOQA
                              PITrackAW,)
from andes.core.common import dummify
from andes.core.param import NumParam
from andes.core.service import ConstService, PostInitService, VarService
from andes.core.var import Algeb, ExtAlgeb
from andes.models.exciter.excbase import ExcBase, ExcBaseData


class ESST4BData(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)

        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u.',
                           )

        self.KPR = NumParam(info='Proportional gain 1',
                            tex_name='K_{PR}',
                            default=1,
                            unit='p.u.',
                            )
        self.KIR = NumParam(info='Integral gain 1',
                            tex_name='K_{IR}',
                            default=0,
                            unit='p.u.',
                            )

        self.VRMAX = NumParam(info='Maximum regulator limit',
                              tex_name='V_{RMAX}',
                              default=8,
                              unit='p.u.',
                              vrange=(0.5, 10),
                              )
        self.VRMIN = NumParam(info='Minimum regulator limit',
                              tex_name='V_{RMIN}',
                              default=0,
                              unit='p.u.',
                              vrange=(-10, 0.5),
                              )
        self.TA = NumParam(info='Lag time constant',
                           tex_name='T_A',
                           default=0.1,
                           vrange=(0, 1),
                           )
        self.KPM = NumParam(info='Proportional gain 2',
                            tex_name='K_{PM}',
                            default=1,
                            unit='p.u.',
                            )
        self.KIM = NumParam(info='Integral gain 2',
                            tex_name='K_{IM}',
                            default=0,
                            unit='p.u.',
                            )
        self.VMMAX = NumParam(info='Maximum inner loop limit',
                              tex_name='V_{RMAX}',
                              default=8,
                              unit='p.u.',
                              vrange=(0.5, 10),
                              )
        self.VMMIN = NumParam(info='Minimum inner loop limit',
                              tex_name='V_{RMIN}',
                              default=0,
                              unit='p.u.',
                              vrange=(-10, 0.5),
                              )
        self.KG = NumParam(info='Feedback gain of inner field regulator',
                           tex_name='K_G',
                           default=1,
                           vrange=(0, 1.1),
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
        self.VBMAX = NumParam(info='VB upper limit',
                              tex_name='V_{BMAX}',
                              default=18,
                              unit='p.u.',
                              vrange=(0, 20),
                              )
        self.KC = NumParam(default=0.1,
                           info='Rectifier loading factor proportional to commutating reactance',
                           tex_name='K_C',
                           vrange=(0, 1),
                           )
        self.XL = NumParam(default=0.01,
                           info='Potential source reactance',
                           tex_name='X_L',
                           vrange=(0, 0.5),
                           )
        self.THETAP = NumParam(info='Rectifier firing angle',
                               tex_name=r'\theta_P',
                               default=0,
                               unit='degree',
                               vrange=(0, 90),
                               )
        self.VGMAX = NumParam(info='VG upper limit',
                              tex_name='V_{GMAX}',
                              default=20,
                              unit='p.u.',
                              vrange=(0, 20),
                              )


class ESST4BModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        self.config.add(OrderedDict((('ksr', 2),
                                     ('ksm', 2),
                                     )))

        self.config.add_extra('_help',
                              ksr='Tracking gain for outer PI controller',
                              ksm='Tracking gain for inner PI controller',
                              )
        self.config.add_extra('_tex',
                              ksr='K_{sr}',
                              ksm='K_{sm}',
                              )

        self.KPC = ConstService(v_str='KP * exp(1j * radians(THETAP))',
                                tex_name='K_{PC}',
                                info='KP polar THETAP',
                                vtype=complex
                                )

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

        # control block begin
        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )

        self.UEL = Algeb(info='Interface var for under exc. limiter',
                         tex_name='U_{EL}',
                         v_str='0',
                         e_str='0 - UEL'
                         )

        # lower part: VB signal
        self.VE = VarService(tex_name='V_E',
                             info='VE',
                             v_str='Abs(KPC*(vd + 1j*vq) + 1j*(KI + KPC*XL)*(Id + 1j*Iq))',
                             )

        self.IN = Algeb(tex_name='I_N',
                        info='Input to FEX',
                        v_str='safe_div(KC * XadIfd, VE)',
                        e_str='ue * (KC * XadIfd - VE * IN)',
                        diag_eps=True,
                        )

        self.FEX = Piecewise(u=self.IN,
                             points=(0, 0.433, 0.75, 1),
                             funs=('1', '1 - 0.577*IN', 'sqrt(0.75 - IN ** 2)', '1.732*(1 - IN)', 0),
                             info='Piecewise function FEX',
                             )

        self.VBMIN = dummify(-9999)
        self.VGMIN = dummify(-9999)

        self.VB = GainLimiter(u='VE*FEX_y',
                              K=1,
                              R=1,
                              upper=self.VBMAX,
                              lower=self.VBMIN,
                              no_lower=True,
                              info='VB with limiter',
                              )

        self.VG = GainLimiter(u=self.vout,
                              K=self.KG,
                              R=1,
                              upper=self.VGMAX,
                              lower=self.VGMIN,
                              no_lower=True,
                              info='Feedback gain with HL',
                              )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='v',
                          e_str='vref0 - vref'
                          )
        self.vref0 = PostInitService(info='Const reference voltage',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='-LG_y + vref - vi',
                        v_str='-v + vref',
                        )

        self.PI1 = PITrackAW(u=self.vi,
                             kp=self.KPR,
                             ki=self.KIR,
                             ks=self.config.ksr,
                             lower=self.VRMIN,
                             upper=self.VRMAX,
                             x0='VG_y'
                             )

        self.LA = Lag(u=self.PI1_y, T=self.TA, K=1.0,
                      info='Regulation delay',
                      )
        self.PI2 = PITrackAW(u='LA_y - VG_y',
                             kp=self.KPM,
                             ki=self.KIM,
                             ks=self.config.ksm,
                             lower=self.VMMIN,
                             upper=self.VMMAX,
                             x0='safe_div(vf0, VB_y)',
                             )

        # TODO: add back LV Gate

        self.vout.e_str = 'ue * VB_y * PI2_y - vout'


class ESST4B(ESST4BData, ESST4BModel):
    def __init__(self, system, config):
        ESST4BData.__init__(self)
        ESST4BModel.__init__(self, system, config)
