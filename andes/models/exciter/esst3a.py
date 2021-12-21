from andes.core.block import (GainLimiter, HVGate, Lag, LagAntiWindup, LeadLag,
                              Piecewise,)
from andes.core.common import dummify
from andes.core.discrete import HardLimiter
from andes.core.param import NumParam
from andes.core.service import ConstService, PostInitService, VarService
from andes.core.var import Algeb, ExtAlgeb
from andes.models.exciter.excbase import ExcBase, ExcBaseData


class ESST3AData(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u.',
                           )
        self.VIMAX = NumParam(default=0.8,
                              info='Max. input voltage',
                              tex_name='V_{IMAX}',
                              vrange=(0, 1),
                              )
        self.VIMIN = NumParam(default=-0.1,
                              info='Min. input voltage',
                              tex_name='V_{IMIN}',
                              vrange=(-1, 0),
                              )

        self.KM = NumParam(default=500,
                           tex_name='K_M',
                           info='Forward gain constant',
                           vrange=(0, 1000),
                           )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=3,
                           vrange=(0, 20),
                           )
        self.TB = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_B',
                           default=15,
                           vrange=(0, 20),
                           )

        self.KA = NumParam(info='Gain in anti-windup lag TF',
                           tex_name='K_A',
                           default=50,
                           vrange=(0, 200),
                           )
        self.TA = NumParam(info='Lag time constant in anti-windup lag',
                           tex_name='T_A',
                           default=0.1,
                           vrange=(0, 1),
                           )
        self.VRMAX = NumParam(info='Maximum excitation limit',
                              tex_name='V_{RMAX}',
                              default=8,
                              unit='p.u.',
                              vrange=(0.5, 10),
                              )
        self.VRMIN = NumParam(info='Minimum excitation limit',
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
        self.VGMAX = NumParam(info='VG upper limit',
                              tex_name='V_{GMAX}',
                              default=4,
                              unit='p.u.',
                              vrange=(0, 20),
                              )
        self.THETAP = NumParam(info='Rectifier firing angle',
                               tex_name=r'\theta_P',
                               default=0,
                               unit='degree',
                               vrange=(0, 90),
                               )
        self.TM = NumParam(default=0.1,
                           info='Inner field regulator forward time constant',
                           tex_name='K_C',
                           )

        self.VMMAX = NumParam(info='Maximum VM limit',
                              tex_name='V_{MMAX}',
                              default=1,
                              unit='p.u.',
                              vrange=(0.5, 1.5),
                              )
        self.VMMIN = NumParam(info='Minimum VM limit',
                              tex_name='V_{RMIN}',
                              default=0.1,
                              unit='p.u.',
                              vrange=(-1.5, 0.5),
                              )


class ESST3AModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

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

        self.UEL0 = ConstService(v_str='-9999',
                                 tex_name='U_{EL0}',
                                 info='initial UEL input'
                                 )

        self.UEL = Algeb(info='Interface var for under exc. limiter',
                         tex_name='U_{EL}',
                         v_str='UEL0',
                         e_str='UEL0 - UEL'
                         )

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
                             funs=('1',
                                   '1 - 0.577*IN',
                                   'sqrt(0.75 - IN ** 2)',
                                   '1.732 * (1 - IN)',
                                   '0'),
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

        self.vrs = Algeb(tex_name='V_{RS}',
                         info='VR subtract feedback VG',
                         v_str='safe_div(vf0, VB_y) / KM',
                         e_str='LAW1_y - VG_y - vrs',
                         )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='(vrs + VG_y) / KA + v',
                          e_str='vref0 - vref',
                          )

        self.vref0 = PostInitService(info='Initial reference voltage input',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='-LG_y + vref - vi',
                        v_str='-v + vref',
                        )

        self.vil = Algeb(info='Input voltage after limit',
                         tex_name='V_{il}',
                         v_str='HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX',
                         e_str='HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX - vil'
                         )

        self.HG = HVGate(u1=self.UEL,
                         u2=self.vil,
                         info='HVGate for under excitation',
                         )

        self.LL = LeadLag(u=self.HG_y, T1=self.TC, T2=self.TB,
                          info='Regulator',
                          zero_out=True,
                          )  # LL_y == VA

        self.LAW1 = LagAntiWindup(u=self.LL_y,
                                  T=self.TA,
                                  K=self.KA,
                                  lower=self.VRMIN,
                                  upper=self.VRMAX,
                                  info='Lag AW on VR',
                                  )  # LAW1_y == VR

        self.HLI = HardLimiter(u=self.vi,
                               lower=self.VIMIN,
                               upper=self.VIMAX,
                               info='Input limiter',
                               )

        self.LAW2 = LagAntiWindup(u=self.vrs,
                                  T=self.TM,
                                  K=self.KM,
                                  lower=self.VMMIN,
                                  upper=self.VMMAX,
                                  info='Lag AW on VM',
                                  )  # LAW2_y == VM

        self.vout.e_str = 'ue * VB_y * LAW2_y - vout'


class ESST3A(ESST3AData, ESST3AModel):
    """
    Static exciter type 3A model
    """

    def __init__(self, system, config):
        ESST3AData.__init__(self)
        ESST3AModel.__init__(self, system, config)
