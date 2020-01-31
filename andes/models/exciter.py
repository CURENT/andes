from andes.core.model import ModelData, Model
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.var import Algeb, ExtState, ExtAlgeb, State
from andes.core.service import ConstService, ExtService
from andes.core.block import LagAntiWindup, LeadLag, Washout, Lag


class ExcBaseData(ModelData):
    """
    Common parameters for exciters.
    """
    def __init__(self):
        super().__init__()
        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            )


class ExcBase(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'Exciter'
        self.flags.update({'tds': True})

        # from synchronous generators, get Sn, Vn, bus; tm0; omega
        self.Sn = ExtParam(src='Sn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='S_m',
                           info='Rated power from generator',
                           unit='MVA',
                           export=False,
                           )
        self.Vn = ExtParam(src='Vn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='V_m',
                           info='Rated voltage from generator',
                           unit='kV',
                           export=False,
                           )
        self.vf0 = ExtService(src='vf',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name='v_{f0}',
                              info='Steady state excitation voltage')
        self.bus = ExtParam(src='bus',
                            model='SynGen',
                            indexer=self.syn,
                            tex_name='bus',
                            info='Bus idx of the generators',
                            export=False,
                            )
        self.omega = ExtState(src='omega',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name=r'\omega',
                              info='Generator speed',
                              )
        self.vf = ExtAlgeb(src='vf',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'v_f',
                           e_str='u*(vout - vf0)',
                           info='Excitation field voltage to generator',
                           )
        # from bus, get a and v
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          )

        # output excitation voltage
        self.vout = Algeb(info='Exciter final output voltage',
                          tex_name='v_{out}',
                          v_str='vf0',
                          )


class EXDC2Data(ExcBaseData):
    def __init__(self):
        super().__init__()
        self.Tr = NumParam(info='Sensing time constant',
                           tex_name='T_r',
                           default=1,
                           unit='p.u.',
                           )
        self.Ta = NumParam(info='Lag time constant in anti-windup lag',
                           tex_name='T_a',
                           default=0.04,
                           unit='p.u.',
                           )
        self.Tc = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_c',
                           default=1,
                           unit='p.u.',
                           )
        self.Tb = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_b',
                           default=1,
                           unit='p.u.',
                           )
        self.Te = NumParam(info='Exciter integrator time constant',
                           tex_name='T_e',
                           default=0.8,
                           unit='p.u.',
                           )
        self.Tf = NumParam(info='Feedback washout time constant',
                           tex_name='T_f',
                           default=1,
                           unit='p.u.',
                           non_zero=True
                           )

        self.Kf = NumParam(info='Feedback washout gain',
                           tex_name='K_f',
                           default=0.03,
                           unit='p.u.',
                           )
        self.Ka = NumParam(info='Gain in anti-windup lag TF',
                           tex_name='K_a',
                           default=40,
                           unit='p.u.',
                           )
        self.Ke = NumParam(info='Gain added to saturation',
                           tex_name='K_e',
                           default=1,
                           unit='p.u.',
                           )
        self.vrmax = NumParam(info='Maximum excitation limit',
                              tex_name='V_{rmax}',
                              default=7.3,
                              unit='p.u.')
        self.vrmin = NumParam(info='Minimum excitation limit',
                              tex_name='V_{rmin}',
                              default=-7.3,
                              unit='p.u.')
        self.Ae = NumParam(info='Gain in saturation',
                           tex_name='A_e',
                           default=0.0,
                           unit='p.u.',
                           )
        self.Be = NumParam(info='Exponential coefficient in saturation',
                           tex_name='B_e',
                           default=0.0,
                           unit='p.u.',
                           )


class EXDC2Model(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)
        self.Se0 = ConstService(info='Initial saturation output',
                                tex_name='S_{e0}',
                                v_str='Ae * exp(Be * vf0)',
                                )
        self.vr0 = ConstService(info='Initial vr',
                                tex_name='V_{r0}',
                                v_str='(Ke + Se0) * vf0')
        self.vb0 = ConstService(info='Initial vb',
                                tex_name='V_{b0}',
                                v_str='vr0 / Ka')

        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name='V_{ref0}',
                                  v_str='vb0 + v')

        self.Se = Algeb(info='Saturation output',
                        tex_name='S_e',
                        unit='p.u.',
                        v_str='Se0',
                        e_str='Ae * exp(Be * vout) - Se'
                        )
        self.vp = State(info='Voltage after saturation feedback, before speed term',
                        tex_name='V_p',
                        unit='p.u.',
                        v_str='vf0',
                        e_str='(LA_x - Ke * vp - Se * vp) / Te')

        self.LS = Lag(u=self.v, T=self.Tr, K=1.0, info='Sensing lag TF')

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='vref0',
                          e_str='vref0 - vref'
                          )
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        v_str='vb0',
                        e_str='(vref - LS_x - W_y) - vi',
                        )

        self.LL = LeadLag(u=self.vi,
                          T1=self.Tc,
                          T2=self.Tb,
                          info='Lead-lag for internal delays',
                          )
        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.Ta,
                                K=self.Ka,
                                upper=self.vrmax,
                                lower=self.vrmin,
                                info='Anti-windup lag',
                                )
        self.W = Washout(u=self.vp,
                         T=self.Tf,
                         K=self.Kf,
                         )
        self.vout.e_str = 'omega * vp - vout'


class EXDC2(EXDC2Data, EXDC2Model):
    """
    EXDC2 model.
    """
    def __init__(self, system, config):
        EXDC2Data.__init__(self)
        EXDC2Model.__init__(self, system, config)
