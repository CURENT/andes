from andes.core import ModelData, IdxParam, NumParam, Model, ExtAlgeb
from andes.core import Lag, ExtService, ConstService, ExtParam, Algeb, State
from andes.core.block import PIController
from andes.core.var import AliasState, AliasAlgeb
# from andes.core.block import LagAntiWindupRate, GainLimiter, PIController

"""
REGC_VSG module.
voltage-controlled VSC with virtual synchronous generator control.
"""


class REGCVSGData(ModelData):
    """
    REGC_VSG model data.
    """
    def __init__(self):
        ModelData.__init__(self)

        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            mandatory=True,
                            )
        self.coi2 = IdxParam(model='COI2',
                             info="center of inertia 2 index",
                             )
        self.Sn = NumParam(default=100.0, tex_name='S_n',
                           info='Model MVA base',
                           unit='MVA',
                           )

        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f',
                           )
        self.Tc = NumParam(default=0.001, tex_name='T_c',
                           info='switch time constant',
                           unit='s',
                           )

        self.kP = NumParam(default=0.05, tex_name='k_P',
                           info='Active power droop on frequency (equivalent Droop)',
                           unit='p.u.',
                           )
        self.kv = NumParam(default=0, tex_name='k_v',
                           info='reactive power droop on voltage',
                           unit='p.u.',
                           )

        self.M = NumParam(default=10, tex_name='M',
                          info='Emulated startup time constant (inertia)',
                          unit='s',
                          )
        self.D = NumParam(default=2, tex_name='D',
                          info='Emulated damiping coefficient',
                          unit='p.u.',
                          )

        self.kp_dv = NumParam(default=20, tex_name=r'kp_{dv}',
                              info='d-axis v controller proportional gain',
                              unit='p.u.',
                              )
        self.ki_dv = NumParam(default=0.001, tex_name=r'ki_{dv}',
                              info='d-axis v controller integral gain',
                              unit='p.u.',
                              )
        self.kp_qv = NumParam(default=20, tex_name=r'kp_{qv}',
                              info='q-axis v controller proportional gain',
                              unit='p.u.',
                              )
        self.ki_qv = NumParam(default=0.001, tex_name=r'ki_{qv}',
                              info='q-axis v controller integral gain',
                              unit='p.u.',
                              )

        self.kp_di = NumParam(default=500, tex_name=r'kp_{di}',
                              info='d-axis i controller proportional gain',
                              unit='p.u.',
                              )
        self.ki_di = NumParam(default=0.2, tex_name=r'ki_{di}',
                              info='d-axis i controller integral gain',
                              unit='p.u.',
                              )
        self.kp_qi = NumParam(default=500, tex_name=r'kp_{qi}',
                              info='q-axis i controller proportional gain',
                              unit='p.u.',
                              )
        self.ki_qi = NumParam(default=0.2, tex_name=r'ki_{qi}',
                              info='q-axis i controller integral gain',
                              unit='p.u.',
                              )


class REGCVSGModel(Model):
    """
    REGC_VSG implementation.
    """
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'RenGen'

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle',
                          e_str='-u * Pe',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name='V',
                          info='Bus voltage magnitude',
                          e_str='-u * Qe',
                          )

        self.Pref = ExtService(model='StaticGen',
                               src='p',
                               indexer=self.gen,
                               tex_name=r'P_{ref}',
                               info='initial P of the static gen',
                               )
        self.Qref = ExtService(model='StaticGen',
                               src='q',
                               indexer=self.gen,
                               tex_name=r'Q_{ref}',
                               info='initial Q of the static gen',
                               )
        self.vref = ExtService(model='StaticGen',
                               src='v',
                               indexer=self.gen,
                               tex_name=r'V_{ref}',
                               info='initial v of the static gen',
                               )
        self.ra = ExtParam(model='StaticGen',
                           src='ra',
                           indexer=self.gen,
                           tex_name='r_a',
                           export=False,
                           )
        self.xs = ExtParam(model='StaticGen',
                           src='xs',
                           indexer=self.gen,
                           tex_name='x_s',
                           export=False,
                           )

        # --- INITIALIZATION ---
        self.ixs = ConstService(v_str='1/xs',
                                tex_name=r'1/xs',
                                )
        self.Id0 = ConstService(tex_name=r'I_{d0}',
                                v_str='u * Pref / v',
                                )
        self.Iq0 = ConstService(tex_name=r'I_{q0}',
                                v_str='- u * Qref / v',
                                )

        self.vd0 = ConstService(tex_name=r'v_{d0}',
                                v_str='u * v',
                                )
        self.vq0 = ConstService(tex_name=r'v_{q0}',
                                v_str='0',
                                )

        self.udref0 = ConstService(tex_name=r'u_{dref0}',
                                   v_str='ra * Id0 - xs * Iq0 + vd0',
                                   )
        self.uqref0 = ConstService(tex_name=r'u_{qref0}',
                                   v_str='ra * Iq0 + xs * Id0 + vq0',
                                   )

        self.Pref2 = Algeb(tex_name=r'P_{ref2}',
                           info='active power reference after adjusted by frequency',
                           e_str='u * Pref - dw * kP - Pref2',
                           v_str='u * Pref')

        self.vref2 = Algeb(tex_name=r'v_{ref2}',
                           info='voltage reference after adjusted by reactive power',
                           e_str='(u * Qref - Qe) * kv + vref - vref2',
                           v_str='u * vref')

        self.dw = State(info='delta virtual rotor speed',
                        unit='pu (Hz)',
                        v_str='0',
                        tex_name=r'\Delta\omega',
                        e_str='Pref2 - Pe - D * dw',
                        t_const=self.M)

        self.omega = Algeb(info='virtual rotor speed',
                           unit='pu (Hz)',
                           v_str='u',
                           tex_name=r'\omega',
                           e_str='1 + dw - omega')

        self.delta = State(info='virtual delta',
                           unit='rad',
                           v_str='a',
                           tex_name=r'\delta',
                           e_str='2 * pi * fn * dw')

        self.vd = Algeb(tex_name='V_d',
                        info='d-axis voltage',
                        e_str='u * v * cos(delta - a) - vd',
                        v_str='vd0')
        self.vq = Algeb(tex_name='V_q',
                        info='q-axis voltage',
                        e_str='- u * v * sin(delta - a) - vq',
                        v_str='vq0')

        self.PIdv = PIController(u='vref2 - vd',
                                 kp=self.kp_dv,
                                 ki=self.ki_dv,
                                 x0='Id0',
                                 )
        self.PIqv = PIController(u='- vq',
                                 kp=self.kp_qv,
                                 ki=self.ki_qv,
                                 x0='Iq0',
                                 )

        self.Idref = AliasAlgeb(self.PIdv_y)
        self.Iqref = AliasAlgeb(self.PIqv_y)

        self.Pe = Algeb(tex_name='P_e',
                        info='active power injection from VSC',
                        e_str='vd * Id + vq * Iq - Pe',
                        v_str='Pref')
        self.Qe = Algeb(tex_name='Q_e',
                        info='reactive power injection from VSC',
                        e_str='- vd * Iq + vq * Id - Qe',
                        v_str='Qref')

        # udLag_y, uqLag_y are ud, uq
        self.Id = Algeb(tex_name='I_d',
                        info='d-axis current',
                        e_str='- ra * ixs * Id  + ixs * (udLag_y - vd) + Iq',
                        v_str='Id0')
        self.Iq = Algeb(tex_name='I_q',
                        info='q-axis current',
                        e_str='- ra * ixs * Iq  + ixs * (uqLag_y - vq) - Id',
                        v_str='Iq0')

        # PIdv_y, PIqv_y are Idref, Iqref
        self.PIdi = PIController(u='PIdv_y - Id',
                                 kp=self.kp_di,
                                 ki=self.ki_di,
                                 )
        self.PIqi = PIController(u='PIqv_y - Iq',
                                 kp=self.kp_qi,
                                 ki=self.ki_qi,
                                 )

        self.udref = Algeb(tex_name=r'u_{dref}',
                           info='ud reference',
                           e_str='PIdi_y + vd - Iq * xs - udref',
                           v_str='udref0',
                           )
        self.uqref = Algeb(tex_name=r'u_{qref}',
                           info='ud reference',
                           e_str='PIqi_y + vq + Id * xs - uqref',
                           v_str='uqref0',
                           )

        self.udLag = Lag(u='udref',
                         T=self.Tc,
                         K=1,
                         )
        self.uqLag = Lag(u='uqref',
                         T=self.Tc,
                         K=1,
                         )

        self.ud = AliasState(self.udLag_y)
        self.uq = AliasState(self.uqLag_y)

    def v_numeric(self, **kwargs):
        """
        Disable the corresponding `StaticGen`s.
        """
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class REGCVSG(REGCVSGData, REGCVSGModel):
    """
    Voltage-controlled VSC with VSG control.
    Double-loop PI control.
    Swing equation based VSG control.
    The measure time-delay of voltage is ignored.
    """
    def __init__(self, system, config):
        REGCVSGData.__init__(self)
        REGCVSGModel.__init__(self, system, config)
