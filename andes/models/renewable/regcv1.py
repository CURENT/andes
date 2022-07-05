"""
REGCV1 model.

Voltage-controlled converter model (virtual synchronous generator) with inertia emulation.
"""

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtService, IdxParam,
                        Lag, Model, ModelData, NumParam, State,)
from andes.core.block import PIController
from andes.core.var import AliasAlgeb, AliasState


class REGCV1Data(ModelData):
    """
    REGCV1 model data.
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
        self.Tc = NumParam(default=0.01, tex_name='T_c',
                           info='switch time constant',
                           unit='s',
                           )

        self.kw = NumParam(default=0.0, tex_name=r'k_\omega',
                           info='speed droop on active power (reciprocal of droop)',
                           unit='p.u.',
                           ipower=True,
                           non_negative=True,
                           )
        self.kv = NumParam(default=0, tex_name='k_v',
                           info='reactive power droop on voltage',
                           unit='p.u.',
                           power=True,
                           non_negative=True,
                           )

        self.M = NumParam(default=10, tex_name='M',
                          info='Emulated startup time constant (M=2H)',
                          unit='s',
                          power=True,
                          )
        self.D = NumParam(default=0, tex_name='D',
                          info='Emulated damping coefficient',
                          unit='p.u.',
                          power=True,
                          )

        self.ra = NumParam(default=0.0,
                           info="resistance",
                           z=True,
                           tex_name='r_a'
                           )
        self.xs = NumParam(default=0.2,
                           info="reactance",
                           z=True,
                           tex_name='x_s'
                           )

        self.gammap = NumParam(default=1.0,
                               info="P ratio of linked static gen",
                               tex_name=r'\gamma_P'
                               )
        self.gammaq = NumParam(default=1.0,
                               info="Q ratio of linked static gen",
                               tex_name=r'\gamma_Q'
                               )


class VSGOuterPIData:
    """
    Outer loop PI controller for d- and q-axis voltages.
    """

    def __init__(self) -> None:
        self.Kpvd = NumParam(default=0.5, tex_name=r'kp_{vd}',
                             info='vd controller proportional gain',
                             unit='p.u.',
                             power=True,
                             )
        self.Kivd = NumParam(default=0.02, tex_name=r'ki_{vd}',
                             info='vd controller integral gain',
                             unit='p.u.',
                             power=True,
                             )
        self.Kpvq = NumParam(default=0.5, tex_name=r'kp_{vq}',
                             info='vq controller proportional gain',
                             unit='p.u.',
                             power=True,
                             )
        self.Kivq = NumParam(default=0.02, tex_name=r'ki_{vq}',
                             info='vq controller integral gain',
                             unit='p.u.',
                             power=True,
                             )


class VSGInnerPIData:
    """
    Inner loop PI controller for d- and q-axis currents.
    """

    def __init__(self):
        self.KpId = NumParam(default=0.2, tex_name=r'kp_{di}',
                             info='Id controller proportional gain',
                             unit='p.u.',
                             power=True,
                             )
        self.KiId = NumParam(default=0.01, tex_name=r'ki_{di}',
                             info='Id controller integral gain',
                             unit='p.u.',
                             power=True,
                             )
        self.KpIq = NumParam(default=0.2, tex_name=r'kp_{qi}',
                             info='Iq controller proportional gain',
                             unit='p.u.',
                             power=True,
                             )
        self.KiIq = NumParam(default=0.01, tex_name=r'ki_{qi}',
                             info='Iq controller integral gain',
                             unit='p.u.',
                             power=True,
                             )


class REGCV1ModelBase(Model):
    """
    Common variables and services for VSG models.
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

        self.p0s = ExtService(model='StaticGen',
                              src='p',
                              indexer=self.gen,
                              tex_name=r'P_{0s}',
                              info='total P of the static gen',
                              )
        self.q0s = ExtService(model='StaticGen',
                              src='q',
                              indexer=self.gen,
                              tex_name=r'Q_{0s}',
                              info='total Q of the static gen',
                              )
        self.Pref = ConstService(v_str='gammap * p0s',
                                 tex_name='P_{ref}',
                                 info='Initial P for the REGCV1 device',
                                 )
        self.Qref = ConstService(v_str='gammaq * q0s',
                                 tex_name='Q_{ref}',
                                 info='Initial Q for the REGCV1 device',
                                 )

        self.vref = ExtService(model='StaticGen',
                               src='v',
                               indexer=self.gen,
                               tex_name=r'V_{ref}',
                               info='initial v of the static gen',
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

        self.Pref2 = Algeb(tex_name=r'P_{ref2}',
                           info='active power reference after adjusting by frequency',
                           e_str='u * Pref - dw * kw - Pref2',
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

        self.Pe = Algeb(tex_name='P_e',
                        info='active power injection from VSC',
                        e_str='vd * Id + vq * Iq - Pe',
                        v_str='Pref')
        self.Qe = Algeb(tex_name='Q_e',
                        info='reactive power injection from VSC',
                        e_str='- vd * Iq + vq * Id - Qe',
                        v_str='Qref')

        self.Id = Algeb(tex_name='I_d',
                        info='d-axis current',
                        v_str='Id0',
                        diag_eps=True,
                        )
        self.Iq = Algeb(tex_name='I_q',
                        info='q-axis current',
                        v_str='Iq0',
                        diag_eps=True,
                        )

    def v_numeric(self, **kwargs):
        """
        Disable the corresponding `StaticGen`s.
        """
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class VSGOuterPIModel:
    """
    Outer PI controllers for REGCV1
    """

    def __init__(self, vderr: str = 'vd-vref2', vqerr: str = 'vq'):
        self.PIvd = PIController(u=vderr,
                                 kp=self.Kpvd,
                                 ki=self.Kivd,
                                 x0='Id0',
                                 )
        self.PIvq = PIController(u=vqerr,
                                 kp=self.Kpvq,
                                 ki=self.Kivq,
                                 x0='Iq0',
                                 )

        self.Idref = AliasAlgeb(self.PIvd_y)
        self.Iqref = AliasAlgeb(self.PIvq_y)


class VSGInnerPIModel:
    """
    Inner current PI controllers for REGCV1
    """

    def __init__(self):
        self.udref0 = ConstService(tex_name=r'u_{dref0}',
                                   v_str='vd0 + ra*Id0 - xs*Iq0'
                                   )
        self.uqref0 = ConstService(tex_name=r'u_{qref0}',
                                   v_str='vq0 + ra*Iq0 + xs*Id0',
                                   )

        # PIvd_y, PIvq_y are Idref, Iqref
        self.PIId = PIController(u='Id - PIvd_y',
                                 kp=self.KpId,
                                 ki=self.KiId,
                                 )
        self.PIIq = PIController(u='Iq - PIvq_y',
                                 kp=self.KpIq,
                                 ki=self.KiIq,
                                 )

        # udLag_y, uqLag_y are ud, uq
        self.Id.e_str = 'vd + ra*Id - xs*Iq - udLag_y'
        self.Iq.e_str = 'vq + ra*Iq + xs*Id - uqLag_y'

        self.udref = Algeb(tex_name=r'u_{dref}',
                           info='ud reference',
                           v_str='udref0',
                           e_str='PIId_y + vd - Iqref * xs - udref',
                           )
        self.uqref = Algeb(tex_name=r'u_{qref}',
                           info='uq reference',
                           v_str='uqref0',
                           e_str='PIIq_y + vq + Idref * xs - uqref',
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


class REGCV1(REGCV1Data, VSGOuterPIData, VSGInnerPIData,
             REGCV1ModelBase, VSGOuterPIModel, VSGInnerPIModel):
    """
    Voltage-controlled VSC with VSG control.

    Includes double-loop PI control and swing equation based VSG control.
    Voltage measurement delays are ignored.

    Notes
    -----
    - Extreme care needs to be taken when coordinating the PI controller
      parameters.
    - Setting the primary frequency control droop ``kw`` can improve
      small-signal stability.
    - The droop ``kv`` for voltage control (pu voltage / pu Q change), if used,
      needs to be chosen carefully. In most cases, ``kv`` should be a very small
      positive value if not zero.

    """

    def __init__(self, system, config):
        REGCV1Data.__init__(self)
        VSGOuterPIData.__init__(self)
        VSGInnerPIData.__init__(self)

        REGCV1ModelBase.__init__(self, system, config)
        VSGOuterPIModel.__init__(self)
        VSGInnerPIModel.__init__(self)
