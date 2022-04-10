import logging

from andes.core import (ModelData, IdxParam, NumParam, Model, State,
                        ExtAlgeb, Algeb, ExtParam, ExtService, ConstService)
from andes.core.service import InitChecker

logger = logging.getLogger(__name__)


class GENBaseData(ModelData):
    def __init__(self):
        super().__init__()
        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            model='StaticGen',
                            mandatory=True,
                            )
        self.coi = IdxParam(model='COI',
                            info="center of inertia index",
                            )
        self.coi2 = IdxParam(model='COI2',
                             info="center of inertia index",
                             )
        self.Sn = NumParam(default=100.0,
                           info="Power rating",
                           tex_name='S_n',
                           unit='MVA',
                           )
        self.Vn = NumParam(default=110.0,
                           info="AC voltage rating",
                           tex_name='V_n',
                           )
        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f',
                           )

        self.D = NumParam(default=0.0,
                          info="Damping coefficient",
                          power=True,
                          tex_name='D'
                          )
        self.M = NumParam(default=6,
                          info="machine start up time (2H)",
                          non_zero=True,
                          non_negative=True,
                          power=True,
                          tex_name='M'
                          )
        self.ra = NumParam(default=0.0,
                           info="armature resistance",
                           z=True,
                           tex_name='r_a'
                           )
        self.xl = NumParam(default=0.0,
                           info="leakage reactance",
                           z=True,
                           tex_name='x_l'
                           )
        self.xd1 = NumParam(default=0.302,
                            info='d-axis transient reactance',
                            tex_name=r"x'_d", z=True)

        self.kp = NumParam(default=0.0,
                           info="active power feedback gain",
                           tex_name='k_p'
                           )
        self.kw = NumParam(default=0.0,
                           info="speed feedback gain",
                           tex_name='k_w'
                           )
        self.S10 = NumParam(default=0.0,
                            info="first saturation factor",
                            tex_name='S_{1.0}'
                            )
        self.S12 = NumParam(default=1.0,
                            info="second saturation factor",
                            tex_name='S_{1.2}',
                            )
        self.gammap = NumParam(default=1.0,
                               info="P ratio of linked static gen",
                               tex_name=r'\gamma_P'
                               )
        self.gammaq = NumParam(default=1.0,
                               info="Q ratio of linked static gen",
                               tex_name=r'\gamma_Q'
                               )


class GENBase(Model):
    """
    Base class for synchronous generators.

    Defines shared network and dq-component variables.
    """

    def __init__(self, system, config):
        super().__init__(system, config)
        self.group = 'SynGen'
        self.flags.update({'tds': True,
                           'nr_iter': False,
                           })
        self.config.add(vf_lower=1.0,
                        vf_upper=5.0,
                        )

        self.config.add_extra("_help",
                              vf_lower="lower limit for vf warning",
                              vf_upper="upper limit for vf warning",
                              )

        # state variables
        self.delta = State(info='rotor angle',
                           unit='rad',
                           v_str='delta0',
                           tex_name=r'\delta',
                           e_str='u * (2 * pi * fn) * (omega - 1)')
        self.omega = State(info='rotor speed',
                           unit='pu (Hz)',
                           v_str='u',
                           tex_name=r'\omega',
                           e_str='u * (tm - te - D * (omega - 1))',
                           t_const=self.M,
                           )

        # network algebraic variables
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          e_str='-u * (vd * Id + vq * Iq)',
                          ename='P',
                          tex_ename='P',
                          is_input=True,
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          e_str='-u * (vq * Id - vd * Iq)',
                          ename='Q',
                          tex_ename='Q',
                          is_input=True,
                          )

        # algebraic variables
        # Need to be provided by specific generator models
        self.Id = Algeb(info='d-axis current',
                        v_str='u * Id0',
                        tex_name=r'I_d',
                        e_str=''
                        )  # to be completed by subclasses
        self.Iq = Algeb(info='q-axis current',
                        v_str='u * Iq0',
                        tex_name=r'I_q',
                        e_str=''
                        )  # to be completed

        self.vd = Algeb(info='d-axis voltage',
                        v_str='u * vd0',
                        e_str='u * v * sin(delta - a) - vd',
                        tex_name=r'V_d',
                        )
        self.vq = Algeb(info='q-axis voltage',
                        v_str='u * vq0',
                        e_str='u * v * cos(delta - a) - vq',
                        tex_name=r'V_q',
                        )

        self.tm = Algeb(info='mechanical torque',
                        tex_name=r'\tau_m',
                        v_str='tm0',
                        e_str='tm0 - tm'
                        )
        self.te = Algeb(info='electric torque',
                        tex_name=r'\tau_e',
                        v_str='u * tm0',
                        e_str='u * (psid * Iq - psiq * Id) - te',
                        )
        self.vf = Algeb(info='excitation voltage',
                        unit='pu',
                        v_str='u * vf0',
                        e_str='u * vf0 - vf',
                        tex_name=r'v_f'
                        )

        self._vfc = InitChecker(u=self.vf,
                                info='(vf range)',
                                lower=self.config.vf_lower,
                                upper=self.config.vf_upper,
                                )

        self.XadIfd = Algeb(tex_name='X_{ad}I_{fd}',
                            info='d-axis armature excitation current',
                            unit='p.u (kV)',
                            v_str='u * vf0',
                            e_str='u * vf0 - XadIfd'
                            )  # e_str to be provided. Not available in GENCLS

        self.subidx = ExtParam(model='StaticGen',
                               src='subidx',
                               indexer=self.gen,
                               export=False,
                               info='Generator idx in plant; only used by PSS/E data',
                               vtype=str,
                               )

        # declaring `Vn_bus` as ExtParam will fail for PSS/E parser
        self.Vn_bus = ExtService(model='Bus',
                                 src='Vn',
                                 indexer=self.bus,
                                 )
        self._vnc = InitChecker(u=self.Vn,
                                info='Vn and Bus Vn',
                                equal=self.Vn_bus,
                                )

        # ----------service consts for initialization----------
        self.p0s = ExtService(model='StaticGen',
                              src='p',
                              indexer=self.gen,
                              tex_name='P_{0s}',
                              info='initial P of the static gen',
                              )
        self.q0s = ExtService(model='StaticGen',
                              src='q',
                              indexer=self.gen,
                              tex_name='Q_{0s}',
                              info='initial Q of the static gen',
                              )
        self.p0 = ConstService(v_str='p0s * gammap',
                               tex_name='P_0',
                               info='initial P of this gen',
                               )
        self.q0 = ConstService(v_str='q0s * gammaq',
                               tex_name='Q_0',
                               info='initial Q of this gen',
                               )

        self.Pe = Algeb(tex_name='P_e',
                        info='active power injection',
                        e_str='u * (vd * Id + vq * Iq) - Pe',
                        v_str='u * (vd0 * Id0 + vq0 * Iq0)')
        self.Qe = Algeb(tex_name='Q_e',
                        info='reactive power injection',
                        e_str='u * (vq * Id - vd * Iq) - Qe',
                        v_str='u * (vq0 * Id0 - vd0 * Iq0)')

    def v_numeric(self, **kwargs):
        """
        Disable static generators with a linked synchronous machine.
        """

        mask_idx = [self.gen.v[i] for i in range(self.n) if self.u.v[i] == 1]
        self.system.groups['StaticGen'].set(src='u', idx=mask_idx, attr='v', value=0)


class Flux0:
    """
    Flux model without electro-magnetic transients and ignore speed deviation
    """

    def __init__(self):
        self.psid = Algeb(info='d-axis flux',
                          tex_name=r'\psi_d',
                          v_str='u * psid0',
                          e_str='u * (ra*Iq + vq) - psid',
                          )
        self.psiq = Algeb(info='q-axis flux',
                          tex_name=r'\psi_q',
                          v_str='u * psiq0',
                          e_str='u * (ra*Id + vd) + psiq',
                          )

        self.Id.e_str += '+ psid'
        self.Iq.e_str += '+ psiq'


class Flux1:
    """
    Flux model without electro-magnetic transients but considers speed deviation.
    """

    def __init__(self):
        self.psid = Algeb(info='d-axis flux',
                          tex_name=r'\psi_d',
                          v_str='u * psid0',
                          e_str='u * (ra * Iq + vq) - omega * psid',
                          )
        self.psiq = Algeb(info='q-axis flux',
                          tex_name=r'\psi_q',
                          v_str='u * psiq0',
                          e_str='u * (ra * Id + vd) + omega * psiq',
                          )

        self.Id.e_str += '+ psid'
        self.Iq.e_str += '+ psiq'


class Flux2:
    """
    Flux model with electro-magnetic transients.
    """

    def __init__(self):
        self.psid = State(info='d-axis flux',
                          tex_name=r'\psi_d',
                          v_str='u * psid0',
                          e_str='u * 2 * pi * fn * (ra * Id + vd + omega * psiq)',
                          )
        self.psiq = State(info='q-axis flux',
                          tex_name=r'\psi_q',
                          v_str='u * psiq0',
                          e_str='u * 2 * pi * fn * (ra * Iq + vq - omega * psid)',
                          )

        self.Id.e_str += '+ psid'
        self.Iq.e_str += '+ psiq'
