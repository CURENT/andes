"""
Synchronous generator classes
"""
import logging
from andes.core.model import Model, ModelData
from andes.core.param import IdxParam, NumParam, ExtParam
from andes.core.var import Algeb, State, ExtAlgeb
from andes.core.discrete import LessThan
from andes.core.service import ConstService, ExtService  # NOQA

logger = logging.getLogger(__name__)


class GENBaseData(ModelData):
    def __init__(self):
        super().__init__()
        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            mandatory=True,
                            )
        self.coi = IdxParam(model='COI',
                            info="center of inertia index",
                            )
        self.Sn = NumParam(default=100.0,
                           info="Power rating",
                           tex_name='S_n',
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
        self.xq = NumParam(default=1.7,
                           info="q-axis synchronous reactance",
                           z=True,
                           tex_name='x_q'
                           )
        # NOTE: assume `xd1 = xq` for GENCLS, TODO: replace xq with xd1

        self.kp = NumParam(default=0,
                           info="active power feedback gain",
                           tex_name='k_p'
                           )
        self.kw = NumParam(default=0,
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
                            non_zero=True
                            )


class GENBase(Model):
    def __init__(self, system, config):
        super().__init__(system, config)
        self.group = 'SynGen'
        self.flags.update({'tds': True,
                           'nr_iter': False,
                           })

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
                           e_str='(u / M) * (tm - te - D * (omega - 1))')

        # network algebraic variables
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          e_str='-u * (vd * Id + vq * Iq)'
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          e_str='-u * (vq * Id - vd * Iq)'
                          )

        # algebraic variables
        # Need to be provided by specific generator models
        self.Id = Algeb(info='d-axis current',
                        v_str='Id0',
                        tex_name=r'I_d',
                        e_str=''
                        )  # to be completed by subclasses
        self.Iq = Algeb(info='q-axis current',
                        v_str='Iq0',
                        tex_name=r'I_q',
                        e_str=''
                        )  # to be completed

        self.vd = Algeb(info='d-axis voltage',
                        v_str='vd0',
                        e_str='v * sin(delta - a) - vd',
                        tex_name=r'V_d',
                        )
        self.vq = Algeb(info='q-axis voltage',
                        v_str='vq0',
                        e_str='v * cos(delta - a) - vq',
                        tex_name=r'V_q',
                        )

        self.tm = Algeb(info='mechanical torque',
                        tex_name=r'\tau_m',
                        v_str='tm0',
                        v_setter=True,
                        e_str='tm0 - tm'
                        )
        self.te = Algeb(info='electric torque',
                        tex_name=r'\tau_e',
                        v_str='p0',
                        v_setter=True,
                        e_str='psid * Iq - psiq * Id - te',
                        )
        self.vf = Algeb(info='excitation voltage',
                        unit='pu',
                        v_str='vf0',
                        v_setter=True,
                        e_str='vf0 - vf',
                        tex_name=r'v_f'
                        )

        self.subidx = ExtParam(model='StaticGen',
                               src='subidx',
                               indexer=self.gen,
                               tex_name='idx_{sub}',
                               export=False,
                               )
        # ----------service consts for initialization----------
        self.p0 = ExtService(model='StaticGen',
                             src='p',
                             indexer=self.gen,
                             tex_name='P_0',
                             )
        self.q0 = ExtService(model='StaticGen',
                             src='q',
                             indexer=self.gen,
                             tex_name='Q_0',
                             )

    def v_numeric(self, **kwargs):
        # disable corresponding `StaticGen`
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class Flux0(object):
    """
    Flux model without electro-magnetic transients and ignore speed deviation
    """

    def __init__(self):
        self.psid = Algeb(info='d-axis flux',
                          tex_name=r'\psi_d',
                          v_str='psid0',
                          e_str='u * (ra * Iq + vq) - psid',
                          )
        self.psiq = Algeb(info='q-axis flux',
                          tex_name=r'\psi_q',
                          v_str='psiq0',
                          e_str='u * (ra * Id + vd) + psiq',
                          )

        self.Id.e_str += '+ psid'
        self.Iq.e_str += '+ psiq'


class Flux1(object):
    """
    Flux model without electro-magnetic transients but consider speed deviation
    """

    def __init__(self):
        self.psid = Algeb(info='d-axis flux',
                          tex_name=r'\psi_d',
                          v_str='psid0',
                          e_str='u * (ra * Iq + vq) - omega * psid',
                          )
        self.psiq = Algeb(info='q-axis flux',
                          tex_name=r'\psi_q',
                          v_str='psiq0',
                          e_str='u * (ra * Id + vd) + omega * psiq',
                          )

        self.Id.e_str += '+ psid'
        self.Iq.e_str += '+ psiq'


class Flux2(object):
    """
    Flux model with electro-magnetic transients
    """

    def __init__(self):
        self.psid = State(info='d-axis flux',
                          tex_name=r'\psi_d',
                          v_str='psid0',
                          e_str='u * 2 * pi * fn * (ra * Id + vd + omega * psiq)',
                          )
        self.psiq = State(info='q-axis flux',
                          tex_name=r'\psi_q',
                          v_str='psiq0',
                          e_str='u * 2 * pi * fn * (ra * Iq + vq - omega * psid)',
                          )

        self.Id.e_str += '+ psid'
        self.Iq.e_str += '+ psiq'


class GENCLSModel(object):
    def __init__(self):
        # internal voltage and rotor angle calculation
        self._V = ConstService(v_str='v * exp(1j * a)',
                               tex_name='V_c',
                               )
        self._S = ConstService(v_str='p0 - 1j * q0',
                               tex_name='S',
                               )
        self._I = ConstService(v_str='_S / conj(_V)',
                               tex_name='I_c',
                               )
        self._E = ConstService(tex_name='E')
        self._deltac = ConstService(tex_name=r'\delta_c')
        self.delta0 = ConstService(tex_name=r'\delta_0')

        self.vdq = ConstService(v_str='u * (_V * exp(1j * 0.5 * pi - _deltac))',
                                tex_name='V_{dq}')
        self.Idq = ConstService(v_str='u * (_I * exp(1j * 0.5 * pi - _deltac))',
                                tex_name='I_{dq}')

        self.Id0 = ConstService(v_str='re(Idq)',
                                tex_name=r'I_{d0}')
        self.Iq0 = ConstService(v_str='im(Idq)',
                                tex_name=r'I_{q0}')
        self.vd0 = ConstService(v_str='re(vdq)',
                                tex_name=r'V_{d0}')
        self.vq0 = ConstService(v_str='im(vdq)',
                                tex_name=r'V_{q0}')

        self.tm0 = ConstService(tex_name=r'\tau_{m0}',
                                v_str='u * ((vq0 + ra * Iq0) * Iq0 + (vd0 + ra * Id0) * Id0)')
        self.psid0 = ConstService(tex_name=r"\psi_{d0}",
                                  v_str='u * (ra * Iq0) + vq0')
        self.psiq0 = ConstService(tex_name=r"\psi_{q0}",
                                  v_str='-u * (ra * Id0) - vd0')
        self.vf0 = ConstService(tex_name=r'v_{f0}')

        # initialization of internal voltage and delta
        self._E.v_str = '_V + _I * (ra + 1j * xq)'
        self._deltac.v_str = 'log(_E / abs(_E))'
        self.delta0.v_str = 'u * im(_deltac)'

        self.Id.e_str += '+ xq * Id - vf'
        self.Iq.e_str += '+ xq * Iq'
        self.vf0.v_str = '(vq0 + ra * Iq0) + xq * Id0'


class GENCLS(GENBaseData, GENBase, GENCLSModel, Flux0):
    def __init__(self, system, config):
        GENBaseData.__init__(self)
        GENBase.__init__(self, system, config)
        GENCLSModel.__init__(self)
        Flux0.__init__(self)


class GENROUData(GENBaseData):
    def __init__(self):
        super().__init__()
        self.xd = NumParam(default=1.9, info='d-axis synchronous reactance',
                           tex_name=r'x_d', z=True)
        self.xd1 = NumParam(default=0.302, info='d-axis transient reactance',
                            tex_name=r"x \prime_d", z=True)
        self.xd2 = NumParam(default=0.204, info='d-axis sub-transient reactance',
                            tex_name=r"x \prime \prime_d", z=True)

        self.xq1 = NumParam(default=0.5, info='q-axis transient reactance',
                            tex_name=r"x \prime_q", z=True)
        self.xq2 = NumParam(default=0.3, info='q-axis sub-transient reactance',
                            tex_name=r"x \prime \prime_q", z=True)

        self.Td10 = NumParam(default=8.0, info='d-axis transient time constant',
                             tex_name=r"T \prime_{d0}")
        self.Td20 = NumParam(default=0.04, info='d-axis sub-transient time constant',
                             tex_name=r"T \prime \prime_{d0}")
        self.Tq10 = NumParam(default=0.8, info='q-axis transient time constant',
                             tex_name=r"T \prime_{q0}")
        self.Tq20 = NumParam(default=0.02, info='q-axis sub-transient time constant',
                             tex_name=r"T \prime \prime_{q0}")


class GENROUModel(object):
    def __init__(self):
        self.gd1 = ConstService(v_str='(xd2 - xl) / (xd1 - xl)',
                                tex_name=r"\gamma_{d1}")
        self.gq1 = ConstService(v_str='(xq2 - xl) / (xq1 - xl)',
                                tex_name=r"\gamma_{q1}")
        self.gd2 = ConstService(v_str='(xd1 - xd2) / (xd1 - xl) ** 2',
                                tex_name=r"\gamma_{d2}")
        self.gq2 = ConstService(v_str='(xq1 - xq2) / (xq1 - xl) ** 2',
                                tex_name=r"\gamma_{q2}")
        self.gqd = ConstService(v_str='(xq - xl) / (xd - xl)',
                                tex_name=r"\gamma_{qd}")

        # Saturation services
        # when S10 = 0, S12 = 1, Saturation is disabled. Thus, Sat = 0, A = 1, B = 0
        self.Sat = ConstService(v_str='sqrt((S10 * 1) / (S12 * 1.2))',
                                tex_name=r"S_{at}")
        self.SA = ConstService(v_str='1.2 + 0.2 / (Sat - 1)',
                               tex_name='S_A')
        self.SB = ConstService(v_str='((Sat < 0) + (Sat > 0)) * 1.2 * S12 * ((Sat - 1) / 0.2) ** 2',
                               tex_name='S_B')

        # internal voltage and rotor angle calculation

        # Initialization reference: OpenIPSL at
        #   https://github.com/OpenIPSL/OpenIPSL/blob/master/OpenIPSL/Electrical/Machines/PSSE/GENROU.mo

        self._V = ConstService(v_str='v * exp(1j * a)', tex_name='V_c', info='complex terminal voltage')
        self._S = ConstService(v_str='p0 - 1j * q0', tex_name='S', info='complex terminal power')
        self._Zs = ConstService(v_str='ra + 1j * xd2', tex_name='Z_s', info='equivalent impedance')
        self._It = ConstService(v_str='_S / conj(_V)', tex_name='I_t', info='complex terminal current')
        self._Is = ConstService(tex_name='I_s', v_str='_It + _V / _Zs', info='equivalent current source')

        self.psia0 = ConstService(tex_name=r"\psi_{a0}", v_str='_Is * _Zs',
                                  info='subtransient flux linkage in stator reference')
        self.psia0_arg = ConstService(tex_name=r"\theta_{\psi a0}", v_str='arg(psia0)')
        self.psia0_abs = ConstService(tex_name=r"|\psi_{a0}|", v_str='abs(psia0)')
        self._It_arg = ConstService(tex_name=r"\theta_{It0}", v_str='arg(_It)')
        self._psia0_It_arg = ConstService(tex_name=r"\theta_{\psi a It}",
                                          v_str='psia0_arg - _It_arg')

        self.Se0 = ConstService(tex_name=r"S_{e0}",
                                v_str='(psia0_abs >= SA) * (psia0_abs - SA) ** 2 * SB / psia0_abs')

        self._a = ConstService(tex_name=r"a", v_str='psia0_abs + psia0_abs * Se0 * gqd')
        self._b = ConstService(tex_name=r"b", v_str='abs(_It) * (xq2 - xq)')  # xd2 == xq2

        self.delta0 = ConstService(tex_name=r'\delta_0',
                                   v_str='atan(_b * cos(_psia0_It_arg) / (_b * sin(_psia0_It_arg) - _a)) + '
                                         'psia0_arg')
        self._Tdq = ConstService(tex_name=r"T_{dq}",
                                 v_str='cos(delta0) - 1j * sin(delta0)')
        self.psia0_dq = ConstService(tex_name=r"\psi_{a0,dq}",
                                     v_str='psia0 * _Tdq')
        self.It_dq = ConstService(tex_name=r"I_{t,dq}",
                                  v_str='conj(_It * _Tdq)')

        self.psiad0 = ConstService(tex_name=r"\psi_{ad0}",
                                   v_str='re(psia0_dq)')
        self.psiaq0 = ConstService(tex_name=r"\psi_{aq0}",
                                   v_str='im(psia0_dq)')

        self.Id0 = ConstService(v_str='im(It_dq)', tex_name=r'I_{d0}')
        self.Iq0 = ConstService(v_str='re(It_dq)', tex_name=r'I_{q0}')

        self.vd0 = ConstService(v_str='-(psiaq0 - xq2*Iq0) - ra * Id0', tex_name=r'V_{d0}')
        self.vq0 = ConstService(v_str='psiad0 - xd2*Id0 - ra*Iq0', tex_name=r'V_{q0}')

        self.tm0 = ConstService(tex_name=r'\tau_{m0}',
                                v_str='u * ((vq0 + ra * Iq0) * Iq0 + (vd0 + ra * Id0) * Id0)')

        self.vf0 = ConstService(tex_name=r'v_{f0}', v_str='(Se0 + 1)*psiad0 + (xd - xd2) * Id0')
        self.psid0 = ConstService(tex_name=r"\psi_{d0}",
                                  v_str='u * (ra * Iq0) + vq0')
        self.psiq0 = ConstService(tex_name=r"\psi_{q0}",
                                  v_str='-u * (ra * Id0) - vd0')

        # initialization of internal voltage and delta
        self.e1q0 = ConstService(tex_name="e'_{q0}",
                                 v_str='Id0*(-xd + xd1) - Se0*psiad0 + vf0')

        self.e1d0 = ConstService(tex_name="e'_{d0}",
                                 v_str='Iq0*(xq - xq1) + Se0*gqd*psiaq0')

        self.e2d0 = ConstService(tex_name="e''_{d0}",
                                 v_str='Id0*(xl - xd) - Se0*psiad0 + vf0')
        self.e2q0 = ConstService(tex_name="e''_{q0}",
                                 v_str='Iq0*(xl - xq) - Se0*gqd*psiaq0')

        # begin variables and equations
        self.psiaq = Algeb(tex_name=r"\psi_{aq}", info='q-axis air gap flux',
                           v_str='psiaq0',
                           e_str='psiq + xq2 * Iq - psiaq')

        self.psiad = Algeb(tex_name=r"\psi_{ad}", info='d-axis air gap flux',
                           v_str='psiad0',
                           e_str='-psiad + gd1 * e1q + gd2 * (xd1 - xl) * e2d')

        self.psia = Algeb(tex_name=r"\psi_{a}", info='air gap flux magnitude',
                          v_str='abs(psia0_dq)',
                          e_str='sqrt(psiad **2 + psiaq ** 2) - psia')

        self.Slt = LessThan(u=self.psia, bound=self.SA, equal=False, enable=True)

        self.Se = Algeb(tex_name=r"S_e(|\psi_{a}|)", info='saturation output',
                        v_str='Se0',
                        e_str='Slt_z0 * (psia - SA) ** 2 * SB / psia - Se')

        self.e1q = State(info='q-axis transient voltage',
                         tex_name=r"e'_q",
                         v_str='e1q0',
                         e_str='(-e1q - (xd - xd1) * (Id - gd2 * e2d - (1 - gd1) * Id + gd2 * e1q) - '
                               'Se * psiad + vf) / Td10')
        self.e1d = State(info='d-axis transient voltage',
                         tex_name=r"e'_d",
                         v_str='e1d0',
                         e_str='(-e1d + (xq - xq1) * (Iq - gq2 * e2q - (1 - gq1) * Iq - gq2 * e1d) + '
                               'Se * gqd * psiaq) / Tq10')

        self.e2d = State(info='d-axis sub-transient voltage',
                         tex_name=r"e''_d",
                         v_str='e2d0',
                         e_str='(-e2d + e1q - (xd1 - xl) * Id) / Td20')

        self.e2q = State(info='q-axis sub-transient voltage',
                         tex_name=r"e''_q",
                         v_str='e2q0',
                         e_str='(-e2q - e1d - (xq1 - xl) * Iq) / Tq20')

        self.Id.e_str += '+ xd2 * Id - gd1 * e1q - (1 - gd1) * e2d'
        self.Iq.e_str += '+ xq2 * Iq + gq1 * e1d - (1 - gq1) * e2q'


class GENROU(GENROUData, GENBase, GENROUModel, Flux0):
    """
    Round rotor generator with quadratic saturation
    """
    def __init__(self, system, config):
        GENROUData.__init__(self)
        GENBase.__init__(self, system, config)
        Flux0.__init__(self)
        GENROUModel.__init__(self)
