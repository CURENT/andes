"""
Synchronous generator classes
"""
import logging

from andes.core.model import Model, ModelData
from andes.core.param import IdxParam, NumParam, ExtParam
from andes.core.var import Algeb, State, ExtAlgeb
from andes.core.discrete import LessThan
from andes.core.service import ConstService, ExtService
from andes.core.service import InitChecker, FlagValue

from andes.models.exciter import ExcQuadSat

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
                               info='Generator idx in plant; only used by PSS/E data'
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
        # disable corresponding `StaticGen`
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


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


class GENCLSModel:
    def __init__(self):
        # internal voltage and rotor angle calculation
        self.xq = ExtService(model='GENCLS', src='xd1', indexer=self.idx,
                             )
        self._V = ConstService(v_str='v * exp(1j * a)',
                               tex_name='V_c',
                               vtype=complex,
                               )
        self._S = ConstService(v_str='p0 - 1j * q0',
                               tex_name='S',
                               vtype=complex,
                               )
        self._I = ConstService(v_str='_S / conj(_V)',
                               tex_name='I_c',
                               vtype=complex,
                               )
        self._E = ConstService(tex_name='E', vtype=complex)
        self._deltac = ConstService(tex_name=r'\delta_c', vtype=complex)
        self.delta0 = ConstService(tex_name=r'\delta_0')

        self.vdq = ConstService(v_str='u * (_V * exp(1j * 0.5 * pi - _deltac))',
                                tex_name='V_{dq}', vtype=complex)
        self.Idq = ConstService(v_str='u * (_I * exp(1j * 0.5 * pi - _deltac))',
                                tex_name='I_{dq}', vtype=complex)

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
    """
    Classical generator model.
    """

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
        self.xq = NumParam(default=1.7,
                           info="q-axis synchronous reactance",
                           tex_name='x_q',
                           z=True,
                           )
        self.xd2 = NumParam(default=0.204, info='d-axis sub-transient reactance',
                            tex_name=r"x''_d", z=True)

        self.xq1 = NumParam(default=0.5, info='q-axis transient reactance',
                            tex_name=r"x'_q", z=True)
        self.xq2 = NumParam(default=0.3, info='q-axis sub-transient reactance',
                            tex_name=r"x''_q", z=True)

        self.Td10 = NumParam(default=8.0, info='d-axis transient time constant',
                             tex_name=r"T'_{d0}")
        self.Td20 = NumParam(default=0.04, info='d-axis sub-transient time constant',
                             tex_name=r"T''_{d0}")
        self.Tq10 = NumParam(default=0.8, info='q-axis transient time constant',
                             tex_name=r"T'_{q0}")
        self.Tq20 = NumParam(default=0.02, info='q-axis sub-transient time constant',
                             tex_name=r"T''_{q0}")


class GENROUModel:
    def __init__(self):
        # parameter checking for `xl`
        self._xlc = InitChecker(u=self.xl,
                                info='(xl <= xd2)',
                                upper=self.xd2
                                )

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

        # correct S12 to 1.0 if is zero
        self._fS12 = FlagValue(self.S12, value=0)
        self._S12 = ConstService(v_str='S12 + (1-_fS12)',
                                 info='Corrected S12',
                                 tex_name='S_{1.2}'
                                 )
        # Saturation services
        # Note:
        # To disable saturation, set S10 = 0, S12 = 1 so that SAT_B = 0.
        self.SAT = ExcQuadSat(1.0, self.S10, 1.2, self.S12, tex_name='S_{AT}')

        # Initialization reference: OpenIPSL at
        #   https://github.com/OpenIPSL/OpenIPSL/blob/master/OpenIPSL/Electrical/Machines/PSSE/GENROU.mo

        # internal voltage and rotor angle calculation
        self._V = ConstService(v_str='v * exp(1j * a)', tex_name='V_c', info='complex bus voltage',
                               vtype=complex)
        self._S = ConstService(v_str='p0 - 1j * q0', tex_name='S', info='complex terminal power',
                               vtype=complex)
        self._Zs = ConstService(v_str='ra + 1j * xd2', tex_name='Z_s', info='equivalent impedance',
                                vtype=complex)
        self._It = ConstService(v_str='_S / conj(_V)', tex_name='I_t', info='complex terminal current',
                                vtype=complex)
        self._Is = ConstService(tex_name='I_s', v_str='_It + _V / _Zs', info='equivalent current source',
                                vtype=complex)

        self.psi20 = ConstService(tex_name=r"\psi''_0", v_str='_Is * _Zs',
                                  info='sub-transient flux linkage in stator reference',
                                  vtype=complex)
        self.psi20_arg = ConstService(tex_name=r"\theta_{\psi''0}", v_str='arg(psi20)')
        self.psi20_abs = ConstService(tex_name=r"|\psi''_0|", v_str='abs(psi20)')
        self._It_arg = ConstService(tex_name=r"\theta_{It0}", v_str='arg(_It)')
        self._psi20_It_arg = ConstService(tex_name=r"\theta_{\psi a It}",
                                          v_str='psi20_arg - _It_arg')

        self.Se0 = ConstService(tex_name=r"S_{e0}",
                                v_str='Indicator(psi20_abs>=SAT_A) * (psi20_abs - SAT_A) ** 2 * SAT_B / psi20_abs')

        self._a = ConstService(tex_name=r"a'", v_str='psi20_abs * (1 + Se0*gqd)')
        self._b = ConstService(tex_name=r"b'", v_str='abs(_It) * (xq2 - xq)')  # xd2 == xq2

        self.delta0 = ConstService(tex_name=r'\delta_0',
                                   v_str='atan(_b * cos(_psi20_It_arg) / (_b * sin(_psi20_It_arg) - _a)) + '
                                         'psi20_arg')
        self._Tdq = ConstService(tex_name=r"T_{dq}",
                                 v_str='cos(delta0) - 1j * sin(delta0)',
                                 vtype=complex)
        self.psi20_dq = ConstService(tex_name=r"\psi''_{0,dq}",
                                     v_str='psi20 * _Tdq',
                                     vtype=complex)
        self.It_dq = ConstService(tex_name=r"I_{t,dq}",
                                  v_str='conj(_It * _Tdq)',
                                  vtype=complex)

        self.psi2d0 = ConstService(tex_name=r"\psi_{ad0}",
                                   v_str='re(psi20_dq)')
        self.psi2q0 = ConstService(tex_name=r"\psi_{aq0}",
                                   v_str='-im(psi20_dq)')

        self.Id0 = ConstService(v_str='im(It_dq)', tex_name=r'I_{d0}')
        self.Iq0 = ConstService(v_str='re(It_dq)', tex_name=r'I_{q0}')

        self.vd0 = ConstService(v_str='psi2q0 + xq2*Iq0 - ra * Id0', tex_name=r'V_{d0}')
        self.vq0 = ConstService(v_str='psi2d0 - xd2*Id0 - ra*Iq0', tex_name=r'V_{q0}')

        self.tm0 = ConstService(tex_name=r'\tau_{m0}',
                                v_str='u * ((vq0 + ra * Iq0) * Iq0 + (vd0 + ra * Id0) * Id0)')

        # `vf0` is also equal to `vq + xd*Id +ra*Iq + Se*psi2d` from phasor diagram
        self.vf0 = ConstService(tex_name=r'v_{f0}', v_str='(Se0 + 1)*psi2d0 + (xd - xd2) * Id0')
        self.psid0 = ConstService(tex_name=r"\psi_{d0}",
                                  v_str='u * (ra * Iq0) + vq0')
        self.psiq0 = ConstService(tex_name=r"\psi_{q0}",
                                  v_str='-u * (ra * Id0) - vd0')

        # initialization of internal voltage and delta
        self.e1q0 = ConstService(tex_name="e'_{q0}",
                                 v_str='Id0*(-xd + xd1) - Se0*psi2d0 + vf0')

        self.e1d0 = ConstService(tex_name="e'_{d0}",
                                 v_str='Iq0*(xq - xq1) - Se0*gqd*psi2q0')

        self.e2d0 = ConstService(tex_name="e''_{d0}",
                                 v_str='Id0*(xl - xd) - Se0*psi2d0 + vf0')
        self.e2q0 = ConstService(tex_name="e''_{q0}",
                                 v_str='-Iq0*(xl - xq) - Se0*gqd*psi2q0')

        # begin variables and equations
        self.psi2q = Algeb(tex_name=r"\psi_{aq}", info='q-axis air gap flux',
                           v_str='psi2q0',
                           e_str='gq1*e1d + (1-gq1)*e2q - psi2q',
                           )

        self.psi2d = Algeb(tex_name=r"\psi_{ad}", info='d-axis air gap flux',
                           v_str='u * psi2d0',
                           e_str='gd1*e1q + gd2*(xd1-xl)*e2d - psi2d')

        self.psi2 = Algeb(tex_name=r"\psi_a", info='air gap flux magnitude',
                          v_str='u * abs(psi20_dq)',
                          e_str='psi2d **2 + psi2q ** 2 - psi2 ** 2',
                          diag_eps=True,
                          )

        # `LT` is a reserved keyword for SymPy
        self.SL = LessThan(u=self.psi2, bound=self.SAT_A, equal=False, enable=True, cache=False)

        self.Se = Algeb(tex_name=r"S_e(|\psi_{a}|)", info='saturation output',
                        v_str='u * Se0',
                        e_str='SL_z0 * (psi2 - SAT_A) ** 2 * SAT_B - psi2 * Se',
                        diag_eps=True,
                        )

        # separated `XadIfd` from `e1q` using \dot(e1q) = (vf - XadIfd) / Td10
        self.XadIfd.e_str = 'u * (e1q + (xd-xd1) * (gd1*Id - gd2*e2d + gd2*e1q) + Se*psi2d) - XadIfd'

        # `XadI1q` can also be given in `(xq-xq1)*gq2*(e1d-e2q+(xq1-xl)*Iq) + e1d - Iq*(xq-xq1) + Se*psi2q*gqd`
        self.XaqI1q =\
            Algeb(tex_name='X_{aq}I_{1q}',
                  info='q-axis reaction',
                  unit='p.u (kV)',
                  v_str='0',
                  e_str='e1d + (xq-xq1) * (gq2*e1d - gq2*e2q - gq1*Iq) + Se*psi2q*gqd - XaqI1q'
                  )

        self.e1q = State(info='q-axis transient voltage',
                         tex_name=r"e'_q",
                         v_str='u * e1q0',
                         e_str='(-XadIfd + vf)',
                         t_const=self.Td10,
                         )

        self.e1d = State(info='d-axis transient voltage',
                         tex_name=r"e'_d",
                         v_str='e1d0',
                         e_str='-XaqI1q',
                         t_const=self.Tq10,
                         )

        self.e2d = State(info='d-axis sub-transient voltage',
                         tex_name=r"e''_d",
                         v_str='u * e2d0',
                         e_str='(-e2d + e1q - (xd1 - xl) * Id)',
                         t_const=self.Td20,
                         )

        self.e2q = State(info='q-axis sub-transient voltage',
                         tex_name=r"e''_q",
                         v_str='e2q0',
                         e_str='(-e2q + e1d + (xq1 - xl) * Iq)',
                         t_const=self.Tq20,
                         )

        self.Iq.e_str += '+ xq2*Iq + psi2q'

        self.Id.e_str += '+ xd2*Id - psi2d'


class GENROU(GENROUData, GENBase, GENROUModel, Flux0):
    """
    Round rotor generator with quadratic saturation.
    """
    def __init__(self, system, config):
        GENROUData.__init__(self)
        GENBase.__init__(self, system, config)
        Flux0.__init__(self)
        GENROUModel.__init__(self)
