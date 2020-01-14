"""
Synchronous generator classes
"""
import logging
from andes.core.model import Model, ModelData  # NOQA
from andes.core.param import IdxParam, NumParam, ExtParam  # NOQA
from andes.core.var import Algeb, State, ExtAlgeb  # NOQA
from andes.core.discrete import Selector  # NOQA
from andes.core.service import ConstService, ExtService  # NOQA
from andes.core.block import MagneticQuadSat, MagneticExpSat  # NOQA
from andes.shared import np  # NOQA

logger = logging.getLogger(__name__)


class GENBaseData(ModelData):
    def __init__(self):
        super().__init__()
        self.bus = IdxParam(model='Bus', info="interface bus id", mandatory=True)
        self.gen = IdxParam(info="static generator index", mandatory=True)
        self.coi = IdxParam(model='COI', info="center of inertia index")

        self.Sn = NumParam(default=100.0, info="Power rating", tex_name='S_n')
        self.Vn = NumParam(default=110.0, info="AC voltage rating", tex_name='V_n')
        self.fn = NumParam(default=60.0, info="rated frequency", tex_name='f')

        self.D = NumParam(default=0.0, info="Damping coefficient", power=True, tex_name='D')
        self.M = NumParam(default=6, info="machine start up time (2H)", non_zero=True, power=True,
                          tex_name='M')
        self.ra = NumParam(default=0.0, info="armature resistance", z=True, tex_name='r_a')
        self.xl = NumParam(default=0.0, info="leakage reactance", z=True, tex_name='x_l')
        self.xq = NumParam(default=1.7, info="q-axis synchronous reactance", z=True, tex_name='x_q')
        # NOTE: assume `xd1 = xq` for GENCLS, TODO: replace xq with xd1

        self.kp = NumParam(default=0, info="active power feedback gain", tex_name='k_p')
        self.kw = NumParam(default=0, info="speed feedback gain", tex_name='k_w')
        self.S10 = NumParam(default=1, info="first saturation factor", tex_name='S_{1.0}')
        self.S12 = NumParam(default=1, info="second saturation factor", tex_name='S_{1.2}')


class GENBase(Model):
    def __init__(self, system, config):
        super().__init__(system, config)
        self.group = 'SynGen'
        self.flags.update({'tds': True})

        # state variables
        self.delta = State(v_str='delta0', tex_name=r'\delta',
                           e_str='u * fn * (omega - 1)')
        self.omega = State(v_str='u', tex_name=r'\omega',
                           e_str='(u / M ) * (tm - te - D * (omega - 1))')

        # network algebraic variables
        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          e_str='-u * (vd * Id + vq * Iq)')
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name=r'V',
                          info='Bus voltage magnitude',
                          e_str='-u * (vq * Id - vd * Iq)')

        # algebraic variables
        # Need to be provided by specific generator models
        self.Id = Algeb(v_str='Id0', tex_name=r'I_d')  # to be completed by subclasses
        self.Iq = Algeb(v_str='Iq0', tex_name=r'I_q')  # to be completed

        self.vd = Algeb(v_str='vd0', e_str='v * sin(delta - a) - vd', tex_name=r'V_d')
        self.vq = Algeb(v_str='vq0', e_str='v * cos(delta - a) - vq', tex_name=r'V_q')

        self.tm = Algeb(info='mechanical torque', tex_name=r'\tau_m',
                        v_str='tm0', v_setter=True, e_str='tm0 - tm')
        self.te = Algeb(info='electric torque', tex_name=r'\tau_e',
                        v_str='p0', v_setter=True, e_str='psid * Iq - psiq * Id - te', )
        self.vf = Algeb(v_str='vf0', v_setter=True, e_str='vf0 - vf', tex_name=r'v_f')

        # ----------service consts for initialization----------
        self.p0 = ExtService(model='StaticGen', src='p', indexer=self.gen, tex_name='P_0')
        self.q0 = ExtService(model='StaticGen', src='q', indexer=self.gen, tex_name='Q_0')

        # internal voltage and rotor angle calculation
        self._V = ConstService(v_str='v * exp(1j * a)', tex_name='V_c')
        self._S = ConstService(v_str='p0 - 1j * q0', tex_name='S')
        self._I = ConstService(v_str='_S / conj(_V)', tex_name='I_c')
        self._E = ConstService(v_str='_V + _I * (ra + 1j * xq)', tex_name='E')
        self._deltac = ConstService(v_str='log(_E / abs(_E))', tex_name=r'\delta_c')
        self.delta0 = ConstService(v_str='u * im(_deltac)', tex_name=r'\delta_0')

        self.vdq = ConstService(v_str='u * (_V * exp(1j * 0.5 * pi - _deltac))', tex_name='V_{dq}')
        self.Idq = ConstService(v_str='u * (_I * exp(1j * 0.5 * pi - _deltac))', tex_name='I_{dq}')

        self.Id0 = ConstService(v_str='re(Idq)', tex_name=r'I_{d0}')
        self.Iq0 = ConstService(v_str='im(Idq)', tex_name=r'I_{q0}')
        self.vd0 = ConstService(v_str='re(vdq)', tex_name=r'V_{d0}')
        self.vq0 = ConstService(v_str='im(vdq)', tex_name=r'V_{q0}')

        self.tm0 = ConstService(tex_name=r'\tau_{m0}',
                                v_str='u * ((vq0 + ra * Iq0) * Iq0 + (vd0 + ra * Id0) * Id0)')
        self.vf0 = ConstService(tex_name=r'v_{f0}')

        # --------------------------------------------------Experimental-----
        # self.Idq_max = Algeb(v_str='maximum(Id, Iq)', diag_eps=1e-6,
        #                      e_str='Idqs_s0 * Id + Idqs_s1 * Iq - Idq_max',
        #                      tex_name='I_{dq_{max}}')
        #
        # self.Idqs = Selector(self.Id, self.Iq, fun=np.maximum.reduce, tex_name=r'I_{dq,max}')
        # self.sat = MagneticQuadSat(self.vd, self.S10, self.S12, tex_name='{sat}')

    def v_numeric(self, **kwargs):
        # disable corresponding `StaticGen`
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class Flux0(object):
    """
    Flux model without electro-magnetic transients and ignore speed deviation
    """
    def __init__(self):
        self.psid = Algeb(tex_name=r'\psi_d', v_str='u * (ra * Iq0) + vq0',
                          e_str='u * (ra * Iq + vq) - psid')
        self.psiq = Algeb(tex_name=r'\psi_q', v_str='-u * (ra * Id0) - vd0',
                          e_str='u * (ra * Id + vd) + psiq')

        self.Id.e_str += ' + psid'
        self.Iq.e_str += ' + psiq'


class Flux1(object):
    """
    Flux model without electro-magnetic transients but consider speed deviation
    """
    def __init__(self):
        self.psid = Algeb(tex_name=r'\psi_d', v_str='u * (ra * Iq0) + vq0',
                          e_str='u * (ra * Iq + vq) - omega * psid')
        self.psiq = Algeb(tex_name=r'\psi_q', v_str='-u * (ra * Id0) - vd0',
                          e_str='u * (ra * Id + vd) + omega * psiq')

        self.Id.e_str += ' + psid'
        self.Iq.e_str += ' + psiq'


class Flux2(object):
    """
    Flux model with electro-magnetic transients
    """
    def __init__(self):
        self.psid = State(tex_name=r'\psi_d', v_str='u * (ra * Iq0) + vq0',
                          e_str='u * 2 * pi * fn * (ra * Id + vd + omega * psiq)')
        self.psiq = State(tex_name=r'\psi_q', v_str='-u * (ra * Id0) - vd0',
                          e_str='u * 2 * pi * fn * (ra * Iq + vq - omega * psid)')

        self.Id.e_str += ' + psid'
        self.Iq.e_str += ' + psiq'


class GENCLSModel(object):
    def __init__(self):
        self.Id.e_str = 'xq * Id - vf'
        self.Iq.e_str = 'xq * Iq'
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
                            tex_name=r"x'_d", z=True)
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
        self.Taa = NumParam(default=0.0, info='d-axis additional leakage time constant',
                            tex_name=r"T_{aa}")


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

        self.e1d = State(tex_name=r"e'_d",
                         e_str='(-e1d + (xq - xq1) * (Iq - gq2 * psi2q - (1 - gq1) * Iq - gd2 * e1d)) / Tq10')

        self.e1q = State(tex_name=r"e'_q",
                         e_str='(-e1q - (xd - xd1) * (Id - gd2 * psi2d - (1 - gd1) * Id + gd2 * e1q) + vf) / Td10')

        self.psi2d = State(tex_name=r"\psi''_d",
                           e_str='(-psi2d + e1q - (xd1 - xl) * Id) / Td20')
        self.psi2q = State(tex_name=r"\psi''_q",
                           e_str='(-psi2q - e2d - (xq1 - xl) * Iq) / Tq20')

        self.Id.e_str = 'xd2 * Id - gd1 * e1q - (1 - gd1) * psi2d'
        self.Iq.e_str = 'xq2 * Iq + gq1 * e1d - (1 - gq1) * psi2q'


class GENROU(GENROUData, GENBase, GENROUModel, Flux0):
    def __init__(self, system, config):
        GENROUData.__init__(self)
        GENBase.__init__(self, system, config)
        GENROUModel.__init__(self)
        Flux0.__init__(self)
