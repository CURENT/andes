"""
Synchronous generator classes
"""

import logging
from andes.core.model import Model, ModelData  # NOQA
from andes.core.param import IdxParam, NumParam, ExtParam  # NOQA
from andes.core.var import Algeb, State, Calc, ExtAlgeb  # NOQA
from andes.core.limiter import Comparer, SortedLimiter  # NOQA
from andes.core.service import ServiceConst, ExtService  # NOQA
logger = logging.getLogger(__name__)


class GENCLSData(ModelData):
    def __init__(self):
        super().__init__()
        self.bus = IdxParam(model='Bus', info="interface bus id", mandatory=True)
        self.gen = IdxParam(info="static generator index", mandatory=True)
        self.coi = IdxParam(model='COI', info="center of inertia index")

        self.Sn = NumParam(default=100.0, info="Power rating")
        self.Vn = NumParam(default=110.0, info="AC voltage rating")
        self.fn = NumParam(default=60.0, info="rated frequency", tex_name='f')

        self.D = NumParam(default=0.0, info="Damping coefficient", power=True)
        self.M = NumParam(default=6, info="machine start up time (2H)", non_zero=True, power=True)
        self.ra = NumParam(default=0.0, info="armature resistance", z=True, tex_name='r_a')
        self.xl = NumParam(default=0.0, info="leakage reactance", z=True, tex_name='x_l')
        self.xq = NumParam(default=1.7, info="q-axis synchronous reactance", z=True, tex_name='x_q')
        # NOTE: assume `xd1 = xq` for GENCLS

        self.kp = NumParam(default=0, info="active power feedback gain", tex_name='k_p')
        self.kw = NumParam(default=0, info="speed feedback gain", tex_name='k_w')
        self.S10 = NumParam(default=0, info="first saturation factor", tex_name='S_{10}')
        self.S12 = NumParam(default=0, info="second saturation factor", tex_name='S_{20}')


class GENBase(Model):
    def __init__(self, system, config):
        super().__init__(system, config)
        self.group = 'SynGen'
        self.flags.update({'tds': True})

        # state variables
        self.delta = State(v_init='delta0', tex_name=r'\delta',
                           e_str='u * fn * (omega - 1)')
        self.omega = State(v_init='u', tex_name=r'\omega',
                           e_str='(u / M ) * (pm - pe - D * (omega - 1))')

        # network algebraic variables
        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta',
                          e_str='-u * (vd * Id + vq * Iq)')
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name=r'V',
                          e_str='-u * (vq * Id - vd * Iq)')

        self.p = Calc(e_str='-u * (vd * Id + vq * Iq)', tex_name='p')
        self.q = Calc(e_str='-u * (vq * Id - vd * Iq)', tex_name='q')

        # algebraic variables
        # Need to be provided by specific generator models
        self.Id = Algeb(v_init='Id0', tex_name=r'I_d')  # to be completed by subclasses
        self.Iq = Algeb(v_init='Iq0', tex_name=r'I_q')  # to be completed
        self.vd = Algeb(v_init='vd0', e_str='v * sin(delta - a) - vd', tex_name=r'v_d')
        self.vq = Algeb(v_init='vq0', e_str='v * cos(delta - a) - vq', tex_name=r'v_q')

        self.pm = Algeb(v_init='pm0', v_setter=True, e_str='pm0 - pm', tex_name=r'p_m')
        self.pe = Algeb(v_init='p0', v_setter=True, e_str='-pe', tex_name=r'p_e')  # to be completed by subclasses
        self.vf = Algeb(v_init='vf0', v_setter=True, e_str='vf0 - vf', tex_name=r'v_f')

        # ----------service consts for initialization----------
        self.p0 = ExtService(model='StaticGen', src='p', indexer=self.gen, tex_name='p_0')
        self.q0 = ExtService(model='StaticGen', src='q', indexer=self.gen, tex_name='q_0')

        # internal voltage and rotor angle calculation
        self._V = ServiceConst(v_str='v * exp(1j * a)', tex_name='V_c')
        self._S = ServiceConst(v_str='p0 - 1j * q0', tex_name='S')
        self._I = ServiceConst(v_str='_S / conj(_V)', tex_name='I_c')
        self._E = ServiceConst(v_str='_V + _I * (ra + 1j * xq)', tex_name='E')
        self._deltac = ServiceConst(v_str='log(_E / abs(_E))', tex_name=r'\delta_c')
        self.delta0 = ServiceConst(v_str='u * im(_deltac)', tex_name=r'\delta_0')

        self.vdq = ServiceConst(v_str='u * (_V * exp(1j * 0.5 * pi - _deltac))', tex_name='v_{dq}')
        self.Idq = ServiceConst(v_str='u * (_I * exp(1j * 0.5 * pi - _deltac))', tex_name='I_{dq}')

        self.Id0 = ServiceConst(v_str='re(Idq)', tex_name=r'I_{d0}')
        self.Iq0 = ServiceConst(v_str='im(Idq)', tex_name=r'I_{q0}')
        self.vd0 = ServiceConst(v_str='re(vdq)', tex_name=r'v_{d0}')
        self.vq0 = ServiceConst(v_str='im(vdq)', tex_name=r'v_{q0}')

        self.pm0 = ServiceConst(v_str='u * ((vq0 + ra * Iq0) * Iq0 + (vd0 + ra * Id0) * Id0)', tex_name=r'p_{m0}')
        self.vf0 = ServiceConst(v_numeric=self._vf0, tex_name=r'v_{f0}')

    @staticmethod
    def _vf0(**kwargs):
        raise NotImplementedError

    def v_numeric(self, **kwargs):
        # disable corresponding `StaticGen`
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class Flux0(object):
    def __init__(self):
        self.psid = Algeb(tex_name=r'\psi_d', v_init='u * (ra * Iq0) + vq0',
                          e_str='u * (ra * Iq + vq) - psid')
        self.psiq = Algeb(tex_name=r'\psi_q', v_init='- u * (ra * Id0) - vd0',
                          e_str='u * (ra * Id + vd) + psiq')

        self.Id.e_str += ' + psid'
        self.Iq.e_str += ' + psiq'
        self.pe.e_str += ' + psid * Iq - psiq * Id'


class GENCLSModel(object):
    def __init__(self):
        self.Id.e_str = 'xq * Id - vf'
        self.Iq.e_str = 'xq * Iq'


class GENCLS(GENCLSData, GENBase, GENCLSModel, Flux0):
    def __init__(self, system, config):
        GENCLSData.__init__(self)
        GENBase.__init__(self, system, config)
        GENCLSModel.__init__(self)
        Flux0.__init__(self)

    @staticmethod
    def _vf0(vq0, ra, Iq0, xq, Id0, **kwargs):
        return vq0 + ra * Iq0 + xq * Id0


# class GEN2AxisData(ModelData):
#     def __init__(self):
#         super().__init__()
#
#         self.Sn = NumParam(default=100.0, info="Power rating")
#         self.Vn = NumParam(default=110.0, info="AC voltage rating")
#         self.fn = NumParam(default=60.0, info="rated frequency")
#
#         self.bus = IdxParam(model='Bus', info="interface bus idx", mandatory=True)
#         self.coi = IdxParam(model='COI', info="center of inertia index")
#         self.gen = DataParam(info="static generator index", mandatory=True)
#
#         self.D = NumParam(default=0.0, info="Damping coefficient", power=True)
#         self.M = NumParam(default=6, info="machine start up time (2xH)", non_zero=True, power=True)
#         self.ra = NumParam(default=0.0, info="armature resistance", z=True)
#         self.xd1 = NumParam(default=1.9, info="synchronous reactance", z=True)
#         self.xq1 = NumParam(default=0.5, info="q-axis transient reactance", mandatory=True, z=True)
#         self.xl = NumParam(default=0.0, info="leakage reactance", z=True)
#         self.xd = NumParam(default=1.9, info="d-axis synchronous reactance", mandatory=True, z=True)
#         self.xq = NumParam(default=1.7, info="q-axis synchronous reactance", z=True)
#         self.Td10 = NumParam(default=8.0, info="d-axis transient time constant", mandatory=True)
#         self.Tq10 = NumParam(default=0.8, info="q-axis transient time constant", mandatory=True)
#
#         self.gammap = NumParam(default=1.0, info="active power ratio of all generators on this bus")
#         self.gammaq = NumParam(default=1.0, info="reactive power ratio")
#         self.kp = NumParam(default=0., info="active power feedback gain")
#         self.kw = NumParam(default=0., info="speed feedback gain")
#
#         self.S10 = NumParam(default=0., info="first saturation factor")
#         self.S12 = NumParam(default=0., info="second saturation factor")
#
#
# class GEN2Axis(GEN2AxisData, Model):
#     def __init__(self, system=None, config=None):
#         GEN2AxisData.__init__(self)
#         Model.__init__(self, system=system, config=config)
#
#         self.group = 'SynGen'
#         self.flags.update({'tds': True})
#
#         self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus)
#         self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus)
#
#         self.delta = State(v_init='delta0')
#         self.omega = State(v_init='1')
#         self.vd = Algeb(v_init='vd0')
#         self.vq = Algeb(v_init='vq0')
#         self.tm = Algeb(v_init='tm0', v_setter=True)
#         self.vf = Algeb(v_init='vf0', v_setter=True)
#
#         # NOTE: `Algeb` and `State` variables need to be declared in the initialization order
#         self.e1d = State(v_init='vq + ra * Iq + xd1 * Id')
#         self.e1q = State(v_init='e1q0')
#
#         # NOTE: assume that one static gen can only correspond to one syn
#         # Does not support automatic PV gen combination
#         self.p0 = ExtService(model='StaticGen', src='p', indexer=self.gen)
#         self.q0 = ExtService(model='StaticGen', src='q', indexer=self.gen)
#
#         self.Vc = ServiceConst(v_str='v * exp(1j * a)')
#         self.S = ServiceConst(v_str='p0 - 1j * q0')
#         self.Ic = ServiceConst(v_str='S / conj(Vc)')
#         self.E = ServiceConst(v_str='Vc + Ic * (ra + 1j * xq)')
#         self.deltac = ServiceConst(v_str='log(E / abs(E))')
#         self.delta0 = ServiceConst(v_str='u * im(deltac)')
#
#         self.vdq = ServiceConst(v_str='u * (Vc * exp(1j * 0.5 * pi - deltac))')
#         self.Idq = ServiceConst(v_str='u * (Ic * exp(1j * 0.5 * pi - deltac))')
#         self.Id = ServiceConst(v_str='re(Idq)')
#         self.Iq = ServiceConst(v_str='im(Idq)')
#         self.vd0 = ServiceConst(v_str='re(vdq)')
#         self.vq0 = ServiceConst(v_str='im(vdq)')
#
#         self.tm0 = ServiceConst(v_str='(vq0 + ra * Iq) * Iq + (vd0 + ra * Id) * Id')
#         self.e1q0 = ServiceConst(v_str='vd0 + ra * Id - xq1 * Iq')
#         self.vf0 = ServiceConst(v_numeric=self._vf0)
#
#         # NOTE: All non-iterative initialization can be completed by using `Service`
#         #       for computation and setting the variable initial values to
#         #       the corresponding `Service`.
#         #       Create as many temp service variables as needed
#
#         # DAE
#
#         # Method: define a function `set_param` in base class??
#
#     @staticmethod
#     def _vf0(e1q0, xd, xd1, Id, **kwargs):
#         return e1q0 + (xd - xd1) * Id
#
#     def v_numeric(self, **kwargs):
#         # disable corresponding `StaticGen`
#         self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)
