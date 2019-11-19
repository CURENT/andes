"""
Synchronous generator classes
"""

import logging
from andes.core.model import Model, ModelData  # NOQA
from andes.core.param import IdxParam, DataParam, NumParam, ExtParam  # NOQA
from andes.core.var import Algeb, State, ExtAlgeb  # NOQA
from andes.core.limiter import Comparer, SortedLimiter  # NOQA
from andes.core.service import Service, ExtService  # NOQA
logger = logging.getLogger(__name__)


class GEN2AxisData(ModelData):
    def __init__(self):
        super().__init__()

        self.Sn = NumParam(default=100.0, info="Power rating")
        self.Vn = NumParam(default=110.0, info="AC voltage rating")
        self.fn = NumParam(default=60.0, info="rated frequency")

        self.bus = IdxParam(model='Bus', info="interface bus idx", mandatory=True)
        self.coi = IdxParam(model='COI', info="center of inertia index")
        self.gen = DataParam(info="static generator index", mandatory=True)

        self.D = NumParam(default=0.0, info="Damping coefficient", power=True)
        self.M = NumParam(default=6, info="machine start up time (2xH)", non_zero=True, power=True)
        self.ra = NumParam(default=0.0, info="armature resistance", z=True)
        self.xd1 = NumParam(default=1.9, info="synchronous reactance", z=True)
        self.xq1 = NumParam(default=0.5, info="q-axis transient reactance", mandatory=True, z=True)
        self.xl = NumParam(default=0.0, info="leakage reactance", z=True)
        self.xd = NumParam(default=1.9, info="d-axis synchronous reactance", mandatory=True, z=True)
        self.xq = NumParam(default=1.7, info="q-axis synchronous reactance", z=True)
        self.Td10 = NumParam(default=8.0, info="d-axis transient time constant", mandatory=True)
        self.Tq10 = NumParam(default=0.8, info="q-axis transient time constant", mandatory=True)

        self.gammap = NumParam(default=1.0, info="active power ratio of all generators on this bus")
        self.gammaq = NumParam(default=1.0, info="reactive power ratio")
        self.kp = NumParam(default=0., info="active power feedback gain")
        self.kw = NumParam(default=0., info="speed feedback gain")

        self.S10 = NumParam(default=0., info="first saturation factor")
        self.S12 = NumParam(default=0., info="second saturation factor")


class GEN2Axis(GEN2AxisData, Model):
    def __init__(self, system=None, config=None):
        GEN2AxisData.__init__(self)
        Model.__init__(self, system=system, config=config)

        self.group = 'SynGen'
        self.flags.update({'tds': True})

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus)
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus)

        self.delta = State(v_init='delta0')
        self.omega = State(v_init='1')
        self.vd = Algeb(v_init='vd0')
        self.vq = Algeb(v_init='vq0')
        self.tm = Algeb(v_init='tm0', v_setter=True)
        self.vf = Algeb(v_init='vf0', v_setter=True)

        # NOTE: `Algeb` and `State` variables need to be declared in the initialization order
        self.e1d = State(v_init='vq + ra * Iq + xd1 * Id')
        self.e1q = State(v_init='e1q0')

        # NOTE: assume that one static gen can only correspond to one syn
        # Does not support automatic PV gen combination
        self.p0 = ExtService(model='StaticGen', src='p', indexer=self.gen)
        self.q0 = ExtService(model='StaticGen', src='q', indexer=self.gen)

        self.Vc = Service(v_str='v * exp(1j * a)')
        self.S = Service(v_str='p0 - 1j * q0')
        self.Ic = Service(v_str='S / conj(Vc)')
        self.E = Service(v_str='Vc + Ic * (ra + 1j * xq)')
        self.deltac = Service(v_str='log(E / abs(E))')
        self.delta0 = Service(v_str='u * im(deltac)')

        self.vdq = Service(v_str='u * (Vc * exp(1j * 0.5 * pi - deltac))')
        self.Idq = Service(v_str='u * (Ic * exp(1j * 0.5 * pi - deltac))')
        self.Id = Service(v_str='re(Idq)')
        self.Iq = Service(v_str='im(Idq)')
        self.vd0 = Service(v_str='re(vdq)')
        self.vq0 = Service(v_str='im(vdq)')

        self.tm0 = Service(v_str='(vq0 + ra * Iq) * Iq + (vd0 + ra * Id) * Id')
        self.e1q0 = Service(v_str='vd0 + ra * Id - xq1 * Iq')
        self.vf0 = Service(v_numeric=self._vf0)

        # NOTE: All non-iterative initialization can be completed by using `Service`
        #       for computation and setting the variable initial values to
        #       the corresponding `Service`.
        #       Create as many temp service variables as needed

        # DAE

        # TODO: substitute static generators
        # Method: define a function `set_param` in base class??

    @staticmethod
    def _vf0(e1q0, xd, xd1, Id, **kwargs):
        return e1q0 + (xd - xd1) * Id

    def v_numeric(self, **kwargs):
        # disable corresponding `StaticGen`
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)
