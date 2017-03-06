"""Synchronous generator classes"""

from cvxopt import matrix, sparse, spmatrix
from cvxopt import mul, div, log, sin, cos
from .base import ModelBase
from ..consts import *
from ..utils.math import *


class SynBase(ModelBase):
    """Base class for synchronous generators"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'Synchronous'
        self._data.update({'fn': 60.0,
                           'bus': None,
                           'D': 0.0,
                           'M': 6,
                           'ra': 0.0,
                           'xl': 0.0,
                           'xq': 1.7,
                           'gammap': 1.0,
                           'gammaq': 1.0,
                           'coi': None,
                           'gen': None,
                           'kp': 0,
                           'kw': 0,
                           })
        self._params.extend(['D', 'M', 'ra', 'xl', 'xq', 'gammap', 'gammaq', 'coi', 'gen', 'kp', 'kw'])
        self._descr.update({'fn': 'rated frequency',
                            'bus': 'interface bus id',
                            'D': 'rotor damping',
                            'M': 'machine start up time (2H)',
                            'ra': 'armature resistance',
                            'xl': 'leakage reactance',
                            'xq': 'q-axis synchronous reactance',
                            'gammap': 'active power ratio of all generators on this bus',
                            'gammaq': 'reactive power ratio',
                            'coi': 'center of inertia index',
                            'gen': 'static generator index',
                            'kp': 'active power feedback gain',
                            'kw': 'speed feedback gain',
                            })
        self._units.update({'M': 'MWs/MVA',
                            'D': 'pu',
                            'fn': 'Hz',
                            'ra': 'pu',
                            'xd': 'pu',
                            'gammap': 'pu',
                            'gammaq': 'pu',
                            })
        self.calls.update({'init1': True, 'dyngen': True,
                           'gcall': True, 'gycall': True,
                           'fcall': True, 'fxcall': True,
                           'jac0': True,
                           })
        self._ac = {'bus': ['a', 'v']}
        self._states = ['delta', 'omega']
        self._fnamex = ['\\delta', '\\omega']
        self._algebs = ['p', 'q',
                        'pm', 'vf',
                        'Id', 'Iq',
                        'vd', 'vq',
                        ]

        self._powers = ['M', 'D']
        self._z = ['ra', 'xl', 'xq']
        self._zeros = ['M']
        self._mandatory = ['bus', 'gen']
        self._service = ['pm0', 'vf0', 'c1', 'c2', 'c3', 'ss', 'cc',
                         'im']

    def build_service(self):
        """Build service variables"""
        self.iM = div(1, self.M)

    def setup(self):
        super().setup()
        self.build_service()

    def init1(self, dae):
        self.system.rmgen(self.gen)

        p0 = mul(self.u, self.system.Bus.Pg[self.a], self.gammap)
        q0 = mul(self.u, self.system.Bus.Qg[self.a], self.gammaq)
        v0 = mul(self.u, dae.y[self.v])
        theta0 = dae.y[self.a]
        v = polar(v0, theta0)
        S = p0 - q0*1j
        I = div(S, conj(v))
        E = v + mul(self.ra + self.xq*1j, I)

        dae.y[self.p] = p0
        dae.y[self.q] = q0

        delta = log(div(E, abs(E) + 0j))
        dae.x[self.delta] = mul(self.u, delta.imag())
        dae.x[self.omega] = matrix(1.0, (self.n, 1), 'd')

        # d- and q-axis voltages and currents
        vdq = mul(self.u, mul(v, exp(jpi2 - delta)))
        idq = mul(self.u, mul(I, exp(jpi2 - delta)))
        vd = dae.y[self.vd] = vdq.real()
        vq = dae.y[self.vq] = vdq.imag()
        Id = dae.y[self.Id] = idq.real()
        Iq = dae.y[self.Iq] = idq.imag()

        # electro-mechanical torques / powers
        self.pm0 = mul(vq + mul(self.ra, Iq), Iq) + mul(vd + mul(self.ra, Id), Id)
        dae.y[self.pm] = self.pm0

    def gcall(self, dae):
        nzeros = [0] * self.n
        v = mul(self.u, dae.y[self.v])
        vd = dae.y[self.vd]
        vq = dae.y[self.vq]
        Id = dae.y[self.Id]
        Iq = dae.y[self.Iq]
        self.ss = sin(dae.x[self.delta] - dae.y[self.a])
        self.cc = cos(dae.x[self.delta] - dae.y[self.a])

        dae.g -= spmatrix(dae.y[self.p], self.a, nzeros, (dae.m, 1), 'd')
        dae.g -= spmatrix(dae.y[self.q], self.v, nzeros, (dae.m, 1), 'd')
        dae.g -= spmatrix(vd - mul(v, self.ss), self.vd, nzeros, (dae.m, 1), 'd')
        dae.g -= spmatrix(vq - mul(v, self.cc), self.vq, nzeros, (dae.m, 1), 'd')
        dae.g += spmatrix(mul(vd, Id) + mul(vq, Iq) - dae.y[self.p], self.p, nzeros, (dae.m, 1), 'd')
        dae.g += spmatrix(mul(vq, Id) - mul(vd, Iq) - dae.y[self.q], self.q, nzeros, (dae.m, 1), 'd')
        dae.g += spmatrix(dae.y[self.pm] - self.pm0, self.pm, nzeros, (dae.m, 1), 'd')
        dae.g += spmatrix(dae.y[self.vf] - self.vf0, self.vf, nzeros, (dae.m, 1), 'd')

    def saturation(self, e1q):
        """Saturation characteristic function"""
        return e1q









