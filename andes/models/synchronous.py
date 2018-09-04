"""Synchronous generator classes"""

from cvxopt import matrix, sparse, spmatrix  # NOQA
from cvxopt import mul, div, log, sin, cos  # NOQA
from .base import ModelBase

from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA
from ..consts import jpi2
from ..utils.math import polar, conj, exp


class SynBase(ModelBase):
    """Base class for synchronous generators"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'Synchronous'
        self._data.update({
            'fn': 60.0,
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
            'S10': 0,
            'S12': 0,
        })
        self._params.extend([
            'D',
            'M',
            'ra',
            'xl',
            'xq',
            'gammap',
            'gammaq',
            'kp',
            'kw',
            'S10',
            'S12',
        ])
        self._descr.update({
            'fn':
            'rated frequency',
            'bus':
            'interface bus id',
            'D':
            'Damping coefficient',
            'M':
            'machine start up time (2H)',
            'ra':
            'armature resistance',
            'xl':
            'leakage reactance',
            'xq':
            'q-axis synchronous reactance',
            'gammap':
            'active power ratio of all generators on this bus',
            'gammaq':
            'reactive power ratio',
            'coi':
            'center of inertia index',
            'gen':
            'static generator index',
            'kp':
            'active power feedback gain',
            'kw':
            'speed feedback gain',
            'S10':
            'first saturation factor',
            'S12':
            'second saturation factor',
        })
        self._units.update({
            'M': 'MWs/MVA',
            'D': 'pu',
            'fn': 'Hz',
            'ra': 'omh',
            'xd': 'omh',
            'gammap': 'pu',
            'gammaq': 'pu',
        })
        self.calls.update({
            'init1': True,
            'dyngen': True,
            'gcall': True,
            'gycall': True,
            'fcall': True,
            'fxcall': True,
            'jac0': True,
        })
        self._ac = {'bus': ['a', 'v']}
        self._states = ['delta', 'omega']
        self._fnamex = ['\\delta', '\\omega']
        self._algebs = ['p', 'q', 'pm', 'vf', 'Id', 'Iq', 'vd', 'vq']
        self._fnamey = ['P', 'Q', 'P_m', 'V_f', 'I_d', 'I_q', 'V_d', 'V_q']
        self._powers = ['M', 'D']
        self._z = ['ra', 'xl', 'xq']
        self._zeros = ['M']
        self._mandatory = ['bus', 'gen']
        self._service = ['pm0', 'vf0', 'c1', 'c2', 'c3', 'ss', 'cc', 'iM']

    def build_service(self):
        """Build service variables"""
        self.iM = div(1, self.M)

    def set_vf0(self, vf):
        """set value for self.vf0 and dae.y[self.vf]"""
        self.vf0 = vf
        self.system.dae.y[self.vf] = matrix(vf)

    def init1(self, dae):
        self.system.rmgen(self.gen)
        self.build_service()

        p0 = mul(self.u, self.system.Bus.Pg[self.a], self.gammap)
        q0 = mul(self.u, self.system.Bus.Qg[self.a], self.gammaq)
        v0 = mul(1, dae.y[self.v])
        theta0 = dae.y[self.a]
        v = polar(v0, theta0)
        S = p0 - q0 * 1j
        I = div(S, conj(v))  # NOQA
        E = v + mul(self.ra + self.xq * 1j, I)

        dae.y[self.p] = p0
        dae.y[self.q] = q0

        delta = log(div(E, abs(E) + 0j))
        dae.x[self.delta] = mul(self.u, delta.imag())
        dae.x[self.omega] = matrix(self.u, (self.n, 1), 'd')

        # d- and q-axis voltages and currents
        vdq = mul(self.u, mul(v, exp(jpi2 - delta)))
        idq = mul(self.u, mul(I, exp(jpi2 - delta)))
        vd = dae.y[self.vd] = vdq.real()
        vq = dae.y[self.vq] = vdq.imag()
        Id = dae.y[self.Id] = idq.real()
        Iq = dae.y[self.Iq] = idq.imag()

        # electro-mechanical torques / powers
        self.pm0 = mul(vq + mul(self.ra, Iq), Iq) + mul(
            vd + mul(self.ra, Id), Id)
        dae.y[self.pm] = self.pm0

    def gcall(self, dae):
        nzeros = [0] * self.n
        v = mul(self.u, dae.y[self.v])
        vd = mul(self.u, dae.y[self.vd])
        vq = mul(self.u, dae.y[self.vq])
        Id = mul(self.u, dae.y[self.Id])
        Iq = mul(self.u, dae.y[self.Iq])
        self.ss = sin(dae.x[self.delta] - dae.y[self.a])
        self.cc = cos(dae.x[self.delta] - dae.y[self.a])

        dae.g -= spmatrix(dae.y[self.p], self.a, nzeros, (dae.m, 1), 'd')
        dae.g -= spmatrix(dae.y[self.q], self.v, nzeros, (dae.m, 1), 'd')
        dae.g -= spmatrix(vd - mul(v, self.ss), self.vd, nzeros, (dae.m, 1),
                          'd')  # note d(vd)/d(delta)
        dae.g -= spmatrix(vq - mul(v, self.cc), self.vq, nzeros, (dae.m, 1),
                          'd')  # note d(vq)/d(delta)
        dae.g += spmatrix(
            mul(self.u, vd, Id) + mul(self.u, vq, Iq) - dae.y[self.p], self.p,
            nzeros, (dae.m, 1), 'd')
        dae.g += spmatrix(
            mul(self.u, vq, Id) - mul(self.u, vd, Iq) - dae.y[self.q], self.q,
            nzeros, (dae.m, 1), 'd')
        dae.g += spmatrix(dae.y[self.pm] - mul(self.u, self.pm0), self.pm,
                          nzeros, (dae.m, 1), 'd')
        dae.g += spmatrix(dae.y[self.vf] - mul(self.u, self.vf0), self.vf,
                          nzeros, (dae.m, 1), 'd')

    def saturation(self, e1q):
        """Saturation characteristic function"""
        return e1q

    def fcall(self, dae):
        dae.f[self.delta] = mul(self.u, self.system.wb,
                                dae.x[self.omega] - 1)

    def jac0(self, dae):
        dae.add_jac(Gy0, -self.u, self.a, self.p)
        dae.add_jac(Gy0, -self.u, self.v, self.q)
        dae.add_jac(Gy0, -self.u + 1e-6, self.vd, self.vd)
        dae.add_jac(Gy0, -self.u + 1e-6, self.vq, self.vq)
        dae.add_jac(Gy0, -1.0, self.p, self.p)
        dae.add_jac(Gy0, -1.0, self.q, self.q)
        dae.add_jac(Gy0, 1.0, self.pm, self.pm)
        dae.add_jac(Gy0, 1.0, self.vf, self.vf)

        dae.add_jac(Fx0, 1e-6, self.delta, self.delta)
        dae.add_jac(Fx0, mul(self.u, self.system.wb), self.delta,
                    self.omega)

    def gycall(self, dae):
        dae.add_jac(Gy, mul(self.u, dae.y[self.Id]), self.p, self.vd)
        dae.add_jac(Gy, mul(self.u, dae.y[self.Iq]), self.p, self.vq)
        dae.add_jac(Gy, mul(self.u, dae.y[self.vd]), self.p, self.Id)
        dae.add_jac(Gy, mul(self.u, dae.y[self.vq]), self.p, self.Iq)

        dae.add_jac(Gy, -mul(self.u, dae.y[self.Iq]), self.q, self.vd)
        dae.add_jac(Gy, mul(self.u, dae.y[self.Id]), self.q, self.vq)
        dae.add_jac(Gy, mul(self.u, dae.y[self.vq]), self.q, self.Id)
        dae.add_jac(Gy, -mul(self.u, dae.y[self.vd]), self.q, self.Iq)

        dae.add_jac(Gy, -mul(self.u, dae.y[self.v], self.cc), self.vd, self.a)
        dae.add_jac(Gy, mul(self.u, self.ss), self.vd, self.v)

        dae.add_jac(Gy, mul(self.u, dae.y[self.v], self.ss), self.vq, self.a)
        dae.add_jac(Gy, mul(self.u, self.cc), self.vq, self.v)

    def fxcall(self, dae):
        dae.add_jac(Gx, mul(self.u, dae.y[self.v], self.cc), self.vd,
                    self.delta)
        dae.add_jac(Gx, -mul(self.u, dae.y[self.v], self.ss), self.vq,
                    self.delta)


class Ord2(SynBase):
    """2nd order classical model"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'Syn2'
        self._data.update({'xd1': 1.9})
        self._params.extend(['xd1'])
        self._descr.update({'xd1': 'synchronous reactance'})
        self._units.update({'xd1': 'omh'})
        self._z.extend(['xd1'])

    def build_service(self):
        super(Ord2, self).build_service()
        self.xq = self.xd1

    def init1(self, dae):
        super().init1(dae)
        self.set_vf0(dae.y[self.vq] + mul(self.ra, dae.y[self.Iq]) +
                     mul(self.xd1, dae.y[self.Id]))

    def gcall(self, dae):
        super(Ord2, self).gcall(dae)
        nzeros = [0] * self.n
        dae.g += spmatrix(
            mul(self.xd1, dae.y[self.Id]) - dae.y[self.vf], self.Id, nzeros,
            (dae.m, 1), 'd')
        dae.g += spmatrix(
            mul(self.xd1, dae.y[self.Iq]), self.Iq, nzeros, (dae.m, 1), 'd')

    def jac0(self, dae):
        super(Ord2, self).jac0(dae)
        dae.add_jac(Gy0, self.xd1, self.Id, self.Id)
        dae.add_jac(Gy0, -1.0, self.Id, self.vf)

        dae.add_jac(Gy0, self.xd1, self.Iq, self.Iq)


class Ord6a(SynBase):
    """6th order the Marconato's Model"""

    # todo: check e1d
    def __init__(self, system, name):
        super(Ord6a, self).__init__(system, name)
        self._name = 'Syn6a'
        self._data.update({
            'xd': 1.9,
            'xd1': 0.302,
            'xq1': 0.5,
            'xd2': 0.204,
            'xq2': 0.3,
            'Td10': 8.0,
            'Tq10': 0.8,
            'Td20': 0.04,
            'Tq20': 0.02,
            'Taa': 0.0,
            'S10': 0,
            'S12': 0
        })
        self._params.extend([
            'xd', 'xd1', 'xq1', 'xd2', 'xq2', 'Td10', 'Tq10', 'Td20', 'Tq20',
            'Taa', 'S10', 'S12'
        ])
        self._descr.update({
            'xd': 'd-axis synchronous reactance',
            'xd1': 'd-axis transient reactance',
            'xq1': 'q-axis transient reactance',
            'xd2': 'd-axis sub-transient reactance',
            'xq2': 'q-axis sub-transient reactance',
            'Td10': 'd-axis transient time constant',
            'Tq10': 'q-axis transient time constant',
            'Td20': 'd-axis sub-transient time constant',
            'Tq20': 'q-axis sub-transient time constant',
            'Taa': 'd-axis additional leakage time constant',
        })
        self._units.update({
            'xd': 'omh',
            'xd1': 'omh',
            'xd2': 'omh',
            'xq1': 'omh',
            'xq2': 'omh',
            'Td10': 's',
            'Tq10': 's',
            'Td20': 's',
            'Tq20': 's',
            'Taa': 's',
            'S10': 'n/a',
            'S12': 'n/a'
        })
        self._mandatory.extend([
            'xd', 'xq', 'xd1', 'xq1', 'xd2', 'xq2', 'Td10', 'Tq10', 'Td20',
            'Tq20'
        ])
        self._service.extend([
            'c1', 'c2', 'c3', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'b1', 'b2',
            'b3', 'b4'
        ])
        self._states.extend(['e1d', 'e1q', 'e2d', 'e2q'])
        self._fnamex.extend(['e\'_d', 'e\'_q', 'e\'\'_d', 'e\'\'_q'])
        self._z.extend(['xd', 'xd1', 'xq1', 'xd2', 'xq2'])
        self._times.extend(['Td10', 'Tq10', 'Td20', 'Tq20', 'Taa'])

    def build_service(self):
        super(Ord6a, self).build_service()
        K = div(1, self.ra**2 + self.xd2**2)
        self.gd = mul(
            div(self.xd2, self.xd1), div(self.Td20, self.Td10),
            self.xd - self.xd1)
        self.gq = mul(
            div(self.xq2, self.xq1), div(self.Tq20, self.Tq10),
            self.xq - self.xq1)
        self.c1 = mul(self.ra, K)
        self.c2 = mul(self.xd2, K)
        self.c3 = mul(self.xq2, K)
        self.a1 = div(self.u, self.Td20)
        self.a2 = mul(self.a1, self.xd1 - self.xd2 + self.gd)
        self.a3 = mul(self.u, div(self.Taa, mul(self.Td10, self.Td20)))
        self.a4 = div(self.u, self.Td10)
        self.a5 = mul(self.a4, self.xd - self.xd1 - self.gd)
        self.a6 = mul(self.a4, 1 - div(self.Taa, self.Td10))
        self.b1 = div(self.u, self.Tq20)
        self.b2 = mul(self.b1, self.xq1 - self.xq2 + self.gq)
        self.b3 = div(self.u, self.Tq10)
        self.b4 = mul(self.b3, self.xq - self.xq1 - self.gq)

    def init1(self, dae):
        super(Ord6a, self).init1(dae)
        vd = mul(self.u, dae.y[self.vd])
        vq = mul(self.u, dae.y[self.vq])
        Id = mul(self.u, dae.y[self.Id])
        Iq = mul(self.u, dae.y[self.Iq])

        k1 = self.xd - self.xd1 - self.gd
        k2 = self.xd1 - self.xd2 + self.gd
        kq = self.xq - self.xq1 - self.gq
        kt = div(self.Taa, self.Td10)

        dae.x[self.e2q] = vq + mul(self.ra, Iq) + mul(self.xd2, Id)
        dae.x[self.e2d] = vd + mul(self.ra, Id) - mul(self.xq2, Iq)
        dae.x[self.e1d] = mul(Iq, kq)
        dae.x[self.e1q] = dae.x[self.e2q] + mul(k2, Id) - mul(
            kt,
            mul(k1 + k2, Id) + dae.x[self.e2q])
        self.set_vf0(
            div(mul(k1, Id) + self.saturation(dae.x[self.e1q]), 1 - kt))

    def gcall(self, dae):
        super(Ord6a, self).gcall(dae)
        dae.g[self.Id] += mul(self.u, self.xd2, dae.y[self.Id]) - mul(
            self.u, dae.x[self.e2q])
        dae.g[self.Iq] += mul(self.u, self.xq2, dae.y[self.Iq]) + mul(
            self.u, dae.x[self.e2d])

    def fcall(self, dae):
        super(Ord6a, self).fcall(dae)
        Id = mul(self.u, dae.y[self.Id])
        Iq = mul(self.u, dae.y[self.Iq])
        dae.f[self.e1d] = -mul(self.b3, dae.x[self.e1d], self.u) + mul(
            self.b4, Iq, self.u)
        dae.f[self.e1q] = -mul(
            self.saturation(dae.x[self.e1q]), self.a4, self.u) - mul(
                self.a5, Id, self.u) + mul(self.a6, dae.y[self.vf], self.u)
        dae.f[self.e2d] = -mul(self.b1, dae.x[self.e2d], self.u) + mul(
            self.b1, dae.x[self.e1d], self.u) + mul(self.b2, Iq, self.u)
        dae.f[self.e2q] = -mul(self.a1, dae.x[self.e2q], self.u) + mul(
            self.a1, dae.x[self.e1q], self.u) - mul(self.a2, Id, self.u) + mul(
                self.a3, dae.y[self.vf], self.u)

    def jac0(self, dae):
        super(Ord6a, self).jac0(dae)
        dae.add_jac(Gy0, mul(self.u, self.xd2) + 1e-6, self.Id, self.Id)
        dae.add_jac(Gx0, -self.u, self.Id, self.e2q)

        dae.add_jac(Gy0, mul(self.u, self.xq2) + 1e-6, self.Iq, self.Iq)
        dae.add_jac(Gx0, self.u, self.Iq, self.e2d)

        dae.add_jac(Fx0, -mul(self.u, self.b3) + 1e-6, self.e1d, self.e1d)
        dae.add_jac(Fy0, mul(self.u, self.b4), self.e1d, self.Iq)

        dae.add_jac(Fx0, -mul(self.u, self.a4) + 1e-6, self.e1q, self.e1q)
        dae.add_jac(Fy0, mul(self.u, self.a6), self.e1q, self.vf)
        dae.add_jac(Fy0, -mul(self.u, self.a5), self.e1q, self.Id)

        dae.add_jac(Fx0, -mul(self.u, self.b1) + 1e-6, self.e2d, self.e2d)
        dae.add_jac(Fx0, mul(self.u, self.b1), self.e2d, self.e1d)
        dae.add_jac(Fy0, mul(self.u, self.b2), self.e2d, self.Iq)

        dae.add_jac(Fx0, -mul(self.u, self.a1) + 1e-6, self.e2q, self.e2q)
        dae.add_jac(Fx0, mul(self.u, self.a1), self.e2q, self.e1q)
        dae.add_jac(Fy0, mul(self.u, self.a3), self.e2q, self.vf)
        dae.add_jac(Fy0, -mul(self.u, self.a2), self.e2q, self.Id)


class Flux0(object):
    """The simplified flux model as an appendix to generator models.
         0 = ra*id + psiq + vd
         0 = ra*iq - psid + vq
    """

    def __init__(self):
        self._algebs.extend(['psid', 'psiq'])
        self._fnamey.extend(['\\psi_d', '\\psi_q'])
        self._init()

    def init1(self, dae):
        dae.y[
            self.psiq] = -mul(self.u, self.ra, dae.y[self.Id]) - dae.y[self.vd]
        dae.y[self.psid] = mul(self.u, self.ra,
                               dae.y[self.Iq]) + dae.y[self.vq]

    def gcall(self, dae):
        dae.g[self.psiq] = mul(self.u, self.ra,
                               dae.y[self.Id]) + dae.y[self.psiq] + mul(
                                   self.u, dae.y[self.vd])
        dae.g[self.psid] = mul(self.u, self.ra,
                               dae.y[self.Iq]) - dae.y[self.psid] + mul(
                                   self.u, dae.y[self.vq])
        dae.g[self.Id] += mul(self.u, dae.y[self.psid])
        dae.g[self.Iq] += mul(self.u, dae.y[self.psiq])

    def gycall(self, dae):
        dae.add_jac(Gy, mul(self.u, self.ra), self.psiq, self.Id)
        dae.add_jac(Gy, mul(self.u, self.ra), self.psid, self.Iq)

    def fcall(self, dae):
        dae.f[self.omega] = mul(
            self.u, self.iM,
            dae.y[self.pm] - mul(dae.y[self.psid], dae.y[self.Iq]) + mul(
                dae.y[self.psiq], dae.y[self.Id]) - mul(
                    self.D, dae.x[self.omega] - 1))

    def fxcall(self, dae):
        dae.add_jac(Fy, mul(self.u, dae.y[self.Id], self.iM), self.omega,
                    self.psiq)
        dae.add_jac(Fy, mul(self.u, dae.y[self.psiq], self.iM), self.omega,
                    self.Id)
        dae.add_jac(Fy, -mul(self.u, dae.y[self.Iq], self.iM), self.omega,
                    self.psid)
        dae.add_jac(Fy, -mul(self.u, dae.y[self.psid], self.iM), self.omega,
                    self.Iq)

    def jac0(self, dae):
        dae.add_jac(Gy0, 1.0, self.psiq, self.psiq)
        dae.add_jac(Gy0, self.u, self.psiq, self.vd)

        dae.add_jac(Gy0, -1.0, self.psid, self.psid)
        dae.add_jac(Gy0, self.u, self.psid, self.vq)

        dae.add_jac(Gy0, self.u, self.Id, self.psid)

        dae.add_jac(Gy0, self.u, self.Iq, self.psiq)

        dae.add_jac(Fx0, -mul(self.u, self.iM, self.D) + 1e-6, self.omega,
                    self.omega)
        dae.add_jac(Fy0, mul(self.u, self.iM), self.omega, self.pm)


class Flux2(object):
    """Full electromagnetic transient of flux linkage
       d(psid) = wb * (ra * id + omega * psiq + vd)
       d(psiq) = wb * (ra * iq - omega * psid + vq)
       """

    def __init__(self):
        self._states.extend(['psid', 'psiq'])
        self._fnamex.extend(['\\psi_d', '\\psi_q'])
        self._init()

    def init1(self, dae):
        dae.x[self.psiq] = -mul(self.ra, dae.y[self.Id]) - dae.y[self.vd]
        dae.x[self.psid] = mul(self.ra, dae.y[self.Iq]) + dae.y[self.vq]

    def fcall(self, dae):
        wn = mul(self.system.wb, self.u)
        dae.f[self.psid] = mul(
            wn,
            mul(self.ra, dae.y[self.Id]) + mul(
                dae.x[self.omega], dae.x[self.psiq]) + dae.y[self.vd])
        dae.f[self.psiq] = mul(
            wn,
            mul(self.ra, dae.y[self.Iq]) - mul(
                dae.x[self.omega], dae.x[self.psid]) + dae.y[self.vq])

    def fxcall(self, dae):
        wn = mul(self.system.wb, self.u)
        dae.add_jac(Fy, mul(wn, self.ra), self.psid, self.Id)
        dae.add_jac(Fx, mul(wn, dae.x[self.omega]), self.psid, self.psiq)
        dae.add_jac(Fx, mul(wn, dae.x[self.psiq]), self.psid, self.omega)

        dae.add_jac(Fy, mul(wn, self.ra), self.psiq, self.Iq)
        dae.add_jac(Fx, -mul(wn, dae.x[self.omega]), self.psiq, self.psid)
        dae.add_jac(Fx, -mul(wn, dae.x[self.psid]), self.psiq, self.omega)

    def jac0(self, dae):
        wn = mul(self.system.wb, self.u)
        dae.add_jac(Fy0, wn, self.psid, self.vd)

        dae.add_jac(Fy0, wn, self.psiq, self.vq)


class Syn2(Ord2, Flux0):
    """2nd-order generator model. Inherited from (Ord2, Flux0)  """

    def __init__(self, system, name):
        Ord2.__init__(self, system, name)
        Flux0.__init__(self)

    def init1(self, dae):
        Ord2.init1(self, dae)
        Flux0.init1(self, dae)

    def gcall(self, dae):
        Ord2.gcall(self, dae)
        Flux0.gcall(self, dae)

    def fcall(self, dae):
        Ord2.fcall(self, dae)
        Flux0.fcall(self, dae)

    def jac0(self, dae):
        Ord2.jac0(self, dae)
        Flux0.jac0(self, dae)

    def gycall(self, dae):
        Ord2.gycall(self, dae)
        Flux0.gycall(self, dae)

    def fxcall(self, dae):
        Ord2.fxcall(self, dae)
        Flux0.fxcall(self, dae)


class Syn6a(Ord6a, Flux0):
    """GENROU, 6th-order generator model with xd2 = xq2."""

    def __init__(self, system, name):
        Ord6a.__init__(self, system, name)
        Flux0.__init__(self)

    def init1(self, dae):
        Ord6a.init1(self, dae)
        Flux0.init1(self, dae)

    def gcall(self, dae):
        Ord6a.gcall(self, dae)
        Flux0.gcall(self, dae)

    def fcall(self, dae):
        Ord6a.fcall(self, dae)
        Flux0.fcall(self, dae)

    def jac0(self, dae):
        Ord6a.jac0(self, dae)
        Flux0.jac0(self, dae)

    def gycall(self, dae):
        Ord6a.gycall(self, dae)
        Flux0.gycall(self, dae)

    def fxcall(self, dae):
        Ord2.fxcall(self, dae)
        Flux0.fxcall(self, dae)
