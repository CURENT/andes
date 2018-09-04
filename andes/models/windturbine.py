from cvxopt import matrix, spmatrix, sparse
from cvxopt import mul, div, sin, cos, exp

from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA
from ..consts import pi
from .base import ModelBase

try:
    from cvxoptklu.klu import linsolve
except ImportError:
    from cvxopt.umfpack import linsolve

from ..utils.math import zeros, ones, agtb, ageb, altb, aleb, aandb, aneb
from ..utils.math import mfloor, mround, mmax, mmin, not0

import logging
logger = logging.getLogger(__name__)


class MPPT(object):
    """MPPT control algorithm"""

    def __init__(self, system, name):
        self._algebs.extend(['pwa'])
        self._fnamey.extend(['P_w^{opt}'])

    def init1(self, dae):
        dae.y[self.pwa] = mmax(mmin(2 * dae.x[self.omega_m] - 1, 1), 0)

    def gcall(self, dae):
        dae.g[self.pwa] = mmax(mmin(2 * dae.x[self.omega_m] - 1, 1),
                               0) - dae.y[self.pwa]
        dae.hard_limit(self.pwa, 0, 1)

    def gycall(self, dae):
        pass

    def jac0(self, dae):
        dae.add_jac(Gy0, -1, self.pwa, self.pwa)
        dae.add_jac(Gx0, 2, self.pwa, self.omega_m)
        dae.add_jac(Gy0, 1e-6, self.pwa, self.pwa)


class Turbine(object):
    """Generic wind turbine model"""

    def __init__(self, system, name):
        self._algebs.extend(['pw', 'cp', 'lamb', 'ilamb'])
        self._fnamey.extend([
            'P_w',
            'c_p',
            '\\lambda',
            '\\frac{1}{\\lambda}',
        ])
        self._states.extend(['theta_p'])
        self._fnamex.extend(['\\theta_p'])
        self._mandatory.extend(['wind'])
        self._params.extend(
            ['Kp', 'nblade', 'ngen', 'npole', 'R', 'Tp', 'ngb', 'H'])
        self._powers.extend(['H'])
        self._times.extend(['Tp'])
        self._data.update({
            'R': 35,
            'ngb': 0.011235,
            'npole': 4,
            'wind': 0,
            'Kp': 10,
            'H': 2,
            'Tp': 3.0,
            'nblade': 3,
            'ngen': 40,
        })
        self._descr.update({
            'R': 'Rotor radius',
            'npole': 'Number of poles',
            'wind': 'Wind time series idx',
            'ngb': 'Gear box ratio',
            'Kp': 'Pitch control gain',
            'H': 'Machine rotor and turbine inertia constant',
            'Tp': 'Pitch control time constant',
            'nblade': 'Number of blades',
            'ngen': 'Number of generators',
        })
        self._units.update({
            'R': 'm',
            'Kp': 'pu',
            'H': 'MWs/MVA',
            'Tp': 's',
        })

    @property
    def phi(self):
        deg1 = pi / 180
        dae = self.system.dae
        above = agtb(dae.x[self.omega_m], 1)
        phi_degree_step = mfloor((dae.x[self.omega_m] - 1) / deg1) * deg1
        return mul(phi_degree_step, above)

    def servcall(self, dae):
        self.copy_data_ext('Wind', 'vw', 'vw', self.wind)
        self.copy_data_ext('Wind', 'rho', 'rho', self.wind)
        self.copy_data_ext('Wind', 'Vwn', 'Vwn', self.wind)
        self.mva_mega = 100e6

    def windpower(self, ngen, rho, vw, Ar, R, omega, theta, derivative=False):
        mva_mega = self.system.mva * 1e6
        lamb = omega * R / vw
        ilamb = 1 / (1 / (lamb + 0.08 * theta) - 0.035 / (theta**3 + 1))
        cp = 0.22 * (116 / ilamb - 0.4 * theta - 5) * exp(-12.5 / ilamb)
        pw = 0.5 * ngen * rho * cp * Ar * vw**3 / mva_mega

        a1 = exp(-12.5 / ilamb)
        a2 = (lamb + 0.08 * theta)**2
        a3 = 116. / ilamb - 0.4 * theta - 5
        a4 = -9.28 / (lamb + 0.08 * theta) ** 2 + \
            12.180 * theta * theta / (theta ** 3 + 1) ** 2 - 0.4
        a5 = 1.000 / (lamb + 0.08 * theta) ** 2 - \
            1.3125 * theta * theta / (theta ** 3 + 1) ** 2

        jac = ones(1, 3)
        jac[0] = ngen * R * a1 * rho * vw * vw * Ar * (
            -12.760 + 1.3750 * a3) / a2 / mva_mega
        jac[1] = ngen * (omega * R * (12.760 - 1.3750 * a3) / a2 +
                         0.330 * a3 * vw) * vw * Ar * rho * a1 / mva_mega
        jac[2] = ngen * 0.110 * rho * (
            a4 + a3 * a5) * a1 * Ar * vw**3 / mva_mega

        return pw, jac

    def init1(self, dae):
        # electrical torque in pu
        # te = mul(
        #     xmu,
        #     mul(dae.x[self.irq], dae.y[self.isd]) - mul(
        #         dae.x[self.ird], dae.y[self.isq]))

        for i in range(self.n):
            if self.te0[i] < 0:
                logger.error('Pe < 0 on bus <{}>. Wind speed initialize failed.'
                             .format(self.bus[i]))

        # wind power in pu
        omega = dae.x[self.omega_m]
        theta = dae.x[self.theta_p]
        pw = mul(self.te0, dae.x[self.omega_m])
        dae.y[self.pw] = pw

        # wind speed initialization loop

        R = 4 * pi * self.system.config.freq * mul(self.R, self.ngb,
                                                   div(1, self.npole))
        AA = pi * self.R**2
        vw = 0.9 * self.Vwn

        for i in range(self.n):
            mis = 1
            iter = 0
            while abs(mis) > self.system.tds.config.tol:
                if iter > 50:
                    self.message(
                        'Wind <{}> init failed. '
                        'Try increasing the nominal wind speed.'.format(
                            self.wind[i]))
                    break

                pw_iter, jac = self.windpower(self.ngen[i], self.rho[i], vw[i],
                                              AA[i], R[i], omega[i], theta[i])

                mis = pw_iter - pw[i]
                inc = -mis / jac[1]
                vw[i] += inc
                iter += 1

        # set wind speed
        dae.x[self.vw] = div(vw, self.Vwn)

        lamb = div(omega, vw, div(1, R))
        ilamb = div(1, lamb + 0.08 * theta) - div(0.035, theta**3 + 1)
        cp = 0.22 * mul(
            mul(116, ilamb) - 0.4 * theta - 5, exp(mul(-12.5, ilamb)))

        dae.y[self.lamb] = lamb
        dae.y[self.ilamb] = ilamb
        dae.y[self.cp] = cp

    def gcall(self, dae):
        dae.g[self.pw] = -dae.y[self.pw] + mul(
            0.5, dae.y[self.cp], self.ngen, pi, self.rho, (self.R)**2,
            (self.Vwn)**3, div(1, self.mva_mega), (dae.x[self.vw])**3)
        dae.g[self.lamb] = -dae.y[self.lamb] + mul(
            4, self.R, self.fn, self.ngb, dae.x[self.omega_m], pi,
            div(1, self.Vwn), div(1, self.npole), div(1, dae.x[self.vw]))
        dae.g[self.cp] = -dae.y[self.cp] + mul(
            -1.1 + mul(25.52, dae.y[self.ilamb]) + mul(-0.08800000000000001,
                                                       dae.x[self.theta_p]),
            exp(mul(-12.5, dae.y[self.ilamb])))
        dae.g[self.ilamb] = div(
            1, dae.y[self.lamb] + mul(
                0.08, dae.x[self.theta_p])) - dae.y[self.ilamb] + mul(
                    -0.035, div(1, 1 + (dae.x[self.theta_p])**3))

    def fcall(self, dae):
        dae.f[self.theta_p] = mul(
            div(1, self.Tp), -dae.x[self.theta_p] + mul(
                self.Kp, self.phi, -1 + dae.x[self.omega_m]))
        dae.anti_windup(self.theta_p, 0, pi)

    def gycall(self, dae):
        dae.add_jac(
            Gy,
            mul(0.5, self.ngen, pi, self.rho, (self.R)**2, (self.Vwn)**3,
                div(1, self.mva_mega), (dae.x[self.vw])**3), self.pw, self.cp)
        dae.add_jac(
            Gy,
            mul(25.52, exp(mul(-12.5, dae.y[self.ilamb]))) + mul(
                -12.5, -1.1 + mul(25.52, dae.y[self.ilamb]) + mul(
                    -0.08800000000000001, dae.x[self.theta_p]),
                exp(mul(-12.5, dae.y[self.ilamb]))), self.cp, self.ilamb)
        dae.add_jac(Gy,
                    -(dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))**-2,
                    self.ilamb, self.lamb)

    def fxcall(self, dae):
        dae.add_jac(
            Gx,
            mul(1.5, dae.y[self.cp],
                self.ngen, pi, self.rho, (self.R)**2, (self.Vwn)**3,
                div(1, self.mva_mega), (dae.x[self.vw])**2), self.pw, self.vw)
        dae.add_jac(Gx, mul(-0.088, exp(mul(-12.5, dae.y[self.ilamb]))),
                    self.cp, self.theta_p)
        dae.add_jac(
            Gx,
            mul(-4, self.R, self.fn, self.ngb, dae.x[self.omega_m], pi,
                div(1, self.Vwn), div(1, self.npole), (dae.x[self.vw])**-2),
            self.lamb, self.vw)
        dae.add_jac(
            Gx,
            mul(4, self.R, self.fn, self.ngb, pi, div(1, self.Vwn),
                div(1, self.npole), div(1, dae.x[self.vw])), self.lamb,
            self.omega_m)
        dae.add_jac(
            Gx,
            mul(-0.08,
                (dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))**-2) + mul(
                    0.10500000000000001, (dae.x[self.theta_p])**2,
                    (1 + (dae.x[self.theta_p])**3)**-2), self.ilamb,
            self.theta_p)

    def jac0(self, dae):
        dae.add_jac(Gy0, -1, self.pw, self.pw)
        dae.add_jac(Gy0, -1, self.cp, self.cp)
        dae.add_jac(Gy0, -1, self.lamb, self.lamb)
        dae.add_jac(Gy0, -1, self.ilamb, self.ilamb)
        dae.add_jac(Gy0, 1e-6, self.pw, self.pw)
        dae.add_jac(Gy0, 1e-6, self.cp, self.cp)
        dae.add_jac(Gy0, 1e-6, self.lamb, self.lamb)
        dae.add_jac(Gy0, 1e-6, self.ilamb, self.ilamb)
        dae.add_jac(Fx0, -div(1, self.Tp), self.theta_p, self.theta_p)
        dae.add_jac(Fx0, mul(self.Kp, self.phi, div(1, self.Tp)), self.theta_p,
                    self.omega_m)


class WTG4DC(ModelBase, Turbine, MPPT):
    """Wind turbine type IV DC output"""

    def __init__(self, system, name):
        ModelBase.__init__(self, system, name)
        Turbine.__init__(self, system, name)
        MPPT.__init__(self, system, name)
        self._group = 'WTG'
        self._name = 'WTG4DC'
        self._algebs.extend(['isd', 'vsd', 'vsq', 'ps', 'te'])
        self._fnamex.extend(['\\omega_m', 'I_{s, q}'])
        self._fnamey.extend(
            ['I_{s, d}', 'V_{s, d}', 'V_{s, q}', 'P_s', '\\tau_e'])
        self._mandatory.extend(['node1', 'node2', 'dcgen', 'wind'])
        self._params.extend([
            'fn', 'rs', 'xd', 'xq', 'psip', 'Tep', 'Teq', 'pmax', 'pmin',
            'qmax', 'qmin', 'u', 'Sn', 'Kdc', 'Ki', 'Kcoi'
        ])
        self._powers.extend(['H', 'pmax', 'pmin', 'qmax', 'qmin'])
        self._service.extend(['qs0'])
        self._states.extend(['omega_m', 'isq'])
        self._times.extend(['Tep', 'Teq', 'Tp'])
        self._r.extend(['rs', 'xd', 'xq'])
        self._data.update({
            'fn': 60,
            'Sn': 40,
            'Tep': 0.01,
            'Teq': 0.01,
            'dcgen': 0,
            'node1': 0,
            'node2': 0,
            'pmax': 1.0,
            'pmin': 0,
            'psip': 0.1,
            'qmax': 0.6,
            'qmin': -0.6,
            'rs': 0.01,
            'u': 1,
            'wind': 0,
            'xd': 0.08,
            'xq': 0.1,
            'Kdc': 0,
            'Ki': 0,
            'Kcoi': 0,
            'busfreq': None,
            'coi': None,
        })
        self._descr.update({
            'Sn':
            'Power rating',
            'Tep':
            'Active power time constant',
            'Teq':
            'Reactive power time constant',
            'dcgen':
            'Static generator index',
            'node1':
            'DC node 1',
            'node2':
            'DC node 2',
            'pmax':
            'Maximum active power',
            'pmin':
            'Minimum reactive power',
            'psip':
            'Permanent field flux',
            'qmax':
            'Maximum reactive power',
            'qmin':
            'Minimum reactive power',
            'rs':
            'Stator resistance',
            'u':
            'Connection status',
            'wind':
            'Wind time series index',
            'xd':
            'd-axis reactance',
            'xq':
            'q-axis reactance',
            'Kdc':
            'DC voltage droop on P reference',
            'Ki':
            'Bus frequency derivative droop on P reference',
            'Kcoi':
            'COI frequency derivative droop on P reference',
        })
        self._units.update({
            'Sn': 'MVA',
            'Tep': 's',
            'Teq': 's',
            'pmax': 'pu',
            'psip': 'Weber',
            'qmax': 'pu',
            'rs': 'pu',
            'xd': 'pu',
            'xq': 'pu'
        })
        self.calls.update({
            'fcall': True,
            'fxcall': True,
            'gcall': True,
            'gycall': True,
            'init1': True,
            'jac0': True
        })
        self._dc = {
            'node1': 'v1',
            'node2': 'v2',
        }

        self._init()

    def _init(self):
        super(WTG4DC, self)._init()

    def base(self):
        super(WTG4DC, self).base()

    def servcall(self, dae):
        self.copy_data_ext('DCgen', 'u', 'u0', self.dcgen)
        self.copy_data_ext('DCgen', 'P', 'p0', self.dcgen)
        self.copy_data_ext('Wind', 'vw', 'vw', self.wind)
        self.copy_data_ext('Wind', 'rho', 'rho', self.wind)
        self.copy_data_ext('Wind', 'Vwn', 'Vwn', self.wind)
        # self.copy_data_ext('Node', 'v', 'v1', self.node1)
        # self.copy_data_ext('Node', 'v', 'v2', self.node2)
        # self.qs0 = 0
        # TODO: Fix this dirty hard code
        if self.busfreq[0] is not None:
            self.copy_data_ext('BusFreq', 'dwdt', 'dwdt', self.busfreq)
        else:
            self.dwdt = matrix(0, (self.n, 1))
        if self.coi[0] is not None:
            self.copy_data_ext('COI', 'dwdt', 'dwdt_coi', self.coi)
        else:
            self.dwdt_coi = matrix(0, (self.n, 1))
        Turbine.servcall(self, dae)

    def init1(self, dae):
        self.servcall(dae)
        mva = self.system.mva
        self.p0 = mul(self.p0, 1)
        self.v120 = self.v12

        self.toMb = div(mva, self.Sn)  # to machine base
        self.toSb = self.Sn / mva  # to system base
        rs = matrix(self.rs)
        xd = matrix(self.xd)
        xq = matrix(self.xq)
        psip = matrix(self.psip)
        Pg = matrix(self.p0)

        # rotor speed
        omega = 1 * (ageb(mva * Pg, self.Sn)) + \
            mul(0.5 + 0.5 * mul(Pg, self.toMb),
                aandb(agtb(Pg, 0), altb(mva * Pg, self.Sn))) + \
            0.5 * (aleb(mva * Pg, 0))

        theta = mul(self.Kp, mround(1000 * (omega - 1)) / 1000)
        theta = mmax(theta, 0)

        # variables to initialize iteratively: vsd, vsq, isd, isq

        vsd = matrix(0.8, (self.n, 1))
        vsq = matrix(0.6, (self.n, 1))
        isd = matrix(self.p0 / 2)
        isq = matrix(self.p0 / 2)

        for i in range(self.n):
            # vsd = 0.5
            # vsq = self.psip[i]
            # isd = Pg / 2
            # isq = Pg / 2
            x = matrix([vsd[i], vsq[i], isd[i], isq[i]])

            mis = ones(4, 1)
            jac = sparse(matrix(0, (4, 4), 'd'))
            iter = 0
            while (max(abs(mis))) > self.system.tds.config.tol:
                if iter > 40:
                    logger.error(
                        'Initialization of WTG4DC <{}> failed.'.format(
                            self.name[i]))
                    break
                mis[0] = x[0] * x[2] + x[1] * x[3] - Pg[i]
                # mis[1] = omega[i]*x[3] * (psip[i] + (xq[i] - xd[i]) * x[2])\
                #     - Pg[i]
                mis[1] = omega[i] * x[3] * (psip[i] - xd[i] * x[2]) - Pg[i]
                mis[2] = -x[0] - rs[i] * x[2] + omega[i] * xq[i] * x[3]
                mis[3] = x[1] + rs[i] * x[3] + omega[i] * xd[i] * x[2] - \
                    omega[i] * psip[i]

                jac[0, 0] = x[2]
                jac[0, 1] = x[3]
                jac[0, 2] = x[0]
                jac[0, 3] = x[1]

                jac[1, 2] = omega[i] * x[3] * (-xd[i])
                jac[1, 3] = omega[i] * (psip[i] + (-xd[i]) * x[2])

                jac[2, 0] = -1
                jac[2, 2] = -rs[i]
                jac[2, 3] = omega[i] * xq[i]
                jac[3, 1] = 1
                jac[3, 2] = omega[i] * xd[i]
                jac[3, 3] = rs[i]

                linsolve(jac, mis)
                x -= mis
                iter += 1

            vsd[i] = x[0]
            vsq[i] = x[1]
            isd[i] = x[2]
            isq[i] = x[3]

        dae.y[self.isd] = isd
        dae.y[self.vsd] = vsd
        dae.y[self.vsq] = vsq

        dae.x[self.isq] = isq
        dae.x[self.omega_m] = mul(self.u0, omega)
        dae.x[self.theta_p] = mul(self.u0, theta)
        dae.y[self.pwa] = mmax(mmin(2 * dae.x[self.omega_m] - 1, 1), 0)

        self.ps0 = mul(vsd, isd) + mul(vsq, isq)
        self.qs0 = mul(vsq, isd) - mul(vsd, isq)
        self.te0 = mul(isq, psip + mul(xq - xd, isd))
        dae.y[self.te] = self.te0
        dae.y[self.ps] = self.ps0

        MPPT.init1(self, dae)
        Turbine.init1(self, dae)

        self.system.rmgen(self.dcgen)

    def gcall(self, dae):
        Turbine.gcall(self, dae)
        MPPT.gcall(self, dae)
        dae.g[self.isd] = -self.qs0 + mul(dae.y[self.isd],
                                          dae.y[self.vsq]) - mul(
                                              dae.x[self.isq], dae.y[self.vsd])
        dae.g[self.vsd] = -dae.y[self.vsd] - mul(
            dae.y[self.isd], self.rs) + mul(dae.x[self.isq],
                                            dae.x[self.omega_m], self.xq)
        dae.g[self.vsq] = -dae.y[self.vsq] - mul(
            dae.x[self.isq], self.rs) - mul(
                dae.x[self.omega_m],
                -self.psip + mul(dae.y[self.isd], self.xd))
        dae.g[self.ps] = -dae.y[self.ps] + mul(
            dae.y[self.isd], dae.y[self.vsd]) + mul(dae.x[self.isq],
                                                    dae.y[self.vsq])
        dae.g[self.te] = -dae.y[self.te] + mul(
            dae.x[self.isq],
            self.psip + mul(dae.y[self.isd], self.xq - self.xd))
        dae.g += spmatrix(
            -mul(dae.y[self.ps], div(1, dae.y[self.v1] - dae.y[self.v2])),
            self.v1, [0] * self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(
            mul(dae.y[self.ps], div(1, dae.y[self.v1] - dae.y[self.v2])),
            self.v2, [0] * self.n, (dae.m, 1), 'd')

    def fcall(self, dae):
        Turbine.gcall(self, dae)
        dae.f[self.omega_m] = mul(
            0.5, div(1, self.H),
            -dae.y[self.te] + mul(dae.y[self.pw], div(1, dae.x[self.omega_m])))

        # dae.f[self.isq] = mul(
        #     div(1, self.Teq), -dae.x[self.isq] + mul(
        #         self.toSb, dae.y[self.pwa] - mul(self.Kdc, self.v12),
        #         div(1, dae.x[self.omega_m]),
        #         div(1, self.psip - mul(dae.y[self.isd], self.xd))))

        dae.f[self.isq] = mul(
            div(1, self.Teq), -dae.x[self.isq] + mul(
                self.toSb, div(1, dae.x[self.omega_m]),
                div(1, self.psip - mul(dae.y[self.isd], self.xd)),
                dae.y[self.pwa] - mul(self.Kcoi, dae.y[self.dwdt_coi]) - mul(
                    self.Kdc,
                    (dae.y[self.v1] - dae.y[self.v2]) - self.v120) - mul(
                        self.Ki, dae.y[self.dwdt])))

    @property
    def v12(self):
        dae = self.system.dae
        return dae.y[self.v1] - dae.y[self.v2]

    def gycall(self, dae):
        Turbine.gycall(self, dae)
        MPPT.gycall(self, dae)
        dae.add_jac(Gy, -dae.x[self.isq], self.isd, self.vsd)
        dae.add_jac(Gy, dae.y[self.vsq], self.isd, self.isd)
        dae.add_jac(Gy, dae.y[self.isd], self.isd, self.vsq)
        dae.add_jac(Gy, -mul(dae.x[self.omega_m], self.xd), self.vsq, self.isd)
        dae.add_jac(Gy, dae.y[self.isd], self.ps, self.vsd)
        dae.add_jac(Gy, dae.y[self.vsd], self.ps, self.isd)
        dae.add_jac(Gy, dae.x[self.isq], self.ps, self.vsq)
        dae.add_jac(Gy, mul(dae.x[self.isq], self.xq - self.xd), self.te,
                    self.isd)
        dae.add_jac(Gy, -div(1, dae.y[self.v1] - dae.y[self.v2]), self.v1,
                    self.ps)
        dae.add_jac(
            Gy, -mul(dae.y[self.ps], (dae.y[self.v1] - dae.y[self.v2])**-2),
            self.v1, self.v2)
        dae.add_jac(Gy,
                    mul(dae.y[self.ps], (dae.y[self.v1] - dae.y[self.v2])**-2),
                    self.v1, self.v1)
        dae.add_jac(Gy, div(1, dae.y[self.v1] - dae.y[self.v2]), self.v2,
                    self.ps)
        dae.add_jac(Gy,
                    mul(dae.y[self.ps], (dae.y[self.v1] - dae.y[self.v2])**-2),
                    self.v2, self.v2)
        dae.add_jac(
            Gy, -mul(dae.y[self.ps], (dae.y[self.v1] - dae.y[self.v2])**-2),
            self.v2, self.v1)

    def fxcall(self, dae):
        Turbine.jac0(self, dae)
        dae.add_jac(Gx, -dae.y[self.vsd], self.isd, self.isq)
        dae.add_jac(Gx, mul(dae.x[self.isq], self.xq), self.vsd, self.omega_m)
        dae.add_jac(Gx, mul(dae.x[self.omega_m], self.xq), self.vsd, self.isq)
        dae.add_jac(Gx, self.psip - mul(dae.y[self.isd], self.xd), self.vsq,
                    self.omega_m)
        dae.add_jac(Gx, dae.y[self.vsq], self.ps, self.isq)
        dae.add_jac(Gx, self.psip + mul(dae.y[self.isd], self.xq - self.xd),
                    self.te, self.isq)
        dae.add_jac(
            Fx,
            mul(-0.5, dae.y[self.pw], div(1, self.H), (dae.x[self.omega_m])
                ** -2), self.omega_m, self.omega_m)

        dae.add_jac(
            Fx, -mul(
                div(1, self.Teq), (dae.x[self.omega_m])**-2,
                div(1, self.psip - mul(dae.y[self.isd], self.xd)),
                dae.y[self.pwa] - mul(self.Kcoi, dae.y[self.dwdt_coi]) - mul(
                    self.Kdc, dae.y[self.v1] - dae.y[self.v2]) - mul(
                        self.Ki, dae.y[self.dwdt])), self.isq, self.omega_m)
        dae.add_jac(Fy, mul(0.5, div(1, self.H), div(1, dae.x[self.omega_m])),
                    self.omega_m, self.pw)

        dae.add_jac(
            Fy,
            mul(self.Kdc, div(1, self.Teq), div(1, dae.x[self.omega_m]),
                div(1, self.psip - mul(dae.y[self.isd], self.xd))), self.isq,
            self.v2)
        dae.add_jac(
            Fy, -mul(self.Ki, div(1, self.Teq), div(1, dae.x[self.omega_m]),
                     div(1, self.psip - mul(dae.y[self.isd], self.xd))),
            self.isq, self.dwdt)
        dae.add_jac(
            Fy,
            mul(
                div(1, self.Teq), div(1, dae.x[self.omega_m]),
                div(1, self.psip - mul(dae.y[self.isd], self.xd))), self.isq,
            self.pwa)
        dae.add_jac(
            Fy, -mul(self.Kdc, div(1, self.Teq), div(1, dae.x[self.omega_m]),
                     div(1, self.psip - mul(dae.y[self.isd], self.xd))),
            self.isq, self.v1)
        dae.add_jac(
            Fy,
            mul(
                self.xd, div(1, self.Teq), div(1, dae.x[self.omega_m]),
                (self.psip - mul(dae.y[self.isd], self.xd))**-2,
                dae.y[self.pwa] - mul(self.Kcoi, dae.y[self.dwdt_coi]) - mul(
                    self.Kdc, dae.y[self.v1] - dae.y[self.v2]) - mul(
                        self.Ki, dae.y[self.dwdt])), self.isq, self.isd)
        dae.add_jac(
            Fy, -mul(self.Kcoi, div(1, self.Teq), div(1, dae.x[self.omega_m]),
                     div(1, self.psip - mul(dae.y[self.isd], self.xd))),
            self.isq, self.dwdt_coi)

    def jac0(self, dae):
        Turbine.jac0(self, dae)
        MPPT.jac0(self, dae)
        dae.add_jac(Gy0, -1, self.vsd, self.vsd)
        dae.add_jac(Gy0, -self.rs, self.vsd, self.isd)
        dae.add_jac(Gy0, -1, self.vsq, self.vsq)
        dae.add_jac(Gy0, -1, self.ps, self.ps)
        dae.add_jac(Gy0, -1, self.te, self.te)
        dae.add_jac(Gx0, -self.rs, self.vsq, self.isq)
        dae.add_jac(Fx0, -div(1, self.Teq), self.isq, self.isq)
        dae.add_jac(Fy0, mul(-0.5, div(1, self.H)), self.omega_m, self.te)
        dae.add_jac(Gy0, 1e-6, self.isd, self.isd)
        dae.add_jac(Gy0, 1e-6, self.vsd, self.vsd)
        dae.add_jac(Gy0, 1e-6, self.vsq, self.vsq)
        dae.add_jac(Gy0, 1e-6, self.ps, self.ps)
        dae.add_jac(Gy0, 1e-6, self.te, self.te)


class WTG3(ModelBase):
    """Wind turbine type III"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'WTG'
        self._name = 'WTG3'
        self._algebs.extend([
            'isd', 'isq', 'vrd', 'vrq', 'vsd', 'vsq', 'vref', 'pwa', 'pw',
            'cp', 'lamb', 'ilamb'
        ])
        self._fnamex.extend(['\\theta_p', '\\omega_m', 'I_{r, d}', 'I_{r, q}'])
        self._fnamey.extend([
            'I_{s, d}', 'I_{s, q}', 'V_{r, d}', 'V_{r, q}', 'V_{s, d}',
            'V_{s, q}', 'V_{ref}', 'P_{\\omega a}', 'P_w', 'c_p', '\\lambda',
            '\\frac{1}{\\lambda}', '\\omega_{ref}'
        ])
        self._mandatory.extend(['bus', 'gen', 'wind'])
        self._params.extend([
            'fn', 'Kp', 'nblade', 'ngen', 'npole', 'R', 'Tp', 'Ts', 'ngb', 'H',
            'rr', 'rs', 'xr', 'xs', 'xmu', 'Te', 'KV', 'pmax', 'pmin', 'qmax',
            'qmin', 'gammap', 'gammaq'
        ])
        self._powers.extend(['H', 'pmax', 'pmin', 'qmax', 'qmin'])
        self._service.extend([
            'u0', 'vref0', 'irq_min', 'ird_min', 'phi', 'fn', 'ird_max', 'x0',
            'irq_max', 'pi', 'irq_off', 'mva_mega', 'x1'
        ])
        self._states.extend(['theta_p', 'omega_m', 'ird', 'irq'])
        self._times.extend(['Tp', 'Te'])
        self._z.extend(['rs', 'xs', 'rr', 'xr', 'xmu'])
        self._ac.update({'bus': ['a', 'v']})
        self._data.update({
            'fn': 60,
            'rs': 0.01,
            'xmu': 3,
            'R': 35,
            'ngb': 0.011235,
            'gammap': 1,
            'npole': 4,
            'qmin': -0.6,
            'KV': 10,
            'xr': 0.08,
            'Te': 0.01,
            'pmin': 0,
            'Ts': 1,
            'Sn': 40,
            'wind': 0,
            'gen': 0,
            'rr': 0.01,
            'pmax': 1.0,
            'gammaq': 1,
            'Kp': 10,
            'xs': 0.1,
            'H': 2,
            'Tp': 3.0,
            'qmax': 0.6,
            'nblade': 3,
            'bus': 0,
            'ngen': 40,
            'u': 1
        })
        self._descr.update({
            'fn': 'Base frequency',
            'rs': 'Stator resistance',
            'xmu': 'Magnetizing reactance',
            'R': 'Rotor radius',
            'pmax': 'Maximum active power',
            'gammap': 'Active power generation ratio',
            'npole': 'Number of poles',
            'qmin': 'Minimum reactive power',
            'KV': 'Voltage control gain',
            'xr': 'Rotor reactance',
            'Te': 'Power control time constant',
            'pmin': 'Minimum reactive power',
            'Ts': 'Speed control time constant',
            'wind': 'Wind time series idx',
            'gen': 'Static generator idx',
            'rr': 'Rotor resistance',
            'ngb': 'Gear box ratio',
            'gammaq': 'Reactive power generation ratio',
            'Kp': 'Pitch control gain',
            'xs': 'Stator reactance',
            'H': 'Machine rotor and turbine inertia constant',
            'Tp': 'Pitch control time constant',
            'qmax': 'Maximum active power',
            'nblade': 'Number of blades',
            'bus': 'Bus idx',
            'ngen': 'Number of generators'
        })
        self._units.update({
            'fn': 'Hz',
            'rs': 'pu',
            'xmu': 'pu',
            'rr': 'pu',
            'R': 'm',
            'pmax': 'pu',
            'qmin': 'pu',
            'Kp': 'pu',
            'xs': 'pu',
            'qmax': 'pu',
            'H': 'MWs/MVA',
            'Tp': 's',
            'KV': 'pu',
            'Te': 's',
            'xr': 'pu',
            'pmin': 'pu'
        })
        self.calls.update({
            'init1': True,
            'gycall': True,
            'fxcall': True,
            'fcall': True,
            'gcall': True,
            'jac0': True
        })
        self._init()

    def servcall(self, dae):
        self.copy_data_ext('StaticGen', 'u', 'ugen', self.gen)
        self.copy_data_ext('Bus', 'Pg', 'p0', self.bus)
        self.copy_data_ext('Bus', 'Qg', 'q0', self.bus)
        self.copy_data_ext('Wind', 'vw', 'vw', self.wind)
        self.copy_data_ext('Wind', 'rho', 'rho', self.wind)
        self.copy_data_ext('Wind', 'Vwn', 'Vwn', self.wind)
        self.vref0 = dae.y[self.v]
        self.x0 = self.xmu + self.xs
        self.x1 = self.xmu + self.xr
        self.ird_min = -999
        self.ird_max = 999
        self.irq_min = -999
        self.irq_max = 999
        self.irq_off = zeros(self.n, 1)
        self.u0 = mul(self.u, self.ugen)
        # self.omega_ref0 = ones(self.n, 1)
        self.mva_mega = 100000000.0

    def init1(self, dae):
        """New initialization function"""
        self.servcall(dae)
        retval = True

        mva = self.system.mva
        self.p0 = mul(self.p0, self.gammap)
        self.q0 = mul(self.q0, self.gammaq)

        dae.y[self.vsd] = mul(dae.y[self.v], -sin(dae.y[self.a]))
        dae.y[self.vsq] = mul(dae.y[self.v], cos(dae.y[self.a]))

        rs = matrix(self.rs)
        rr = matrix(self.rr)
        xmu = matrix(self.xmu)
        x1 = matrix(self.xs) + xmu
        x2 = matrix(self.xr) + xmu
        Pg = matrix(self.p0)
        Qg = matrix(self.q0)
        Vc = dae.y[self.v]
        vsq = dae.y[self.vsq]
        vsd = dae.y[self.vsd]

        toSn = div(mva, self.Sn)  # to machine base
        toSb = self.Sn / mva  # to system base

        # rotor speed
        omega = 1 * (ageb(mva * Pg, self.Sn)) + \
            mul(0.5 + 0.5 * mul(Pg, toSn),
                aandb(agtb(Pg, 0), altb(mva * Pg, self.Sn))) + \
            0.5 * (aleb(mva * Pg, 0))

        slip = 1 - omega
        theta = mul(self.Kp, mround(1000 * (omega - 1)) / 1000)
        theta = mmax(theta, 0)

        # prepare for the iterations

        irq = mul(-x1, toSb, (2 * omega - 1), div(1, Vc), div(1, xmu),
                  div(1, omega))
        isd = zeros(*irq.size)
        isq = zeros(*irq.size)

        # obtain ird isd isq
        for i in range(self.n):
            A = sparse([[-rs[i], vsq[i]], [x1[i], -vsd[i]]])
            B = matrix([vsd[i] - xmu[i] * irq[i], Qg[i]])
            linsolve(A, B)
            isd[i] = B[0]
            isq[i] = B[1]
        ird = -div(vsq + mul(rs, isq) + mul(x1, isd), xmu)
        vrd = -mul(rr, ird) + mul(
            slip,
            mul(x2, irq) + mul(xmu, isq))  # todo: check x1 or x2
        vrq = -mul(rr, irq) - mul(slip, mul(x2, ird) + mul(xmu, isd))

        # main iterations
        for i in range(self.n):
            mis = ones(6, 1)
            rows = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5]
            cols = [0, 1, 3, 0, 1, 2, 2, 4, 3, 5, 0, 1, 2]

            x = matrix([isd[i], isq[i], ird[i], irq[i], vrd[i], vrq[i]])
            # vals = [-rs, x1, xmu, -x1, -rs, -xmu, -rr,
            #         -1, -rr, -1, vsd, vsq, -xmu * Vc / x1]

            vals = [
                -rs[i], x1[i], xmu[i], -x1[i], -rs[i], -xmu[i], -rr[i], -1,
                -rr[i], -1, vsd[i], vsq[i], -xmu[i] * Vc[i] / x1[i]
            ]
            jac0 = spmatrix(vals, rows, cols, (6, 6), 'd')
            iter = 0

            while max(abs(mis)) > self.system.tds.config.tol:
                if iter > 20:
                    logger.error(
                        'Initialization of DFIG <{}> failed.'.format(
                            self.name[i]))
                    retval = False
                    break

                mis[0] = -rs[i] * x[0] + x1[i] * x[1] + xmu[i] * x[3] - vsd[i]
                mis[1] = -rs[i] * x[1] - x1[i] * x[0] - xmu[i] * x[2] - vsq[i]
                mis[2] = -rr[i] * x[2] + slip[i] * (
                    x2[i] * x[3] + xmu[i] * x[1]) - x[4]
                mis[3] = -rr[i] * x[3] - slip[i] * (
                    x2[i] * x[2] + xmu[i] * x[0]) - x[5]
                mis[4] = vsd[i] * x[0] + vsq[i] * x[1] + x[4] * x[2] + \
                    x[5] * x[3] - Pg[i]
                mis[5] = -xmu[i] * Vc[i] * x[2] / x1[i] - \
                    Vc[i] * Vc[i] / x1[i] - Qg[i]

                rows = [2, 2, 3, 3, 4, 4, 4, 4]
                cols = [1, 3, 0, 2, 2, 3, 4, 5]
                vals = [
                    slip[i] * xmu[i], slip[i] * x2[i], -slip[i] * xmu[i],
                    -slip[i] * x2[i], x[4], x[5], x[2], x[3]
                ]

                jac = jac0 + spmatrix(vals, rows, cols, (6, 6), 'd')

                linsolve(jac, mis)

                x -= mis
                iter += 1

            isd[i] = x[0]
            isq[i] = x[1]
            ird[i] = x[2]
            irq[i] = x[3]
            vrd[i] = x[4]
            vrq[i] = x[5]

        dae.x[self.ird] = mul(self.u0, ird)
        dae.x[self.irq] = mul(self.u0, irq)
        dae.y[self.isd] = isd
        dae.y[self.isq] = isq
        dae.y[self.vrd] = vrd
        dae.y[self.vrq] = vrq

        dae.x[self.omega_m] = mul(self.u0, omega)
        dae.x[self.theta_p] = mul(self.u0, theta)
        dae.y[self.pwa] = mmax(mmin(2 * dae.x[self.omega_m] - 1, 1), 0)

        self.vref0 = mul(
            aneb(self.KV, 0), Vc - div(ird + div(Vc, xmu), self.KV))
        dae.y[self.vref] = self.vref0
        k = mul(div(x1, Vc, xmu, omega), toSb)

        self.irq_off = -mul(k, mmax(mmin(2 * omega - 1, 1), 0)) - irq

        # electrical torque in pu
        te = mul(
            xmu,
            mul(dae.x[self.irq], dae.y[self.isd]) - mul(
                dae.x[self.ird], dae.y[self.isq]))

        for i in range(self.n):
            if te[i] < 0:
                logger.error(
                    'Pe < 0 on bus <{}>. Wind speed initialize failed.'.
                    format(self.bus[i]))
                retval = False

        # wind power in pu
        pw = mul(te, omega)
        dae.y[self.pw] = pw

        # wind speed initialization loop

        R = 4 * pi * self.system.freq * mul(self.R, self.ngb,
                                            div(1, self.npole))
        AA = pi * self.R**2
        vw = 0.9 * self.Vwn

        for i in range(self.n):
            mis = 1
            iter = 0
            while abs(mis) > self.system.tds.config.tol:
                if iter > 50:
                    logger.error(
                        'Wind <{}> init failed. '
                        'Try increasing the nominal wind speed.'.
                        format(self.wind[i]))
                    retval = False
                    break

                pw_iter, jac = self.windpower(self.ngen[i], self.rho[i], vw[i],
                                              AA[i], R[i], omega[i], theta[i])

                mis = pw_iter - pw[i]
                inc = -mis / jac[1]
                vw[i] += inc
                iter += 1

        # set wind speed
        dae.x[self.vw] = div(vw, self.Vwn)

        lamb = div(omega, vw, div(1, R))
        ilamb = div(1,
                    (div(1, lamb + 0.08 * theta) - div(0.035, theta**3 + 1)))
        cp = 0.22 * mul(
            div(116, ilamb) - 0.4 * theta - 5, exp(div(-12.5, ilamb)))

        dae.y[self.lamb] = lamb
        dae.y[self.ilamb] = ilamb
        dae.y[self.cp] = cp

        self.system.rmgen(self.gen)

        if not retval:
            logger.error('DFIG initialization failed')

        return retval

    def windpower(self, ngen, rho, vw, Ar, R, omega, theta, derivative=False):
        mva_mega = self.system.mva * 1e6
        lamb = omega * R / vw
        ilamb = 1 / (1 / (lamb + 0.08 * theta) - 0.035 / (theta**3 + 1))
        cp = 0.22 * (116 / ilamb - 0.4 * theta - 5) * exp(-12.5 / ilamb)
        pw = 0.5 * ngen * rho * cp * Ar * vw**3 / mva_mega

        a1 = exp(-12.5 / ilamb)
        a2 = (lamb + 0.08 * theta)**2
        a3 = 116. / ilamb - 0.4 * theta - 5
        a4 = -9.28 / (lamb + 0.08 * theta) ** 2 + \
            12.180 * theta * theta / (theta ** 3 + 1) ** 2 - 0.4
        a5 = 1.000 / (lamb + 0.08 * theta) ** 2 - \
            1.3125 * theta * theta / (theta ** 3 + 1) ** 2

        jac = ones(1, 3)
        jac[0] = ngen * R * a1 * rho * vw * vw * Ar * (
            -12.760 + 1.3750 * a3) / a2 / mva_mega
        jac[1] = ngen * (omega * R * (12.760 - 1.3750 * a3) / a2 +
                         0.330 * a3 * vw) * vw * Ar * rho * a1 / mva_mega
        jac[2] = ngen * 0.110 * rho * (
            a4 + a3 * a5) * a1 * Ar * vw**3 / mva_mega

        return pw, jac

    @property
    def phi(self):
        deg1 = pi / 180
        dae = self.system.dae
        above = agtb(dae.x[self.omega_m], 1)
        phi_degree_step = mfloor((dae.x[self.omega_m] - 1) / deg1) * deg1
        return mul(phi_degree_step, above)

    def gcall(self, dae):
        dae.g[self.isd] = -dae.y[self.vsd] + mul(
            dae.x[self.irq], self.xmu) + mul(dae.y[self.isq], self.x0) - mul(
                dae.y[self.isd], self.rs)
        dae.g[self.isq] = -dae.y[self.vsq] - mul(
            dae.x[self.ird], self.xmu) - mul(dae.y[self.isd], self.x0) - mul(
                dae.y[self.isq], self.rs)
        dae.g[self.vrd] = -dae.y[self.vrd] + mul(
            1 - dae.x[self.omega_m],
            mul(dae.x[self.irq], self.x1) + mul(
                dae.y[self.isq], self.xmu)) - mul(dae.x[self.ird], self.rr)
        dae.g[self.vrq] = -dae.y[self.vrq] - mul(
            dae.x[self.irq], self.rr) - mul(
                1 - dae.x[self.omega_m],
                mul(dae.x[self.ird], self.x1) + mul(dae.y[self.isd], self.xmu))
        dae.g[self.vsd] = -dae.y[self.vsd] - mul(dae.y[self.v],
                                                 sin(dae.y[self.a]))
        dae.g[self.vsq] = -dae.y[self.vsq] + mul(dae.y[self.v],
                                                 cos(dae.y[self.a]))
        dae.g[self.vref] = self.vref0 - dae.y[self.vref]
        dae.g[self.pwa] = mmax(mmin(2 * dae.x[self.omega_m] - 1, 1),
                               0) - dae.y[self.pwa]

        dae.hard_limit(self.pwa, 0, 1)
        dae.g[self.pw] = -dae.y[self.pw] + mul(
            0.5, dae.y[self.cp], self.ngen, pi, self.rho, (self.R)**2,
            (self.Vwn)**3, div(1, self.mva_mega), (dae.x[self.vw])**3)
        dae.g[self.cp] = -dae.y[self.cp] + mul(
            -1.1 + mul(25.52, div(1, dae.y[self.ilamb])) + mul(
                -0.08800000000000001, dae.x[self.theta_p]),
            exp(mul(-12.5, div(1, dae.y[self.ilamb]))))
        dae.g[self.lamb] = -dae.y[self.lamb] + mul(
            4, self.R, self.fn, self.ngb, dae.x[self.omega_m], pi,
            div(1, self.Vwn), div(1, self.npole), div(1, dae.x[self.vw]))
        dae.g[self.ilamb] = div(
            1,
            div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) + mul(
                -0.035, div(1, 1 +
                            (dae.x[self.theta_p])**3))) - dae.y[self.ilamb]
        dae.g += spmatrix(
            mul(
                self.u0, -mul(dae.x[self.ird], dae.y[self.vrd]) - mul(
                    dae.x[self.irq], dae.y[self.vrq]) - mul(
                        dae.y[self.isd], dae.y[self.vsd]) - mul(
                            dae.y[self.isq], dae.y[self.vsq])), self.a,
            [0] * self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(
            mul(
                self.u0,
                mul((dae.y[self.v])**2, div(1, self.x0)) + mul(
                    dae.x[self.ird], dae.y[self.v], self.xmu, div(
                        1, self.x0))), self.v, [0] * self.n, (dae.m, 1), 'd')

    def fcall(self, dae):
        toSb = self.Sn / self.system.mva
        omega = not0(dae.x[self.omega_m])
        dae.f[self.theta_p] = mul(
            div(1, self.Tp), -dae.x[self.theta_p] + mul(
                self.Kp, self.phi, -1 + dae.x[self.omega_m]))
        dae.anti_windup(self.theta_p, 0, pi)

        dae.f[self.omega_m] = mul(
            0.5, div(1, self.H),
            mul(dae.y[self.pw], div(1, dae.x[self.omega_m])) - mul(
                self.xmu,
                mul(dae.x[self.irq], dae.y[self.isd]) - mul(
                    dae.x[self.ird], dae.y[self.isq])))

        dae.f[self.ird] = mul(
            div(1, self.Ts),
            -dae.x[self.ird] + mul(self.KV, dae.y[self.v] - dae.y[self.vref]) -
            mul(dae.y[self.v], div(1, self.xmu)))
        dae.anti_windup(self.ird, self.ird_min, self.irq_max)
        k = mul(self.x0, toSb, div(1, dae.y[self.v]), div(1, self.xmu),
                div(1, omega))
        dae.f[self.irq] = mul(
            div(1, self.Te),
            -dae.x[self.irq] - self.irq_off - mul(dae.y[self.pwa], k))
        dae.anti_windup(self.irq, self.irq_min, self.irq_max)

    def gycall(self, dae):
        dae.add_jac(Gy, mul(self.xmu, 1 - dae.x[self.omega_m]), self.vrd,
                    self.isq)
        dae.add_jac(Gy, -mul(self.xmu, 1 - dae.x[self.omega_m]), self.vrq,
                    self.isd)
        dae.add_jac(Gy, -sin(dae.y[self.a]), self.vsd, self.v)
        dae.add_jac(Gy, -mul(dae.y[self.v], cos(dae.y[self.a])), self.vsd,
                    self.a)
        dae.add_jac(Gy, cos(dae.y[self.a]), self.vsq, self.v)
        dae.add_jac(Gy, -mul(dae.y[self.v], sin(dae.y[self.a])), self.vsq,
                    self.a)
        dae.add_jac(
            Gy,
            mul(0.5, self.ngen, pi, self.rho, (self.R)**2, (self.Vwn)**3,
                div(1, self.mva_mega), (dae.x[self.vw])**3), self.pw, self.cp)
        dae.add_jac(
            Gy,
            mul(-25.52, (dae.y[self.ilamb])**-2,
                exp(mul(-12.5, div(1, dae.y[self.ilamb])))) + mul(
                    12.5, (dae.y[self.ilamb])**-2,
                    -1.1 + mul(25.52, div(1, dae.y[self.ilamb])) + mul(
                        -0.088, dae.x[self.theta_p]),
                    exp(mul(-12.5, div(1, dae.y[self.ilamb])))), self.cp,
            self.ilamb)
        dae.add_jac(
            Gy,
            mul((dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))**-2,
                (div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) +
                 mul(-0.035, div(1, 1 + (dae.x[self.theta_p])**3)))**-2),
            self.ilamb, self.lamb)
        dae.add_jac(Gy, -mul(dae.y[self.isd], self.u0), self.a, self.vsd)
        dae.add_jac(Gy, -mul(dae.x[self.irq], self.u0), self.a, self.vrq)
        dae.add_jac(Gy, -mul(self.u0, dae.y[self.vsq]), self.a, self.isq)
        dae.add_jac(Gy, -mul(dae.x[self.ird], self.u0), self.a, self.vrd)
        dae.add_jac(Gy, -mul(dae.y[self.isq], self.u0), self.a, self.vsq)
        dae.add_jac(Gy, -mul(self.u0, dae.y[self.vsd]), self.a, self.isd)
        dae.add_jac(
            Gy,
            mul(
                self.u0,
                mul(2, dae.y[self.v], div(1, self.x0)) + mul(
                    dae.x[self.ird], self.xmu, div(1, self.x0))), self.v,
            self.v)

    def fxcall(self, dae):
        omega = not0(dae.x[self.omega_m])
        toSb = div(self.Sn, self.system.mva)
        dae.add_jac(Gx, mul(self.x1, 1 - dae.x[self.omega_m]), self.vrd,
                    self.irq)
        dae.add_jac(
            Gx,
            -mul(dae.x[self.irq], self.x1) - mul(dae.y[self.isq], self.xmu),
            self.vrd, self.omega_m)
        dae.add_jac(
            Gx,
            mul(dae.x[self.ird], self.x1) + mul(dae.y[self.isd], self.xmu),
            self.vrq, self.omega_m)
        dae.add_jac(Gx, -mul(self.x1, 1 - dae.x[self.omega_m]), self.vrq,
                    self.ird)
        dae.add_jac(
            Gx,
            mul(1.5, dae.y[self.cp],
                self.ngen, pi, self.rho, (self.R)**2, (self.Vwn)**3,
                div(1, self.mva_mega), (dae.x[self.vw])**2), self.pw, self.vw)
        dae.add_jac(
            Gx,
            mul(-0.088, exp(
                mul(-12.5, div(1, dae.y[self.ilamb])))), self.cp, self.theta_p)
        dae.add_jac(
            Gx,
            mul(-4, self.R, self.fn, self.ngb, dae.x[self.omega_m], pi,
                div(1, self.Vwn), div(1, self.npole), (dae.x[self.vw])**-2),
            self.lamb, self.vw)
        dae.add_jac(
            Gx,
            mul(4, self.R, self.fn, self.ngb, pi, div(1, self.Vwn),
                div(1, self.npole), div(1, dae.x[self.vw])), self.lamb,
            self.omega_m)
        dae.add_jac(
            Gx,
            mul((div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) +
                 mul(-0.035, div(1, 1 + (dae.x[self.theta_p]) ** 3))) ** -2,
                mul(0.08, (dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))
                    ** -2) + mul(-0.105, (dae.x[self.theta_p]) ** 2,
                                 (1 + (dae.x[self.theta_p]) ** 3) ** -2)),
            self.ilamb,
            self.theta_p)
        dae.add_jac(Gx, -mul(self.u0, dae.y[self.vrq]), self.a, self.irq)
        dae.add_jac(Gx, -mul(self.u0, dae.y[self.vrd]), self.a, self.ird)
        dae.add_jac(Gx, mul(self.u0, dae.y[self.v], self.xmu, div(1, self.x0)),
                    self.v, self.ird)

        dae.add_jac(
            Fx,
            mul(dae.y[self.pwa], self.x0, toSb, div(1, self.Te),
                (dae.x[self.omega_m])**-2, div(1, dae.y[self.v]),
                div(1, self.xmu)), self.irq, self.omega_m)
        dae.add_jac(
            Fy,
            mul(dae.y[self.pwa], self.x0, toSb, div(1, self.Te), div(1, omega),
                (dae.y[self.v])**-2, div(1, self.xmu)), self.irq, self.v)
        dae.add_jac(
            Fy, -mul(self.x0, toSb, div(1, self.Te), div(1, omega),
                     div(1, dae.y[self.v]), div(1, self.xmu)), self.irq,
            self.pwa)

        dae.add_jac(Fx, mul(0.5, dae.y[self.isq], self.xmu, div(1, self.H)),
                    self.omega_m, self.ird)
        dae.add_jac(
            Fx,
            mul(-0.5, dae.y[self.pw], div(1, self.H), (dae.x[self.omega_m])
                ** -2), self.omega_m, self.omega_m)
        dae.add_jac(Fx, mul(-0.5, dae.y[self.isd], self.xmu, div(1, self.H)),
                    self.omega_m, self.irq)
        dae.add_jac(Fy, mul(0.5, div(1, self.H), div(1, dae.x[self.omega_m])),
                    self.omega_m, self.pw)
        dae.add_jac(Fy, mul(-0.5, dae.x[self.irq], self.xmu, div(1, self.H)),
                    self.omega_m, self.isd)
        dae.add_jac(Fy, mul(0.5, dae.x[self.ird], self.xmu, div(1, self.H)),
                    self.omega_m, self.isq)

    def jac0(self, dae):
        dae.add_jac(Gy0, -1, self.isd, self.vsd)
        dae.add_jac(Gy0, -self.rs, self.isd, self.isd)
        dae.add_jac(Gy0, self.x0, self.isd, self.isq)
        dae.add_jac(Gy0, -1, self.isq, self.vsq)
        dae.add_jac(Gy0, -self.x0, self.isq, self.isd)
        dae.add_jac(Gy0, -self.rs, self.isq, self.isq)
        dae.add_jac(Gy0, -1, self.vrd, self.vrd)
        dae.add_jac(Gy0, -1, self.vrq, self.vrq)
        dae.add_jac(Gy0, -1, self.vsd, self.vsd)
        dae.add_jac(Gy0, -1, self.vsq, self.vsq)
        dae.add_jac(Gy0, -1, self.vref, self.vref)
        dae.add_jac(Gy0, -1, self.pwa, self.pwa)
        dae.add_jac(Gy0, -1, self.pw, self.pw)
        dae.add_jac(Gy0, -1, self.cp, self.cp)
        dae.add_jac(Gy0, -1, self.lamb, self.lamb)
        dae.add_jac(Gy0, -1, self.ilamb, self.ilamb)
        dae.add_jac(Gx0, self.xmu, self.isd, self.irq)
        dae.add_jac(Gx0, -self.xmu, self.isq, self.ird)
        dae.add_jac(Gx0, -self.rr, self.vrd, self.ird)
        dae.add_jac(Gx0, -self.rr, self.vrq, self.irq)
        dae.add_jac(Gx0, 2, self.pwa, self.omega_m)
        dae.add_jac(Fx0, -div(1, self.Tp), self.theta_p, self.theta_p)
        dae.add_jac(Fx0, mul(self.Kp, self.phi, div(1, self.Tp)), self.theta_p,
                    self.omega_m)
        dae.add_jac(Fx0, -div(1, self.Ts), self.ird, self.ird)
        dae.add_jac(Fx0, -div(1, self.Te), self.irq, self.irq)
        dae.add_jac(Fy0, -mul(self.KV, div(1, self.Ts)), self.ird, self.vref)
        dae.add_jac(Fy0, mul(div(1, self.Ts), self.KV - div(1, self.xmu)),
                    self.ird, self.v)
        dae.add_jac(Gy0, 1e-6, self.isd, self.isd)
        dae.add_jac(Gy0, 1e-6, self.isq, self.isq)
        dae.add_jac(Gy0, 1e-6, self.vrd, self.vrd)
        dae.add_jac(Gy0, 1e-6, self.vrq, self.vrq)
        dae.add_jac(Gy0, 1e-6, self.vsd, self.vsd)
        dae.add_jac(Gy0, 1e-6, self.vsq, self.vsq)
        dae.add_jac(Gy0, 1e-6, self.vref, self.vref)
        dae.add_jac(Gy0, 1e-6, self.pwa, self.pwa)
        dae.add_jac(Gy0, 1e-6, self.pw, self.pw)
        dae.add_jac(Gy0, 1e-6, self.cp, self.cp)
        dae.add_jac(Gy0, 1e-6, self.lamb, self.lamb)
        dae.add_jac(Gy0, 1e-6, self.ilamb, self.ilamb)
