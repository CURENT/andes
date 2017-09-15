from cvxopt import matrix, spmatrix, sparse
from cvxopt import mul, div, sin, cos, exp
from ..consts import *
from .base import ModelBase

from cvxopt.klu import linsolve
from ..utils.math import zeros, ones, mmax, mmin, not0, agtb, mfloor


class WTG3(ModelBase):
    """Wind turbine type III"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'WTG'
        self._name = 'WTG3'
        self._algebs.extend(['isd', 'isq', 'vrd', 'vrq', 'vsd', 'vsq', 'vref', 'pwa', 'pw', 'cp', 'lamb', 'ilamb'])
        self._fnamex.extend(['\\theta_p', '\\omega_m', 'I_{r, d}', 'I_{r, q}'])
        self._fnamey.extend(['I_{s, d}', 'I_{s, q}', 'V_{r, d}', 'V_{r, q}', 'V_{s, d}', 'V_{s, q}', 'V_{ref}', 'P_{\\omega a}', 'P_w', 'c_p', '\\lambda', '\\frac{1}{\\lambda}', '\\omega_{ref}'])
        self._mandatory.extend(['bus', 'gen', 'wind'])
        self._params.extend(['fn', 'Kp', 'nblade', 'ngen', 'npole', 'R', 'Tp', 'Ts', 'ngb', 'H', 'rr', 'rs', 'xr', 'xs', 'xmu', 'Te', 'KV', 'pmax', 'pmin', 'qmax', 'qmin', 'gammap', 'gammaq'])
        self._powers.extend(['H', 'pmax', 'pmin', 'qmax', 'qmin'])
        self._service.extend(['u0', 'vref0', 'irq_min', 'ird_min', 'phi', 'fn', 'ird_max', 'x0', 'irq_max', 'pi', 'irq_off', 'mva_mega', 'x1'])
        self._states.extend(['theta_p', 'omega_m', 'ird', 'irq'])
        self._times.extend(['Tp', 'Te'])
        self._z.extend(['rs', 'xs', 'rr', 'xr', 'xmu'])
        self._ac.update({'bus': ['a', 'v']})
        self._data.update({'fn': 60, 'rs': 0.01, 'xmu': 3, 'R': 35, 'ngb': 0.011235, 'gammap': 1, 'npole': 4, 'qmin': -0.6, 'KV': 10, 'xr': 0.08, 'Te': 0.01, 'pmin': 0, 'Ts': 1, 'Sn': 40, 'wind': 0, 'gen': 0, 'rr': 0.01, 'pmax': 1.0, 'gammaq': 1, 'Kp': 10, 'xs': 0.1, 'H': 2, 'Tp': 3.0, 'qmax': 0.6, 'nblade': 3, 'bus': 0, 'ngen': 40, 'u': 1})
        self._descr.update({'fn': 'Base frequency', 'rs': 'Stator resistance', 'xmu': 'Magnetizing reactance', 'R': 'Rotor radius', 'pmax': 'Maximum active power', 'gammap': 'Active power generation ratio', 'npole': 'Number of poles', 'qmin': 'Minimum reactive power', 'KV': 'Voltage control gain', 'xr': 'Rotor reactance', 'Te': 'Power control time constant', 'pmin': 'Minimum reactive power', 'Ts': 'Speed control time constant', 'wind': 'Wind time series idx', 'gen': 'Static generator idx', 'rr': 'Rotor resistance', 'ngb': 'Gear box ratio', 'gammaq': 'Reactive power generation ratio', 'Kp': 'Pitch control gain', 'xs': 'Stator reactance', 'H': 'Machine rotor and turbine inertia constant', 'Tp': 'Pitch control time constant', 'qmax': 'Maximum active power', 'nblade': 'Number of blades', 'bus': 'Bus idx', 'ngen': 'Number of generators'})
        self._units.update({'fn': 'Hz', 'rs': 'pu', 'xmu': 'pu', 'rr': 'pu', 'R': 'm', 'pmax': 'pu', 'qmin': 'pu', 'Kp': 'pu', 'xs': 'pu', 'qmax': 'pu', 'H': 'MWs/MVA', 'Tp': 's', 'KV': 'pu', 'Te': 's', 'xr': 'pu', 'pmin': 'pu'})
        self.calls.update({'init1': True, 'gycall': True, 'fxcall': True, 'fcall': True, 'gcall': True, 'jac0': True})
        self._inst_meta()

    def servcall(self, dae):
        self.copy_param('StaticGen', 'u', 'ugen', self.gen)
        self.copy_param('Bus', 'Pg', 'p0', self.bus)
        self.copy_param('Bus', 'Qg', 'q0', self.bus)
        self.copy_param('Wind', 'vw', 'vw', self.wind)
        self.copy_param('Wind', 'rho', 'rho', self.wind)
        self.copy_param('Wind', 'Vwn', 'Vwn', self.wind)
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
        self.servcall(dae)
        dae.x[self.omega_m] = 1
        dae.x[self.vw] = 1
        dae.y[self.lamb] = mul(self.R, dae.x[self.omega_m], div(1, dae.x[self.vw]))
        dae.y[self.vsd] = mul(dae.y[self.v], -sin(dae.y[self.a]))
        dae.y[self.vsq] = mul(dae.y[self.v], cos(dae.y[self.a]))
        self.p0 = mul(self.p0, self.gammap)
        self.q0 = mul(self.q0, self.gammaq)
        mva = self.system.Settings.mva
        retval = True
        for i in range(self.n):
            rs = self.rs[i]
            rr = self.rr[i]
            xmu = self.xmu[i]
            x1 = self.xs[i] + xmu
            x2 = self.xr[i] + xmu
            Pg = self.p0[i]
            Qg = self.q0[i]
            Vc = dae.y[self.v[i]]
            vsq = dae.y[self.vsq][i]
            vsd = dae.y[self.vsd][i]

            # base convert constants
            toSn = mva / self.Sn[i]
            toSb = self.Sn[i] / mva

            # rotor speed
            if Pg * mva > self.Sn[i]:
                omega = 1
            elif Pg > 0 and (Pg * mva < self.Sn[i]):
                omega = 0.5 * Pg * toSn + 0.5
            else:
                omega = 0.5

            slip = 1 - omega

            irq = -x1 * toSb * (2 * omega - 1) / Vc / xmu / omega
            A = sparse([[-rs, vsq], [x1, -vsd]])
            B = matrix([vsd - xmu * irq, Qg])
            linsolve(A, B)

            isd = B[0]
            isq = B[1]
            ird = -(vsq + rs * isq + x1 * isd) / xmu
            vrd = - rr * ird + slip * (x2 * irq + xmu * isq) # todo: check x1 or x2
            vrq = - rr * irq - slip * (x2 * ird + xmu * isd)

            mis = ones(6, 1)
            x = matrix([isd, isq, ird, irq, vrd, vrq])

            rows = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5]
            cols = [0, 1, 3, 0, 1, 2, 2, 4, 3, 5, 0, 1, 2]
            vals = [-rs, x1, xmu, -x1, -rs, -xmu, -rr, -1, -rr, -1, vsd, vsq, -xmu * Vc / x1]
            jac0 = spmatrix(vals, rows, cols, (6, 6), 'd')

            iter = 0

            while max(abs(mis)) > self.system.TDS.tol:
                if iter > 20:
                    self.message('Initialization of DFIG <{}> failed.'.format(self.name[i]), ERROR)
                    retval = False
                    break

                mis[0] = -rs * x[0] + x1 * x[1] + xmu * x[3] - vsd
                mis[1] = -rs * x[1] - x1 * x[0] - xmu * x[2] - vsq
                mis[2] = -rr * x[2] + slip * (x2 * x[3] + xmu * x[1]) - x[4]
                mis[3] = -rr * x[3] - slip * (x2 * x[2] + xmu * x[0]) - x[5]
                mis[4] = vsd * x[0] + vsq * x[1] + x[4] * x[2] + x[5] * x[3] - Pg
                mis[5] = -xmu * Vc * x[2] / x1 - Vc * Vc / x1 - Qg

                rows = [2, 2, 3, 3, 4, 4, 4, 4]
                cols = [1, 3, 0, 2, 2, 3, 4, 5]
                vals = [slip * xmu, slip * x2, -slip * xmu, -slip * x2, x[4], x[5], x[2], x[3]]

                jac = jac0 + spmatrix(vals, rows, cols, (6, 6), 'd')

                linsolve(jac, mis)

                x -= mis
                iter += 1

            isd = x[0]
            isq = x[1]
            ird = x[2]
            irq = x[3]
            vrd = x[4]
            vrq = x[5]

            # check limits
            pass

            theta = self.Kp[i] * round(1000 * (omega - 1)) / 1000
            theta = max(theta, 0)

            # states
            dae.x[self.ird[i]] = ird
            dae.x[self.irq[i]] = irq
            dae.x[self.omega_m[i]] = omega
            dae.x[self.theta_p[i]] = theta
            dae.y[self.isd[i]] = isd
            dae.y[self.isq[i]] = isq
            dae.y[self.vrd[i]] = vrd
            dae.y[self.vrq[i]] = vrq
            # dae.y[self.omega_ref[i]] = omega

            # self.omega_ref0[i] = omega

            # voltage control
            if self.KV[i] == 0:
                self.vref0[i] = 0
            else:
                self.vref0[i] = Vc - (ird + Vc/xmu) / self.KV[i]

            dae.y[self.vref[i]] = self.vref0[i]

            # k = x1 * toSb / Vc / xmu
            k = x1 / Vc / xmu /omega * toSb
            self.irq_off[i] = -k *max(min(2*omega - 1, 1), 0) - irq

            # electrical torque in pu
            # te = xmu * (irq * isd - ird * isq)
            te = -xmu * Vc * irq / x1

            if te < 0:
                self.message(
                    'Electric power is negative at bus <{}>. Wind speed initialize failed.'.format(self.bus[i]), ERROR)
                retval = False

            # wind power in pu
            pw = te * omega
            dae.y[self.pw[i]] = pw

            # wind speed
            mis = 1
            iter = 0

            R = 4 * pi * self.system.Settings.freq * self.R[i] * self.ngb[i] / self.npole[i]
            AA = pi * self.R[i] ** 2
            vw = 0.9 * self.Vwn[i]

            while abs(mis) > self.system.TDS.tol:
                if iter > 50:
                    self.message(
                        'Initialization of wind <{}> failed. Try increasing the nominal wind speed.'.format(self.wind[i]))
                    retval = False
                    break

                pw_iter, jac = self.windpower(self.ngen[i], self.rho[i], vw, AA, R, omega, theta)

                mis = pw_iter - pw
                inc = -mis / jac[1]
                vw += inc
                iter += 1
            # set wind speed
            dae.x[self.vw[i]] = vw / self.Vwn[i]

            lamb = omega * R / vw
            ilamb = 1 / (1 / (lamb + 0.08 * theta) - 0.035 / (theta ** 3 + 1))
            cp = 0.22 * (116 / ilamb - 0.4 * theta - 5) * exp(-12.5 / ilamb)

            dae.y[self.lamb[i]] = lamb
            dae.y[self.ilamb[i]] = ilamb
            dae.y[self.cp[i]] = cp

        # remove static gen
        self.system.rmgen(self.gen)

        dae.x[self.ird] = mul(self.u0, dae.x[self.ird])
        dae.x[self.irq] = mul(self.u0, dae.x[self.irq])
        dae.x[self.omega_m] = mul(self.u0, dae.x[self.omega_m])
        dae.x[self.theta_p] = mul(self.u0, dae.x[self.theta_p])
        dae.y[self.pwa] = mmax(mmin(2 * dae.x[self.omega_m] - 1, 1), 0)

        if not retval:
            self.message('DFIG initialization failed', ERROR)

        return retval

    def windpower(self, ngen, rho, vw, Ar, R, omega, theta, derivative=False):
        mva_mega = self.system.Settings.mva * 1e6
        lamb = omega * R / vw
        ilamb = 1 / (1 / (lamb + 0.08 * theta) - 0.035 / (theta ** 3 + 1))
        cp = 0.22 * (116 / ilamb - 0.4 * theta - 5) * exp(-12.5 / ilamb)
        pw = 0.5 * ngen * rho * cp * Ar * vw ** 3 / mva_mega

        a1 = exp(-12.5 / ilamb)
        a2 = (lamb + 0.08 * theta) ** 2
        a3 = 116. / ilamb - 0.4 * theta - 5
        a4 = -9.28 / (lamb + 0.08 * theta) ** 2 + \
             12.180 * theta * theta / (theta ** 3 + 1) ** 2 - 0.4
        a5 = 1.000 / (lamb + 0.08 * theta) ** 2 - \
             1.3125 * theta * theta / (theta ** 3 + 1) ** 2

        jac = ones(1, 3)
        jac[0] = ngen * R * a1 * rho * vw * vw * Ar * (-12.760 + 1.3750 * a3) / a2 / mva_mega
        jac[1] = ngen * (omega * R * (12.760 - 1.3750 * a3) / a2 + 0.330 * a3 * vw) * vw * Ar * rho * a1 / mva_mega
        jac[2] = ngen * 0.110 * rho * (a4 + a3 * a5) * a1 * Ar * vw ** 3 / mva_mega

        return pw, jac

    @property
    def phi(self):
        deg1 = pi / 180
        dae = self.system.DAE
        above = agtb(dae.x[self.omega_m], 1)
        phi_degree_step = mfloor((dae.x[self.omega_m] - 1)/deg1) * deg1
        return mul(phi_degree_step, above)

    def gcall(self, dae):
        toSb = div(self.Sn, self.system.Settings.mva)
        dae.g[self.isd] = -dae.y[self.vsd] + mul(dae.x[self.irq], self.xmu) + mul(dae.y[self.isq], self.x0) - mul(dae.y[self.isd], self.rs)
        dae.g[self.isq] = -dae.y[self.vsq] - mul(dae.x[self.ird], self.xmu) - mul(dae.y[self.isd], self.x0) - mul(dae.y[self.isq], self.rs)
        dae.g[self.vrd] = -dae.y[self.vrd] + mul(1 - dae.x[self.omega_m], mul(dae.x[self.irq], self.x1) + mul(dae.y[self.isq], self.xmu)) - mul(dae.x[self.ird], self.rr)
        dae.g[self.vrq] = -dae.y[self.vrq] - mul(dae.x[self.irq], self.rr) - mul(1 - dae.x[self.omega_m], mul(dae.x[self.ird], self.x1) + mul(dae.y[self.isd], self.xmu))
        dae.g[self.vsd] = -dae.y[self.vsd] - mul(dae.y[self.v], sin(dae.y[self.a]))
        dae.g[self.vsq] = -dae.y[self.vsq] + mul(dae.y[self.v], cos(dae.y[self.a]))
        dae.g[self.vref] = self.vref0 - dae.y[self.vref]
        # dae.g[self.pwa] = mul(2*dae.x[self.omega_m] - 1, toSb) - dae.y[self.pwa]
        dae.g[self.pwa] = mmax(mmin(2 * dae.x[self.omega_m] - 1, 1), 0) - dae.y[self.pwa]


        dae.hard_limit(self.pwa, 0, 1)
        dae.g[self.pw] = -dae.y[self.pw] + mul(0.5, dae.y[self.cp], self.ngen, pi, self.rho, (self.R)**2, (self.Vwn)**3, div(1, self.mva_mega), (dae.x[self.vw])**3)
        dae.g[self.cp] = -dae.y[self.cp] + mul(-1.1 + mul(25.52, div(1, dae.y[self.ilamb])) + mul(-0.08800000000000001, dae.x[self.theta_p]), exp(mul(-12.5, div(1, dae.y[self.ilamb]))))
        dae.g[self.lamb] = -dae.y[self.lamb] + mul(4, self.R, self.fn, self.ngb, dae.x[self.omega_m], pi, div(1, self.Vwn), div(1, self.npole), div(1, dae.x[self.vw]))
        dae.g[self.ilamb] = div(1, div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) + mul(-0.035, div(1, 1 + (dae.x[self.theta_p])**3))) - dae.y[self.ilamb]
        dae.g += spmatrix(mul(self.u0, -mul(dae.x[self.ird], dae.y[self.vrd]) - mul(dae.x[self.irq], dae.y[self.vrq]) - mul(dae.y[self.isd], dae.y[self.vsd]) - mul(dae.y[self.isq], dae.y[self.vsq])), self.a, [0]*self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(mul(self.u0, mul((dae.y[self.v])**2, div(1, self.x0)) + mul(dae.x[self.ird], dae.y[self.v], self.xmu, div(1, self.x0))), self.v, [0]*self.n, (dae.m, 1), 'd')

    def fcall(self, dae):
        toSb = self.Sn / self.system.Settings.mva
        omega = not0(dae.x[self.omega_m])
        dae.f[self.theta_p] = mul(div(1, self.Tp), -dae.x[self.theta_p] + mul(self.Kp, self.phi, -1 + dae.x[self.omega_m]))
        dae.anti_windup(self.theta_p, 0, pi)
        dae.f[self.omega_m] = mul(0.5, div(1, self.H), mul(dae.y[self.pw], div(1, omega)) + mul(dae.x[self.irq], dae.y[self.v], self.xmu, div(1, self.x0)))
        dae.f[self.ird] = mul(div(1, self.Ts), -dae.x[self.ird] + mul(self.KV, dae.y[self.v] - dae.y[self.vref]) - mul(dae.y[self.v], div(1, self.xmu)))
        dae.anti_windup(self.ird, self.ird_min, self.irq_max)
        k = mul(self.x0, toSb, div(1, dae.y[self.v]), div(1, self.xmu), div(1, omega))
        dae.f[self.irq] = mul(div(1, self.Te), -dae.x[self.irq] - self.irq_off - mul(dae.y[self.pwa], k))
        dae.anti_windup(self.irq, self.irq_min, self.irq_max)

    def gycall(self, dae):
        dae.add_jac(Gy, mul(self.xmu, 1 - dae.x[self.omega_m]), self.vrd, self.isq)
        dae.add_jac(Gy, - mul(self.xmu, 1 - dae.x[self.omega_m]), self.vrq, self.isd)
        dae.add_jac(Gy, - sin(dae.y[self.a]), self.vsd, self.v)
        dae.add_jac(Gy, - mul(dae.y[self.v], cos(dae.y[self.a])), self.vsd, self.a)
        dae.add_jac(Gy, cos(dae.y[self.a]), self.vsq, self.v)
        dae.add_jac(Gy, - mul(dae.y[self.v], sin(dae.y[self.a])), self.vsq, self.a)
        dae.add_jac(Gy, mul(0.5, self.ngen, pi, self.rho, (self.R)**2, (self.Vwn)**3, div(1, self.mva_mega), (dae.x[self.vw])**3), self.pw, self.cp)
        dae.add_jac(Gy, mul(-25.52, (dae.y[self.ilamb])**-2, exp(mul(-12.5, div(1, dae.y[self.ilamb])))) + mul(12.5, (dae.y[self.ilamb])**-2, -1.1 + mul(25.52, div(1, dae.y[self.ilamb])) + mul(-0.08800000000000001, dae.x[self.theta_p]), exp(mul(-12.5, div(1, dae.y[self.ilamb])))), self.cp, self.ilamb)
        dae.add_jac(Gy, mul((dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))**-2, (div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) + mul(-0.035, div(1, 1 + (dae.x[self.theta_p])**3)))**-2), self.ilamb, self.lamb)
        dae.add_jac(Gy, - mul(dae.y[self.isd], self.u0), self.a, self.vsd)
        dae.add_jac(Gy, - mul(dae.x[self.irq], self.u0), self.a, self.vrq)
        dae.add_jac(Gy, - mul(self.u0, dae.y[self.vsq]), self.a, self.isq)
        dae.add_jac(Gy, - mul(dae.x[self.ird], self.u0), self.a, self.vrd)
        dae.add_jac(Gy, - mul(dae.y[self.isq], self.u0), self.a, self.vsq)
        dae.add_jac(Gy, - mul(self.u0, dae.y[self.vsd]), self.a, self.isd)
        dae.add_jac(Gy, mul(self.u0, mul(2, dae.y[self.v], div(1, self.x0)) + mul(dae.x[self.ird], self.xmu, div(1, self.x0))), self.v, self.v)

    def fxcall(self, dae):
        omega = not0(dae.x[self.omega_m])
        toSb = div(self.Sn, self.system.Settings.mva)
        dae.add_jac(Gx, mul(self.x1, 1 - dae.x[self.omega_m]), self.vrd, self.irq)
        dae.add_jac(Gx, -mul(dae.x[self.irq], self.x1) - mul(dae.y[self.isq], self.xmu), self.vrd, self.omega_m)
        dae.add_jac(Gx, mul(dae.x[self.ird], self.x1) + mul(dae.y[self.isd], self.xmu), self.vrq, self.omega_m)
        dae.add_jac(Gx, - mul(self.x1, 1 - dae.x[self.omega_m]), self.vrq, self.ird)
        dae.add_jac(Gx, mul(1.5, dae.y[self.cp], self.ngen, pi, self.rho, (self.R)**2, (self.Vwn)**3, div(1, self.mva_mega), (dae.x[self.vw])**2), self.pw, self.vw)
        dae.add_jac(Gx, mul(-0.08800000000000001, exp(mul(-12.5, div(1, dae.y[self.ilamb])))), self.cp, self.theta_p)
        dae.add_jac(Gx, mul(-4, self.R, self.fn, self.ngb, dae.x[self.omega_m], pi, div(1, self.Vwn), div(1, self.npole), (dae.x[self.vw])**-2), self.lamb, self.vw)
        dae.add_jac(Gx, mul(4, self.R, self.fn, self.ngb, pi, div(1, self.Vwn), div(1, self.npole), div(1, dae.x[self.vw])), self.lamb, self.omega_m)
        dae.add_jac(Gx, mul((div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) + mul(-0.035, div(1, 1 + (dae.x[self.theta_p])**3)))**-2, mul(0.08, (dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))**-2) + mul(-0.10500000000000001, (dae.x[self.theta_p])**2, (1 + (dae.x[self.theta_p])**3)**-2)), self.ilamb, self.theta_p)
        dae.add_jac(Gx, - mul(self.u0, dae.y[self.vrq]), self.a, self.irq)
        dae.add_jac(Gx, - mul(self.u0, dae.y[self.vrd]), self.a, self.ird)
        dae.add_jac(Gx, mul(self.u0, dae.y[self.v], self.xmu, div(1, self.x0)), self.v, self.ird)
        dae.add_jac(Fx, mul(0.5, dae.y[self.v], self.xmu, div(1, self.H), div(1, self.x0)), self.omega_m, self.irq)
        dae.add_jac(Fx, mul(-0.5, dae.y[self.pw], div(1, self.H), (dae.x[self.omega_m])**-2), self.omega_m, self.omega_m)
        dae.add_jac(Fx, mul(dae.y[self.pwa], self.x0, toSb, div(1, self.Te), (dae.x[self.omega_m])**-2, div(1, dae.y[self.v]), div(1, self.xmu)), self.irq, self.omega_m)
        dae.add_jac(Fy, mul(0.5, div(1, self.H), div(1, omega)), self.omega_m, self.pw)
        dae.add_jac(Fy, mul(0.5, dae.x[self.irq], self.xmu, div(1, self.H), div(1, self.x0)), self.omega_m, self.v)
        dae.add_jac(Fy, mul(dae.y[self.pwa], self.x0, toSb, div(1, self.Te), div(1, omega), (dae.y[self.v])**-2, div(1, self.xmu)), self.irq, self.v)
        dae.add_jac(Fy, - mul(self.x0, toSb, div(1, self.Te), div(1, omega), div(1, dae.y[self.v]), div(1, self.xmu)), self.irq, self.pwa)

    def jac0(self, dae):
        toSb = div(self.Sn, self.system.Settings.mva)
        dae.add_jac(Gy0, -1, self.isd, self.vsd)
        dae.add_jac(Gy0, - self.rs, self.isd, self.isd)
        dae.add_jac(Gy0, self.x0, self.isd, self.isq)
        dae.add_jac(Gy0, -1, self.isq, self.vsq)
        dae.add_jac(Gy0, - self.x0, self.isq, self.isd)
        dae.add_jac(Gy0, - self.rs, self.isq, self.isq)
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
        dae.add_jac(Gx0, - self.xmu, self.isq, self.ird)
        dae.add_jac(Gx0, - self.rr, self.vrd, self.ird)
        dae.add_jac(Gx0, - self.rr, self.vrq, self.irq)
        dae.add_jac(Gx0, 2, self.pwa, self.omega_m)
        dae.add_jac(Fx0, - div(1, self.Tp), self.theta_p, self.theta_p)
        dae.add_jac(Fx0, mul(self.Kp, self.phi, div(1, self.Tp)), self.theta_p, self.omega_m)
        dae.add_jac(Fx0, - div(1, self.Ts), self.ird, self.ird)
        dae.add_jac(Fx0, - div(1, self.Te), self.irq, self.irq)
        dae.add_jac(Fy0, - mul(self.KV, div(1, self.Ts)), self.ird, self.vref)
        dae.add_jac(Fy0, mul(div(1, self.Ts), self.KV - div(1, self.xmu)), self.ird, self.v)
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
