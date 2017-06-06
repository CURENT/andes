from cvxopt import matrix, spmatrix
from cvxopt import mul, div, sin, cos, exp
from ..consts import *
from .base import ModelBase


class WTG3(ModelBase):
    """Wind turbine type III"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'WTG'
        self._name = 'WTG3'
        self._algebs.extend(['isd', 'isq', 'vrd', 'vrq', 'vsd', 'vsq', 'vref', 'pwa', 'pw', 'cp', 'lamb', 'ilamb', 'omega_ref'])
        self._fnamex.extend(['\\theta_p', '\\omega_m', 'I_{r, d}', 'I_{r, q}'])
        self._fnamey.extend(['I_{s, d}', 'I_{s, q}', 'V_{r, d}', 'V_{r, q}', 'V_{s, d}', 'V_{s, q}', 'V_{ref}', 'P_{\\omega a}', 'P_w', 'c_p', '\\lambda', '\\frac{1}{\\lambda}', '\\omega_{ref}'])
        self._mandatory.extend(['gen', 'wind'])
        self._params.extend(['Kp', 'nblade', 'ngen', 'npole', 'R', 'Tp', 'ngb', 'rho', 'Ks', 'alf_s', 'H', 'rr', 'rs', 'xr', 'xs', 'xmu', 'Te', 'KV', 'pmax', 'pmin', 'qmax', 'qmin', 'gammap', 'gammaq'])
        self._service.extend(['vref0', 'pi', 'x0', 'phi'])
        self._states.extend(['theta_p', 'omega_m', 'ird', 'irq'])
        self._times.extend(['Tp', 'Te'])
        self._data.update({'wind': 0, 'gammap': 1, 'alf_s': 0, 'qmin': 0, 'ngen': 1, 'Kp': 1, 'H': 2, 'ngb': 1, 'gen': 0, 'rr': 0, 'u': 1, 'rs': 0, 'gammaq': 1, 'Te': 0.01, 'rho': 1, 'Sn': 100, 'xmu': 1, 'pmax': 1, 'qmax': 1, 'nblade': 3, 'pmin': 0, 'R': 1, 'Ks': 1, 'npole': 4, 'Tp': 0.3, 'xs': 1, 'KV': 1, 'xr': 1})
        self._descr.update({'wind': 'Wind time series idx', 'gammap': 'Active power generation ratio', 'alf_s': 'Shadow effect factor', 'qmin': 'Minimum reactive power', 'ngen': 'Number of generators', 'Kp': 'Pitch control gain', 'H': 'Machine rotor and turbine inertia constant', 'ngb': 'Gear box ratio', 'gen': 'Static generator idx', 'rr': 'Rotor resistance', 'rs': 'Stator resistance', 'gammaq': 'Reactive power generation ratio', 'Te': 'Power control time constant', 'rho': 'Air density', 'xmu': 'Magnetizing reactance', 'pmax': 'Maximum active power', 'qmax': 'Maximum active power', 'nblade': 'Number of blades', 'pmin': 'Minimum reactive power', 'R': 'Rotor radius', 'Ks': 'Shaft stiffness', 'npole': 'Number of poles', 'Tp': 'Pitch control time constant', 'xs': 'Stator reactance', 'KV': 'Voltage control gain', 'xr': 'Rotor reactance'})
        self._units.update({'KV': 'pu', 'Te': 's', 'xmu': 'pu', 'pmax': 'pu', 'qmax': 'pu', 'Kp': 'pu', 'pmin': 'pu', 'R': 'm', 'H': 'MWs/MVA', 'Tp': 's', 'xr': 'pu', 'rr': 'pu', 'rs': 'pu', 'Ks': 'pu', 'xs': 'pu', 'rho': 'kg/m^3', 'qmin': 'pu'})
        self.calls.update({'fcall': True, 'fxcall': True, 'init1': True, 'jac0': True, 'gcall': True, 'gycall': True})
        self._inst_meta()

    def servcall(self, dae):
        self.copy_param('StaticGen', 'bus', 'bus', self.gen)
        self.copy_param('Bus', 'v', 'v', self.bus)
        self.copy_param('Bus', 'a', 'a', self.bus)
        self.copy_param('Bus', 'Pg', 'p0', self.bus)
        self.copy_param('Bus', 'Qg', 'q0', self.bus)
        self.copy_param('Wind', 'vw', 'vw', self.wind)
        self.vref0 = dae.y[self.v]
        self.x0 = self.xmu + self.xs
        self.phi = 0.1
        self.vw = [1] * self.n

    def init1(self, dae):
        self.servcall(dae)
        dae.x[self.omega_m] = 1
        dae.y[self.lamb] = mul(self.R, dae.x[self.omega_m], div(1, dae.y[self.vw]))

    def gcall(self, dae):
        dae.g[self.isd] = -dae.y[self.vsd] + mul(dae.x[self.irq], self.xmu) + mul(dae.y[self.isq], self.x0) - mul(dae.y[self.isd], self.rs)
        dae.g[self.isq] = -dae.y[self.vsq] - mul(dae.x[self.ird], self.xmu) - mul(dae.y[self.isd], self.x0) - mul(dae.y[self.isq], self.rs)
        dae.g[self.vrd] = mul(1 - dae.x[self.omega_m], mul(dae.x[self.irq], self.x0) + mul(dae.y[self.isq], self.xmu)) - mul(dae.x[self.ird], self.rr)
        dae.g[self.vrq] = -mul(dae.x[self.irq], self.rr) - mul(1 - dae.x[self.omega_m], mul(dae.x[self.ird], self.x0) + mul(dae.y[self.isd], self.xmu))
        dae.g[self.vsd] = - mul(dae.y[self.v], sin(dae.y[self.a]))
        dae.g[self.vsq] = mul(dae.y[self.v], cos(dae.y[self.a]))
        dae.g[self.vref] = self.vref0 - dae.y[self.vref]
        dae.g[self.pwa] = -1 - dae.y[self.pwa] + 2*dae.x[self.omega_m]
        dae.hard_limit(self.pwa, 0, 1)
        dae.g[self.pw] = -dae.y[self.pw] + mul(0.5, dae.y[self.cp], self.ngen, pi, self.rho, (self.R)**2, div(1, self.Sn), (dae.y[self.vw])**3)
        dae.g[self.cp] = -dae.y[self.cp] + mul(-1.1 + mul(25.52, dae.y[self.ilamb]) + mul(-0.08800000000000001, dae.x[self.theta_p]), exp(mul(-12.5, dae.y[self.ilamb])))
        dae.g[self.lamb] = -dae.y[self.lamb] + mul(self.R, dae.x[self.omega_m], div(1, dae.y[self.vw]))
        dae.g[self.ilamb] = div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) - dae.y[self.ilamb] + mul(-0.035, div(1, 1 + (dae.x[self.theta_p])**3))
        dae.g[self.omega_ref] = 1 - dae.y[self.omega_ref]
        dae.g += spmatrix(mul(self.u, -mul(dae.x[self.ird], dae.y[self.vrd]) - mul(dae.x[self.irq], dae.y[self.vrq]) - mul(dae.y[self.isd], dae.y[self.vsd]) - mul(dae.y[self.isq], dae.y[self.vsq])), self.a, [0]*self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(mul(self.u, mul((dae.y[self.v])**2, div(1, self.xmu)) + mul(dae.x[self.ird], dae.y[self.v], self.xmu, div(1, self.x0))), self.v, [0]*self.n, (dae.m, 1), 'd')

    def fcall(self, dae):
        dae.f[self.theta_p] = mul(div(1, self.Tp), -dae.x[self.theta_p] + mul(self.Kp, self.phi, dae.x[self.omega_m] - dae.y[self.omega_ref]))
        dae.f[self.omega_m] = mul(0.5, div(1, self.H), mul(dae.y[self.pw], div(1, dae.x[self.omega_m])) - mul(self.xmu, mul(dae.x[self.irq], dae.y[self.isd]) - mul(dae.x[self.ird], dae.y[self.isq])))
        dae.f[self.ird] = mul(div(1, self.Te), -dae.x[self.irq] - mul(dae.y[self.pwa], self.x0, div(1, dae.x[self.omega_m]), div(1, dae.y[self.v]), div(1, self.xmu)))
        dae.f[self.irq] = -dae.x[self.ird] + mul(self.KV, dae.y[self.v] - dae.y[self.vref]) - mul(dae.y[self.v], div(1, self.xmu))

    def gycall(self, dae):
        dae.add_jac(Gy, mul(self.xmu, 1 - dae.x[self.omega_m]), self.vrd, self.isq)
        dae.add_jac(Gy, - mul(self.xmu, 1 - dae.x[self.omega_m]), self.vrq, self.isd)
        dae.add_jac(Gy, - mul(dae.y[self.v], cos(dae.y[self.a])), self.vsd, self.a)
        dae.add_jac(Gy, - sin(dae.y[self.a]), self.vsd, self.v)
        dae.add_jac(Gy, - mul(dae.y[self.v], sin(dae.y[self.a])), self.vsq, self.a)
        dae.add_jac(Gy, cos(dae.y[self.a]), self.vsq, self.v)
        dae.add_jac(Gy, mul(0.5, self.ngen, pi, self.rho, (self.R)**2, div(1, self.Sn), (dae.y[self.vw])**3), self.pw, self.cp)
        dae.add_jac(Gy, mul(1.5, dae.y[self.cp], self.ngen, pi, self.rho, (self.R)**2, div(1, self.Sn), (dae.y[self.vw])**2), self.pw, self.vw)
        dae.add_jac(Gy, mul(25.52, exp(mul(-12.5, dae.y[self.ilamb]))) + mul(-12.5, -1.1 + mul(25.52, dae.y[self.ilamb]) + mul(-0.08800000000000001, dae.x[self.theta_p]), exp(mul(-12.5, dae.y[self.ilamb]))), self.cp, self.ilamb)
        dae.add_jac(Gy, - mul(self.R, dae.x[self.omega_m], (dae.y[self.vw])**-2), self.lamb, self.vw)
        dae.add_jac(Gy, - (dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))**-2, self.ilamb, self.lamb)
        dae.add_jac(Gy, - mul(dae.x[self.ird], self.u), self.a, self.vrd)
        dae.add_jac(Gy, - mul(dae.y[self.isd], self.u), self.a, self.vsd)
        dae.add_jac(Gy, - mul(self.u, dae.y[self.vsd]), self.a, self.isd)
        dae.add_jac(Gy, - mul(dae.x[self.irq], self.u), self.a, self.vrq)
        dae.add_jac(Gy, - mul(dae.y[self.isq], self.u), self.a, self.vsq)
        dae.add_jac(Gy, - mul(self.u, dae.y[self.vsq]), self.a, self.isq)
        dae.add_jac(Gy, mul(self.u, mul(2, dae.y[self.v], div(1, self.xmu)) + mul(dae.x[self.ird], self.xmu, div(1, self.x0))), self.v, self.v)

    def fxcall(self, dae):
        dae.add_jac(Gx, mul(self.x0, 1 - dae.x[self.omega_m]), self.vrd, self.irq)
        dae.add_jac(Gx, -mul(dae.x[self.irq], self.x0) - mul(dae.y[self.isq], self.xmu), self.vrd, self.omega_m)
        dae.add_jac(Gx, - mul(self.x0, 1 - dae.x[self.omega_m]), self.vrq, self.ird)
        dae.add_jac(Gx, mul(dae.x[self.ird], self.x0) + mul(dae.y[self.isd], self.xmu), self.vrq, self.omega_m)
        dae.add_jac(Gx, mul(-0.08800000000000001, exp(mul(-12.5, dae.y[self.ilamb]))), self.cp, self.theta_p)
        dae.add_jac(Gx, mul(self.R, div(1, dae.y[self.vw])), self.lamb, self.omega_m)
        dae.add_jac(Gx, mul(-0.08, (dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))**-2) + mul(0.10500000000000001, (dae.x[self.theta_p])**2, (1 + (dae.x[self.theta_p])**3)**-2), self.ilamb, self.theta_p)
        dae.add_jac(Gx, - mul(self.u, dae.y[self.vrq]), self.a, self.irq)
        dae.add_jac(Gx, - mul(self.u, dae.y[self.vrd]), self.a, self.ird)
        dae.add_jac(Gx, mul(self.u, dae.y[self.v], self.xmu, div(1, self.x0)), self.v, self.ird)
        dae.add_jac(Fx, mul(-0.5, dae.y[self.isd], self.xmu, div(1, self.H)), self.omega_m, self.irq)
        dae.add_jac(Fx, mul(-0.5, dae.y[self.pw], div(1, self.H), (dae.x[self.omega_m])**-2), self.omega_m, self.omega_m)
        dae.add_jac(Fx, mul(0.5, dae.y[self.isq], self.xmu, div(1, self.H)), self.omega_m, self.ird)
        dae.add_jac(Fx, mul(dae.y[self.pwa], self.x0, div(1, self.Te), (dae.x[self.omega_m])**-2, div(1, dae.y[self.v]), div(1, self.xmu)), self.ird, self.omega_m)
        dae.add_jac(Fy, mul(0.5, div(1, self.H), div(1, dae.x[self.omega_m])), self.omega_m, self.pw)
        dae.add_jac(Fy, mul(-0.5, dae.x[self.irq], self.xmu, div(1, self.H)), self.omega_m, self.isd)
        dae.add_jac(Fy, mul(0.5, dae.x[self.ird], self.xmu, div(1, self.H)), self.omega_m, self.isq)
        dae.add_jac(Fy, - mul(self.x0, div(1, self.Te), div(1, dae.x[self.omega_m]), div(1, dae.y[self.v]), div(1, self.xmu)), self.ird, self.pwa)
        dae.add_jac(Fy, mul(dae.y[self.pwa], self.x0, div(1, self.Te), div(1, dae.x[self.omega_m]), (dae.y[self.v])**-2, div(1, self.xmu)), self.ird, self.v)

    def jac0(self, dae):
        dae.add_jac(Gy0, self.x0, self.isd, self.isq)
        dae.add_jac(Gy0, -1, self.isd, self.vsd)
        dae.add_jac(Gy0, - self.rs, self.isd, self.isd)
        dae.add_jac(Gy0, -1, self.isq, self.vsq)
        dae.add_jac(Gy0, - self.rs, self.isq, self.isq)
        dae.add_jac(Gy0, - self.x0, self.isq, self.isd)
        dae.add_jac(Gy0, -1, self.vref, self.vref)
        dae.add_jac(Gy0, -1, self.pwa, self.pwa)
        dae.add_jac(Gy0, -1, self.pw, self.pw)
        dae.add_jac(Gy0, -1, self.cp, self.cp)
        dae.add_jac(Gy0, -1, self.lamb, self.lamb)
        dae.add_jac(Gy0, -1, self.ilamb, self.ilamb)
        dae.add_jac(Gy0, -1, self.omega_ref, self.omega_ref)
        dae.add_jac(Gx0, self.xmu, self.isd, self.irq)
        dae.add_jac(Gx0, - self.xmu, self.isq, self.ird)
        dae.add_jac(Gx0, - self.rr, self.vrd, self.ird)
        dae.add_jac(Gx0, - self.rr, self.vrq, self.irq)
        dae.add_jac(Gx0, 2, self.pwa, self.omega_m)
        dae.add_jac(Fx0, - div(1, self.Tp), self.theta_p, self.theta_p)
        dae.add_jac(Fx0, mul(self.Kp, self.phi, div(1, self.Tp)), self.theta_p, self.omega_m)
        dae.add_jac(Fx0, - div(1, self.Te), self.ird, self.irq)
        dae.add_jac(Fx0, -1, self.irq, self.ird)
        dae.add_jac(Fy0, - mul(self.Kp, self.phi, div(1, self.Tp)), self.theta_p, self.omega_ref)
        dae.add_jac(Fy0, - self.KV, self.irq, self.vref)
        dae.add_jac(Fy0, self.KV - div(1, self.xmu), self.irq, self.v)
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
        dae.add_jac(Gy0, 1e-6, self.omega_ref, self.omega_ref)
