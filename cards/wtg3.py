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
        self._algebs.extend(['isd', 'isq', 'vrd', 'vrq', 'vsd', 'vsq', 'vref', 'pwa', 'pw', 'cp', 'lamb', 'ilamb'])
        self._fnamex.extend(['\\theta_p', '\\omega_m', 'I_{r, d}', 'I_{r, q}'])
        self._fnamey.extend(['I_{s, d}', 'I_{s, q}', 'V_{r, d}', 'V_{r, q}', 'V_{s, d}', 'V_{s, q}', 'V_{ref}', 'P_{\\omega a}', 'P_w', 'c_p', '\\lambda', '\\frac{1}{\\lambda}', '\\omega_{ref}'])
        self._mandatory.extend(['bus', 'gen', 'wind'])
        self._params.extend(['Kp', 'nblade', 'ngen', 'npole', 'R', 'Tp', 'Ts', 'ngb', 'H', 'rr', 'rs', 'xr', 'xs', 'xmu', 'Te', 'KV', 'pmax', 'pmin', 'qmax', 'qmin', 'gammap', 'gammaq'])
        self._powers.extend(['H', 'pmax', 'pmin', 'qmax', 'qmin'])
        self._service.extend(['vref0', 'x0', 'x1', 'pi', 'phi', 'u0', 'fn', 'mva_mega', 'ird_min', 'ird_max', 'irq_min', 'irq_max', 'irq_off'])
        self._states.extend(['theta_p', 'omega_m', 'ird', 'irq'])
        self._times.extend(['Tp', 'Te'])
        self._z.extend(['rs', 'xs', 'rr', 'xr', 'xmu'])
        self._ac.update(
{'bus': ['a', 'v']})
        self._config_descr.update(
{           'H': 'Machine rotor and turbine inertia constant',
            'KV': 'Voltage control gain',
            'Kp': 'Pitch control gain',
            'R': 'Rotor radius',
            'Te': 'Power control time constant',
            'Tp': 'Pitch control time constant',
            'Ts': 'Speed control time constant',
            'bus': 'Bus idx',
            'gammap': 'Active power generation ratio',
            'gammaq': 'Reactive power generation ratio',
            'gen': 'Static generator idx',
            'nblade': 'Number of blades',
            'ngb': 'Gear box ratio',
            'ngen': 'Number of generators',
            'npole': 'Number of poles',
            'pmax': 'Maximum active power',
            'pmin': 'Minimum reactive power',
            'qmax': 'Maximum active power',
            'qmin': 'Minimum reactive power',
            'rr': 'Rotor resistance',
            'rs': 'Stator resistance',
            'wind': 'Wind time series idx',
            'xmu': 'Magnetizing reactance',
            'xr': 'Rotor reactance',
            'xs': 'Stator reactance'})
        self._data.update(
{           'H': 2,
            'KV': 10,
            'Kp': 10,
            'R': 35,
            'Sn': 40,
            'Te': 0.01,
            'Tp': 3.0,
            'Ts': 1,
            'bus': 0,
            'gammap': 1,
            'gammaq': 1,
            'gen': 0,
            'nblade': 3,
            'ngb': 0.011235,
            'ngen': 40,
            'npole': 4,
            'pmax': 1.0,
            'pmin': 0,
            'qmax': 0.6,
            'qmin': -0.6,
            'rr': 0.01,
            'rs': 0.01,
            'u': 1,
            'wind': 0,
            'xmu': 3,
            'xr': 0.08,
            'xs': 0.1})
        self._units.update(
{           'H': 'MWs/MVA',
            'KV': 'pu',
            'Kp': 'pu',
            'R': 'm',
            'Te': 's',
            'Tp': 's',
            'pmax': 'pu',
            'pmin': 'pu',
            'qmax': 'pu',
            'qmin': 'pu',
            'rr': 'pu',
            'rs': 'pu',
            'xmu': 'pu',
            'xr': 'pu',
            'xs': 'pu'})
        self.calls.update({'gcall': True, 'fcall': True, 'gycall': True, 'fxcall': True, 'jac0': True, 'init1': True})
        self._init()

    def servcall(self, dae):
        self.copy_data_ext('StaticGen', 'u', 'dest=ugen', idx=self.gen)
        self.copy_data_ext('Bus', 'Pg', 'dest=p0', idx=self.bus)
        self.copy_data_ext('Bus', 'Qg', 'dest=q0', idx=self.bus)
        self.copy_data_ext('Wind', 'vw', 'dest=vw', idx=self.wind)
        self.copy_data_ext('Wind', 'rho', 'dest=rho', idx=self.wind)
        self.copy_data_ext('Wind', 'Vwn', 'dest=Vwn', idx=self.wind)
        self.vref0 = dae.y[self.v]
        self.x0 = self.xmu + self.xs
        self.x1 = self.xmu + self.xr
        self.pi = 3.14
        self.phi = 0.1
        self.u0 = mul(self.u, self.ugen)
        self.fn = 60
        self.mva_mega = 100000000.0
        self.ird_min = 0
        self.ird_max = 999
        self.irq_min = 0
        self.irq_max = 999
        self.irq_off = 0

    def init1(self, dae):
        self.servcall(dae)

    def gcall(self, dae):
        dae.g[self.isd] = -dae.y[self.vsd] + mul(dae.x[self.irq], self.xmu) + mul(dae.y[self.isq], self.x0) - mul(dae.y[self.isd], self.rs)
        dae.g[self.isq] = -dae.y[self.vsq] - mul(dae.x[self.ird], self.xmu) - mul(dae.y[self.isd], self.x0) - mul(dae.y[self.isq], self.rs)
        dae.g[self.vrd] = -dae.y[self.vrd] + mul(1 - dae.x[self.omega_m], mul(dae.x[self.irq], self.x1) + mul(dae.y[self.isq], self.xmu)) - mul(dae.x[self.ird], self.rr)
        dae.g[self.vrq] = -dae.y[self.vrq] - mul(dae.x[self.irq], self.rr) - mul(1 - dae.x[self.omega_m], mul(dae.x[self.ird], self.x1) + mul(dae.y[self.isd], self.xmu))
        dae.g[self.vsd] = -dae.y[self.vsd] - mul(dae.y[self.v], sin(dae.y[self.a]))
        dae.g[self.vsq] = -dae.y[self.vsq] + mul(dae.y[self.v], cos(dae.y[self.a]))
        dae.g[self.vref] = self.vref0 - dae.y[self.vref]
        dae.g[self.pwa] = -1 - dae.y[self.pwa] + 2*dae.x[self.omega_m]
        dae.hard_limit(self.pwa, 0, 1)
        dae.g[self.pw] = -dae.y[self.pw] + mul(0.5, dae.y[self.cp], self.ngen, self.pi, self.rho, (self.R)**2, (self.Vwn)**3, div(1, self.mva_mega), (dae.x[self.vw])**3)
        dae.g[self.cp] = -dae.y[self.cp] + mul(-1.1 + mul(25.52, div(1, dae.y[self.ilamb])) + mul(-0.08800000000000001, dae.x[self.theta_p]), exp(mul(-12.5, div(1, dae.y[self.ilamb]))))
        dae.g[self.lamb] = -dae.y[self.lamb] + mul(4, self.R, self.fn, self.ngb, dae.x[self.omega_m], self.pi, div(1, self.Vwn), div(1, self.npole), div(1, dae.x[self.vw]))
        dae.g[self.ilamb] = div(1, div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) + mul(-0.035, div(1, 1 + (dae.x[self.theta_p])**3))) - dae.y[self.ilamb]
        dae.g += spmatrix(mul(self.u0, -mul(dae.x[self.ird], dae.y[self.vrd]) - mul(dae.x[self.irq], dae.y[self.vrq]) - mul(dae.y[self.isd], dae.y[self.vsd]) - mul(dae.y[self.isq], dae.y[self.vsq])), self.a, [0]*self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(mul(self.u0, mul((dae.y[self.v])**2, div(1, self.x0)) + mul(dae.x[self.ird], dae.y[self.v], self.xmu, div(1, self.x0))), self.v, [0]*self.n, (dae.m, 1), 'd')

    def fcall(self, dae):
        dae.f[self.theta_p] = mul(div(1, self.Tp), -dae.x[self.theta_p] + mul(self.Kp, self.phi, -1 + dae.x[self.omega_m]))
        dae.anti_windup(self.theta_p, 0, self.pi)
        dae.f[self.omega_m] = mul(0.5, div(1, self.H), mul(dae.y[self.pw], div(1, dae.x[self.omega_m])) - mul(self.xmu, mul(dae.x[self.irq], dae.y[self.isd]) - mul(dae.x[self.ird], dae.y[self.isq])))
        dae.f[self.ird] = mul(div(1, self.Ts), -dae.x[self.ird] + mul(self.KV, dae.y[self.v] - dae.y[self.vref]) - mul(dae.y[self.v], div(1, self.xmu)))
        dae.anti_windup(self.ird, self.ird_min, self.irq_max)
        dae.f[self.irq] = mul(div(1, self.Te), -dae.x[self.irq] - self.irq_off - mul(dae.y[self.pwa], self.x0, div(1, dae.x[self.omega_m]), div(1, dae.y[self.v]), div(1, self.xmu)))
        dae.anti_windup(self.irq, self.irq_min, self.irq_max)

    def gycall(self, dae):
        dae.add_jac(Gy, mul(self.xmu, 1 - dae.x[self.omega_m]), self.vrd, self.isq)
        dae.add_jac(Gy, mul(self.xmu, -1 + dae.x[self.omega_m]), self.vrq, self.isd)
        dae.add_jac(Gy, - mul(dae.y[self.v], cos(dae.y[self.a])), self.vsd, self.a)
        dae.add_jac(Gy, - sin(dae.y[self.a]), self.vsd, self.v)
        dae.add_jac(Gy, - mul(dae.y[self.v], sin(dae.y[self.a])), self.vsq, self.a)
        dae.add_jac(Gy, cos(dae.y[self.a]), self.vsq, self.v)
        dae.add_jac(Gy, mul(0.5, self.ngen, self.pi, self.rho, (self.R)**2, (self.Vwn)**3, div(1, self.mva_mega), (dae.x[self.vw])**3), self.pw, self.cp)
        dae.add_jac(Gy, mul(-25.52, (dae.y[self.ilamb])**-2, exp(mul(-12.5, div(1, dae.y[self.ilamb])))) + mul(12.5, (dae.y[self.ilamb])**-2, -1.1 + mul(25.52, div(1, dae.y[self.ilamb])) + mul(-0.08800000000000001, dae.x[self.theta_p]), exp(mul(-12.5, div(1, dae.y[self.ilamb])))), self.cp, self.ilamb)
        dae.add_jac(Gy, mul((dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))**-2, (div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) + mul(-0.035, div(1, 1 + (dae.x[self.theta_p])**3)))**-2), self.ilamb, self.lamb)
        dae.add_jac(Gy, - mul(self.u0, dae.y[self.vsq]), self.a, self.isq)
        dae.add_jac(Gy, - mul(dae.x[self.irq], self.u0), self.a, self.vrq)
        dae.add_jac(Gy, - mul(dae.y[self.isd], self.u0), self.a, self.vsd)
        dae.add_jac(Gy, - mul(self.u0, dae.y[self.vsd]), self.a, self.isd)
        dae.add_jac(Gy, - mul(dae.x[self.ird], self.u0), self.a, self.vrd)
        dae.add_jac(Gy, - mul(dae.y[self.isq], self.u0), self.a, self.vsq)
        dae.add_jac(Gy, mul(self.u0, mul(2, dae.y[self.v], div(1, self.x0)) + mul(dae.x[self.ird], self.xmu, div(1, self.x0))), self.v, self.v)

    def fxcall(self, dae):
        dae.add_jac(Gx, mul(self.x1, 1 - dae.x[self.omega_m]), self.vrd, self.irq)
        dae.add_jac(Gx, -mul(dae.x[self.irq], self.x1) - mul(dae.y[self.isq], self.xmu), self.vrd, self.omega_m)
        dae.add_jac(Gx, mul(dae.x[self.ird], self.x1) + mul(dae.y[self.isd], self.xmu), self.vrq, self.omega_m)
        dae.add_jac(Gx, mul(self.x1, -1 + dae.x[self.omega_m]), self.vrq, self.ird)
        dae.add_jac(Gx, mul(1.5, dae.y[self.cp], self.ngen, self.pi, self.rho, (self.R)**2, (self.Vwn)**3, div(1, self.mva_mega), (dae.x[self.vw])**2), self.pw, self.vw)
        dae.add_jac(Gx, mul(-0.08800000000000001, exp(mul(-12.5, div(1, dae.y[self.ilamb])))), self.cp, self.theta_p)
        dae.add_jac(Gx, mul(-4, self.R, self.fn, self.ngb, dae.x[self.omega_m], self.pi, div(1, self.Vwn), div(1, self.npole), (dae.x[self.vw])**-2), self.lamb, self.vw)
        dae.add_jac(Gx, mul(4, self.R, self.fn, self.ngb, self.pi, div(1, self.Vwn), div(1, self.npole), div(1, dae.x[self.vw])), self.lamb, self.omega_m)
        dae.add_jac(Gx, mul((div(1, dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p])) + mul(-0.035, div(1, 1 + (dae.x[self.theta_p])**3)))**-2, mul(0.08, (dae.y[self.lamb] + mul(0.08, dae.x[self.theta_p]))**-2) + mul(-0.10500000000000001, (dae.x[self.theta_p])**2, (1 + (dae.x[self.theta_p])**3)**-2)), self.ilamb, self.theta_p)
        dae.add_jac(Gx, - mul(self.u0, dae.y[self.vrd]), self.a, self.ird)
        dae.add_jac(Gx, - mul(self.u0, dae.y[self.vrq]), self.a, self.irq)
        dae.add_jac(Gx, mul(self.u0, dae.y[self.v], self.xmu, div(1, self.x0)), self.v, self.ird)
        dae.add_jac(Fx, mul(-0.5, dae.y[self.pw], div(1, self.H), (dae.x[self.omega_m])**-2), self.omega_m, self.omega_m)
        dae.add_jac(Fx, mul(0.5, dae.y[self.isq], self.xmu, div(1, self.H)), self.omega_m, self.ird)
        dae.add_jac(Fx, mul(-0.5, dae.y[self.isd], self.xmu, div(1, self.H)), self.omega_m, self.irq)
        dae.add_jac(Fx, mul(dae.y[self.pwa], self.x0, div(1, self.Te), (dae.x[self.omega_m])**-2, div(1, dae.y[self.v]), div(1, self.xmu)), self.irq, self.omega_m)
        dae.add_jac(Fy, mul(0.5, dae.x[self.ird], self.xmu, div(1, self.H)), self.omega_m, self.isq)
        dae.add_jac(Fy, mul(-0.5, dae.x[self.irq], self.xmu, div(1, self.H)), self.omega_m, self.isd)
        dae.add_jac(Fy, mul(0.5, div(1, self.H), div(1, dae.x[self.omega_m])), self.omega_m, self.pw)
        dae.add_jac(Fy, - mul(self.x0, div(1, self.Te), div(1, dae.x[self.omega_m]), div(1, dae.y[self.v]), div(1, self.xmu)), self.irq, self.pwa)
        dae.add_jac(Fy, mul(dae.y[self.pwa], self.x0, div(1, self.Te), div(1, dae.x[self.omega_m]), (dae.y[self.v])**-2, div(1, self.xmu)), self.irq, self.v)

    def jac0(self, dae):
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
