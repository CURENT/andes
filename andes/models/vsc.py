from cvxopt import matrix, mul, spmatrix, div, sin, cos
from .dcbase import DCBase
from ..utils.math import *
from ..consts import *


class VSC(DCBase):
    """VSC model for power flow study"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'AC/DC'
        self._name = 'VSC'
        self._ac = {'bus': ['a', 'v']}
        self._params.extend(['rsh',
                             'xsh',
                             'vshmax',
                             'vshmin',
                             'Ishmax',
                             'pshc',
                             'qshc',
                             'vc',
                             'PQ',
                             'PV',
                             'V',
                             'v0',
                             'vdc0',
                             'k0',
                             'k1',
                             'k2',
                             'droop',
                             'K',
                             'vhigh',
                             'vlow'])
        self._data.update({'rsh': 0.01,
                           'xsh': 0.1,
                           'vshmax': 1.1,
                           'vshmin': 0.9,
                           'Ishmax': 2,
                           'bus': None,
                           'v0': 1.0,
                           'pshc': 0,
                           'qshc': 0,
                           'vc': 1.0,
                           'vdc0': 1.0,
                           'PQ': 0,
                           'PV': 0,
                           'V': 0,
                           'k0': 0,
                           'k1': 0,
                           'k2': 0,
                           'droop': False,
                           'K': 0,
                           'vhigh': 9999,
                           'vlow': 0.0,
                           })
        self._algebs.extend(['ash', 'vsh', 'psh', 'qsh', 'pdc', 'Ish'])
        self._unamey.extend(['ash', 'vsh', 'psh', 'qsh', 'pdc', 'Ish'])
        self._fnamey.extend(['\\theta_{{sh}}', 'V_{{sh}}', 'P_{{sh}}', 'Q_{{sh}}', 'P_{{dc}}', 'I_{{sh}}'])
        self._mandatory.extend(['bus'])
        self._service.extend(['Zsh', 'Ysh', 'glim', 'ylim', 'vio', 'vdcref', 'R'])
        self._dcvoltages.extend(['vdc0'])
        self.calls.update({'init0': True, 'pflow': True,
                           'gcall': True, 'gycall': True,
                           'jac0': True, 'shunt': True,
                           })
        self._inst_meta()
        self.glim = []
        self.ylim = []
        self.vio = {}

    def setup(self):
        super().setup()
        self.K = mul(self.K, self.droop)
        self.R = matrix(0, (self.n, 1), 'd')
        self.Zsh = self.rsh + 1j * self.xsh
        self.Ysh = div(1, self.Zsh)

    def init0(self, dae):
        # behind-transformer AC theta_sh and V_sh - must assign first
        dae.y[self.ash] = dae.y[self.a] + 1e-2
        dae.y[self.vsh] = mul(self.v0, 1 - self.V) + mul(self.vc, self.V) + 1e-6

        Vm = polar(dae.y[self.v], dae.y[self.a] * 1j)
        Vsh = polar(dae.y[self.vsh], dae.y[self.ash] * 1j)  # Initial value for Vsh
        IshC = conj(div(Vm - Vsh, self.Zsh))
        Ssh = mul(Vsh, IshC)

        # PQ PV and V control initials on converters
        dae.y[self.psh] = mul(self.pshc, self.PQ + self.PV)
        dae.y[self.qsh] = mul(self.qshc, self.PQ)
        dae.y[self.v1] = dae.y[self.v2] + mul(dae.y[self.v1], 1 - self.V) + mul(self.vdc0, self.V)

        # PV and V control on AC buses
        dae.y[self.v] = mul(dae.y[self.v], 1 - self.PV - self.V) + mul(self.vc, self.PV + self.V)

        # Converter current initial
        dae.y[self.Ish] = abs(IshC)

        # Converter dc power output
        dae.y[self.pdc] = mul(Vsh, IshC).real() + \
                          (self.k0 + mul(self.k1, dae.y[self.Ish]) + mul(self.k2, mul(dae.y[self.Ish], dae.y[self.Ish])))

    def gcall(self, dae):
        Vm = polar(dae.y[self.v], dae.y[self.a])
        Vsh = polar(dae.y[self.vsh], dae.y[self.ash])
        Ish = mul(self.Ysh, Vm - Vsh)
        IshC = conj(Ish)
        Ssh = mul(Vm, IshC)

        # check the Vsh and Ish limits during PF iterations
        vupper = list(abs(Vsh) - self.vshmax)
        vlower = list(abs(Vsh) - self.vshmin)
        iupper = list(abs(IshC) - self.Ishmax)
        # check for Vsh and Ish limit violations
        if self.system.SPF.iter >= self.system.SPF.ipv2pq:
            for i in range(self.n):
                if vupper[i] > 0 or vlower[i] <0 or iupper[i] > 0:
                    if i not in self.vio.keys():
                        self.vio[i] = list()
                    if vupper[i] > 0:
                        if 'vmax' not in self.vio[i]:
                            self.vio[i].append('vmax')
                            self.system.Log.debug(' * Vmax reached for VSC_{0}'.format(i))
                    elif vlower[i] < 0:
                        if 'vmin' not in self.vio[i]:
                            self.vio[i].append('vmin')
                            self.system.Log.debug(' * Vmin reached for VSC_{0}'.format(i))
                    if iupper[i] > 0:
                        if 'Imax' not in self.vio[i]:
                            self.vio[i].append('Imax')
                            self.system.Log.debug(' * Imax reached for VSC_{0}'.format(i))

        # AC interfaces - power
        dae.g[self.a] += dae.y[self.psh]  # active power load
        dae.g[self.v] += dae.y[self.qsh]  # reactive power load

        # DC interfaces - current
        above = list(dae.y[self.v1] - self.vhigh)
        below = list(dae.y[self.v1] - self.vlow)
        above = matrix([1 if i > 0 else 0 for i in above])
        below = matrix([1 if i < 0 else 0 for i in below])
        self.R = mul(above or below, self.K)
        self.vdcref = mul(self.droop, above, self.vhigh) + mul(self.droop, below, self.vlow)
        dae.g[self.v1] -= div(dae.y[self.pdc], dae.y[self.v1] - dae.y[self.v2])  # current injection
        dae.g += spmatrix(div(dae.y[self.pdc], dae.y[self.v1] - dae.y[self.v2]), self.v2, [0]*self.n, (dae.m, 1), 'd')  # negative current injection

        dae.g[self.ash] = Ssh.real() - dae.y[self.psh]  # (2)
        dae.g[self.vsh] = Ssh.imag() - dae.y[self.qsh]  # (3)

        # PQ, PV or V control
        dae.g[self.psh] = mul(dae.y[self.psh] - self.pshc, (self.PQ + self.PV)) + mul((dae.y[self.v1] - dae.y[self.v2]) - self.vdc0, self.V)  # (12), (15)
        dae.g[self.qsh] = mul(dae.y[self.qsh] - self.qshc, self.PQ) + mul(dae.y[self.v] - self.vc, (self.PV + self.V))  # (13), (16)

        for comp, var in self.vio.items():
            for count, item in enumerate(var):
                if item == 'vmax':
                    yidx = self.vsh[comp]
                    ylim = self.vshmax[comp]
                elif item == 'vmin':
                    yidx = self.vsh[comp]
                    ylim = self.vshmin[comp]
                elif item == 'Imax':
                    yidx = self.Ish[comp]
                    ylim = self.Ishmax[comp]
                else:
                    raise NameError('Unknown limit variable name <{0}>.'.format(item))

                if count == 0:
                    idx = self.qsh[comp]
                else:
                    idx = self.psh[comp]
                self.system.DAE.factorize = True
                dae.g[idx] = dae.y[yidx] - ylim
                if idx not in self.glim:
                    self.glim.append(idx)
                if yidx not in self.ylim:
                    self.ylim.append(yidx)

        dae.g[self.Ish] = abs(IshC) - dae.y[self.Ish]  # (10)

        dae.g[self.pdc] = mul(Vsh, IshC).real() - mul(self.R, dae.y[self.v1] - self.vdcref) + dae.y[self.pdc] - \
                          (self.k0 + mul(self.k1, dae.y[self.Ish]) + mul(self.k2, mul(dae.y[self.Ish], dae.y[self.Ish])))

    def gycall(self, dae):
        Zsh = self.rsh + 1j * self.xsh
        iZsh = div(1, abs(Zsh))
        V = polar(dae.y[self.v], dae.y[self.a] * 1j)
        Vsh = polar(dae.y[self.vsh], dae.y[self.ash] * 1j)
        Ish = div(V - Vsh, Zsh)
        iIsh = div(1, Ish)

        gsh = div(1, Zsh).real()
        bsh = div(1, Zsh).imag()
        V = dae.y[self.v]
        theta = dae.y[self.a]
        Vsh = dae.y[self.vsh]
        thetash = dae.y[self.ash]
        Vdc = dae.y[self.v1] - dae.y[self.v2]

        self.add_jac('Gy', -div(1, Vdc), self.v1, self.pdc)
        self.add_jac('Gy', div(dae.y[self.pdc], mul(Vdc, Vdc)), self.v1, self.v1)
        self.add_jac('Gy', -div(dae.y[self.pdc], mul(Vdc, Vdc)), self.v1, self.v2)

        self.add_jac('Gy', div(1, Vdc), self.v2, self.pdc)
        self.add_jac('Gy', -div(dae.y[self.pdc], mul(Vdc, Vdc)), self.v2, self.v1)
        self.add_jac('Gy', div(dae.y[self.pdc], mul(Vdc, Vdc)), self.v2, self.v2)

        self.add_jac('Gy', 2*mul(gsh, V) - mul(gsh, Vsh, cos(theta - thetash)) - mul(bsh, Vsh, sin(theta - thetash)), self.ash, self.v)
        self.add_jac('Gy', -mul(gsh, V, cos(theta - thetash)) - mul(bsh, V, sin(theta - thetash)), self.ash, self.vsh)
        self.add_jac('Gy', mul(gsh, V, Vsh, sin(theta - thetash)) - mul(bsh, V, Vsh, cos(theta - thetash)), self.ash, self.a)
        self.add_jac('Gy', -mul(gsh, V, Vsh, sin(theta - thetash)) + mul(bsh, V, Vsh, cos(theta - thetash)) + 1e-6, self.ash, self.ash)

        self.add_jac('Gy', -2*mul(bsh, V) - mul(gsh, Vsh, sin(theta - thetash)) + mul(bsh, Vsh, cos(theta - thetash)), self.vsh, self.v)
        self.add_jac('Gy', -mul(gsh, V, sin(theta - thetash)) + mul(bsh, V, cos(theta - thetash)), self.vsh, self.vsh)
        self.add_jac('Gy', -mul(gsh, V, Vsh, cos(theta - thetash)) - mul(bsh, V, Vsh, sin(theta - thetash)), self.vsh, self.a)
        self.add_jac('Gy', mul(gsh, V, Vsh, cos(theta - thetash)) + mul(bsh, V, Vsh, sin(theta - thetash)), self.vsh, self.ash)

        self.add_jac('Gy', 0.5 * mul(2*V - 2*mul(Vsh, cos(theta - thetash)), abs(iIsh), abs(iZsh), abs(iZsh)), self.Ish, self.v)
        self.add_jac('Gy', 0.5 * mul(2*Vsh - 2*mul(V, cos(theta - thetash)), abs(iIsh), abs(iZsh), abs(iZsh)), self.Ish, self.vsh)
        self.add_jac('Gy', 0.5 * mul(2*V, Vsh, sin(theta-thetash), abs(iIsh), abs(iZsh), abs(iZsh)), self.Ish, self.a)
        self.add_jac('Gy', 0.5 * mul(2*V, Vsh, - sin(theta - thetash), abs(iIsh), abs(iZsh), abs(iZsh)), self.Ish, self.ash)

        self.add_jac('Gy', -2 * mul(self.k2, dae.y[self.Ish]), self.pdc, self.Ish)

        self.add_jac('Gy', -mul(2 * gsh, Vsh) + mul(gsh, V, cos(theta - thetash)) - mul(bsh, V, sin(theta - thetash)), self.pdc, self.vsh)
        self.add_jac('Gy', mul(gsh, Vsh, cos(theta - thetash)) - mul(bsh, Vsh, sin(theta - thetash)), self.pdc, self.v)
        self.add_jac('Gy', -mul(gsh, V, Vsh, sin(theta - thetash)) - mul(bsh, V, Vsh, cos(theta - thetash)), self.pdc, self.a)
        self.add_jac('Gy', mul(gsh, V, Vsh, sin(theta - thetash)) + mul(bsh, V, Vsh, cos(theta - thetash)), self.pdc, self.ash)
        self.add_jac('Gy', -self.R, self.pdc, self.v1)

        for gidx, yidx in zip(self.glim, self.ylim):
            self.set_jac('Gy', 0.0, [gidx] * dae.m, range(dae.m))
            self.set_jac('Gy', 1.0, [gidx], [yidx])
            self.set_jac('Gy', 1e-6, [gidx], [gidx])

    def jac0(self, dae):
        self.add_jac(Gy0, -self.u, self.ash, self.psh)
        self.add_jac(Gy0, -self.u, self.vsh, self.qsh)

        self.add_jac(Gy0, mul(self.u, self.PQ + self.PV) + 1e-6, self.psh, self.psh)
        self.add_jac(Gy0, mul(self.u, self.V), self.psh, self.v1)
        self.add_jac(Gy0, -mul(self.u, self.V), self.psh, self.v2)

        self.add_jac(Gy0, mul(self.u, self.PQ) + 1e-6, self.qsh, self.qsh)
        self.add_jac(Gy0, mul(self.u, self.PV + self.V), self.qsh, self.v)

        self.add_jac(Gy0, self.u, self.a, self.psh)
        self.add_jac(Gy0, self.u, self.v, self.qsh)

        self.add_jac(Gy0, -self.u + 1e-6, self.Ish, self.Ish)

        self.add_jac(Gy0, self.u + 1e-6, self.pdc, self.pdc)
        self.add_jac(Gy0, -self.k1, self.pdc, self.Ish)
