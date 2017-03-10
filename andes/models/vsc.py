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
                             'pref0',
                             'qref0',
                             'vref0',
                             'v0',
                             'vdcref0',
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
                           'control': None,
                           'v0': 1.0,
                           'pref0': 0,
                           'qref0': 0,
                           'vref0': 1.0,
                           'vdcref0': 1.0,
                           'k0': 0,
                           'k1': 0,
                           'k2': 0,
                           'droop': False,
                           'K': 0,
                           'vhigh': 9999,
                           'vlow': 0.0,
                           })
        self._units.update({'rsh': 'pu',
                            'xsh': 'pu',
                            'vshmax': 'pu',
                            'vshmin': 'pu',
                            'Ishmax': 'pu',
                            'bus': 'na',
                            'v0': 'pu',
                            'pshc': 'pu',
                            'qshc': 'pu',
                            'vc': 'pu',
                            'vdc0': 'pu',
                            'k0': 'na',
                            'k1': 'na',
                            'k2': 'na',
                            'droop': 'na',
                            'K': 'na',
                            'vhigh': 'pu',
                            'vlow': 'pu',
                            'control': 'na'
                            })
        self._algebs.extend(['ash', 'vsh', 'psh', 'qsh', 'pdc', 'Ish'])
        self._unamey.extend(['ash', 'vsh', 'psh', 'qsh', 'pdc', 'Ish'])
        self._fnamey.extend(['\\theta_{{sh}}', 'V_{{sh}}', 'P_{{sh}}', 'Q_{{sh}}', 'P_{{dc}}', 'I_{{sh}}'])
        self._mandatory.extend(['bus', 'control'])
        self._service.extend(['Zsh', 'Ysh', 'glim', 'ylim', 'vio', 'vdcref', 'R',
                              'PQ', 'PV', 'VV', 'VQ'])
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

        self.PQ = zeros(self.n, 1)
        self.PV = zeros(self.n, 1)
        self.VV = zeros(self.n, 1)
        self.VQ = zeros(self.n, 1)
        for idx, cc in enumerate(self.control):
            if cc not in ['PQ', 'PV', 'VV', 'VQ']:
                raise KeyError('VSC {0} control parameter {1} is invalid.'.format(self.names[idx], cc))
            self.__dict__[cc][idx] = 1

    def init0(self, dae):
        # behind-transformer AC theta_sh and V_sh - must assign first
        dae.y[self.ash] = dae.y[self.a] + 1e-2
        dae.y[self.vsh] = mul(self.v0, 1 - self.VV) + mul(self.vref0, self.VV) + 1e-6

        Vm = polar(dae.y[self.v], dae.y[self.a] * 1j)
        Vsh = polar(dae.y[self.vsh], dae.y[self.ash] * 1j)  # Initial value for Vsh
        IshC = conj(div(Vm - Vsh, self.Zsh))
        Ssh = mul(Vsh, IshC)

        # PQ PV and V control initials on converters
        dae.y[self.psh] = mul(self.pref0, self.PQ + self.PV)
        dae.y[self.qsh] = mul(self.qref0, self.PQ)
        dae.y[self.v1] = dae.y[self.v2] + mul(dae.y[self.v1], 1 - self.VV) + mul(self.vdcref0, self.VV)

        # PV and V control on AC buses
        dae.y[self.v] = mul(dae.y[self.v], 1 - self.PV - self.VV) + mul(self.vref0, self.PV + self.VV)

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
        dae.g[self.psh] = mul(dae.y[self.psh] - self.pref0, self.PQ + self.PV) + mul((dae.y[self.v1] - dae.y[self.v2]) - self.vdcref0, self.VV)  # (12), (15)
        dae.g[self.qsh] = mul(dae.y[self.qsh] - self.qref0, self.PQ + self.VQ) + mul(dae.y[self.v] - self.vref0, (self.PV + self.VV))  # (13), (16)

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

        dae.add_jac(Gy, -div(1, Vdc), self.v1, self.pdc)
        dae.add_jac(Gy, div(dae.y[self.pdc], mul(Vdc, Vdc)), self.v1, self.v1)
        dae.add_jac(Gy, -div(dae.y[self.pdc], mul(Vdc, Vdc)), self.v1, self.v2)

        dae.add_jac(Gy, div(1, Vdc), self.v2, self.pdc)
        dae.add_jac(Gy, -div(dae.y[self.pdc], mul(Vdc, Vdc)), self.v2, self.v1)
        dae.add_jac(Gy, div(dae.y[self.pdc], mul(Vdc, Vdc)), self.v2, self.v2)

        dae.add_jac(Gy, 2*mul(gsh, V) - mul(gsh, Vsh, cos(theta - thetash)) - mul(bsh, Vsh, sin(theta - thetash)), self.ash, self.v)
        dae.add_jac(Gy, -mul(gsh, V, cos(theta - thetash)) - mul(bsh, V, sin(theta - thetash)), self.ash, self.vsh)
        dae.add_jac(Gy, mul(gsh, V, Vsh, sin(theta - thetash)) - mul(bsh, V, Vsh, cos(theta - thetash)), self.ash, self.a)
        dae.add_jac(Gy, -mul(gsh, V, Vsh, sin(theta - thetash)) + mul(bsh, V, Vsh, cos(theta - thetash)) + 1e-6, self.ash, self.ash)

        dae.add_jac(Gy, -2*mul(bsh, V) - mul(gsh, Vsh, sin(theta - thetash)) + mul(bsh, Vsh, cos(theta - thetash)), self.vsh, self.v)
        dae.add_jac(Gy, -mul(gsh, V, sin(theta - thetash)) + mul(bsh, V, cos(theta - thetash)), self.vsh, self.vsh)
        dae.add_jac(Gy, -mul(gsh, V, Vsh, cos(theta - thetash)) - mul(bsh, V, Vsh, sin(theta - thetash)), self.vsh, self.a)
        dae.add_jac(Gy, mul(gsh, V, Vsh, cos(theta - thetash)) + mul(bsh, V, Vsh, sin(theta - thetash)), self.vsh, self.ash)

        dae.add_jac(Gy, 0.5 * mul(2*V - 2*mul(Vsh, cos(theta - thetash)), abs(iIsh), abs(iZsh), abs(iZsh)), self.Ish, self.v)
        dae.add_jac(Gy, 0.5 * mul(2*Vsh - 2*mul(V, cos(theta - thetash)), abs(iIsh), abs(iZsh), abs(iZsh)), self.Ish, self.vsh)
        dae.add_jac(Gy, 0.5 * mul(2*V, Vsh, sin(theta-thetash), abs(iIsh), abs(iZsh), abs(iZsh)), self.Ish, self.a)
        dae.add_jac(Gy, 0.5 * mul(2*V, Vsh, - sin(theta - thetash), abs(iIsh), abs(iZsh), abs(iZsh)), self.Ish, self.ash)

        dae.add_jac(Gy, -2 * mul(self.k2, dae.y[self.Ish]), self.pdc, self.Ish)

        dae.add_jac(Gy, -mul(2 * gsh, Vsh) + mul(gsh, V, cos(theta - thetash)) - mul(bsh, V, sin(theta - thetash)), self.pdc, self.vsh)
        dae.add_jac(Gy, mul(gsh, Vsh, cos(theta - thetash)) - mul(bsh, Vsh, sin(theta - thetash)), self.pdc, self.v)
        dae.add_jac(Gy, -mul(gsh, V, Vsh, sin(theta - thetash)) - mul(bsh, V, Vsh, cos(theta - thetash)), self.pdc, self.a)
        dae.add_jac(Gy, mul(gsh, V, Vsh, sin(theta - thetash)) + mul(bsh, V, Vsh, cos(theta - thetash)), self.pdc, self.ash)
        dae.add_jac(Gy, -self.R, self.pdc, self.v1)

        for gidx, yidx in zip(self.glim, self.ylim):
            dae.set_jac(Gy, 0.0, [gidx] * dae.m, range(dae.m))
            dae.set_jac(Gy, 1.0, [gidx], [yidx])
            dae.set_jac(Gy, 1e-6, [gidx], [gidx])

    def jac0(self, dae):
        dae.add_jac(Gy0, -self.u, self.ash, self.psh)
        dae.add_jac(Gy0, -self.u, self.vsh, self.qsh)

        dae.add_jac(Gy0, mul(self.u, self.PQ + self.PV) + 1e-6, self.psh, self.psh)
        dae.add_jac(Gy0, mul(self.u, self.VV), self.psh, self.v1)
        dae.add_jac(Gy0, -mul(self.u, self.VV), self.psh, self.v2)

        dae.add_jac(Gy0, mul(self.u, self.PQ) + 1e-6, self.qsh, self.qsh)
        dae.add_jac(Gy0, mul(self.u, self.PV + self.VV), self.qsh, self.v)

        dae.add_jac(Gy0, self.u, self.a, self.psh)
        dae.add_jac(Gy0, self.u, self.v, self.qsh)

        dae.add_jac(Gy0, -self.u + 1e-6, self.Ish, self.Ish)

        dae.add_jac(Gy0, self.u + 1e-6, self.pdc, self.pdc)
        dae.add_jac(Gy0, -self.k1, self.pdc, self.Ish)

class VSCBase(DCBase):
    """Shunt-connected dynamic VSC model for transient simulation"""
    def __init__(self, system, name):
        self._group = 'AC/DC'
        self._name = 'VSCDyn'
        self._ac = {'bus': ['a', 'v']}
        self._data.update({'vsc': None,
                           'Kp1': 0.1,
                           'Ki1': 0.1,
                           'Kp2': 0.1,
                           'Ki2': 0.1,
                           'Kp3': 0.1,
                           'Ki3': 0.1,
                           'Kp4': 0.1,
                           'Ki4': 0.1,
                           'Kpdc': 0.1,
                           'Kidc': 0.1,
                           'Tt': 0.02,
                           'Tdc': 0.02,

                           })
        self._params.extend(['vsc', 'Kp1', 'Ki1', 'Kp2', 'Ki2', 'Kp3', 'Ki3', 'Kp4', 'Ki4', 'Kp', 'Ki',
                             'Tt'])
        self._descr.update({'vsc': 'static vsc idx',
                            'Kp1': 'current controller proportional gain',
                            'Ki1': 'current controller integrator gain',
                            'Kp2': 'Q controller proportional gain',
                            'Ki2': 'Q controller integrator gain',
                            'Kp3': 'ac voltage controller proportional gain',
                            'Ki3': 'ac voltage controller integrator gain',
                            'Kp4': 'P controller proportional gain',
                            'Ki4': 'P controller integrator gain',
                            'Kpdc': 'dc voltage controller proportional gain',
                            'Kidc': 'dc voltage controller integrator gain',
                            'Tt': 'ac voltage measurement delay time constant',
                            })
        self._algebs.extend(['uref', 'qref', 'pref', 'udcref', 'Idref', 'Iqref', 'Idcy'])
        self._states.extend(['Id', 'Iq', 'Md', 'Mq', 'ucd', 'ucq', 'Nd', 'Nq', 'Idcx'])
        self._service.extend(['rsh', 'xsh', 'iLsh', 'wn', 'usd', 'usq', 'iTt', 'PQ', 'PV', 'V', 'VQ', 'adq'])
        self._mandatory.extend(['vsc'])
        self._fnamey.extend(['U^{ref}', 'Q^{ref}', 'P^{ref}', 'U_{dc}^{ref}', 'I_d^{ref}', 'I_q^{ref}, I_{dcy}'])
        self._fnamex.extend(['I_d', 'I_q', 'M_d', 'M_q', 'u_c^d', 'u_c^q', 'N_d', 'N_q', 'I_{dcx}'])
        self.calls.update({'init1': True, 'gcall': True,
                           'gycall': True, 'fxcall': True,
                           'jac0': True,
                           })

    def init1(self, dae):
        self.copy_param('VSC', src='rsh', fkey=self.vsc)
        self.copy_param('VSC', src='xsh', fkey=self.vsc)
        self.copy_param('VSC', src='PQ', fkey=self.vsc)
        self.copy_param('VSC', src='PV', fkey=self.vsc)
        self.copy_param('VSC', src='V', fkey=self.vsc)
        self.copy_param('VSC', src='VQ', fkey=self.vsc)
        # self.copy_param('VSC', src='bus', fkey=self.vsc)
        self.copy_param('VSC', src='a', fkey=self.vsc)
        self.copy_param('VSC', src='v', fkey=self.vsc)
        self.copy_param('VSC', src='ash', fkey=self.vsc)
        self.copy_param('VSC', src='vsh', fkey=self.vsc)
        self.copy_param('VSC', src='v1', fkey=self.vsc)
        self.copy_param('VSC', src='v2', fkey=self.vsc)

        self.wn = ones(self.n, 1)  # in per unit
        self.iLsh = div(1.0, self.xsh)
        self.usd = ones(self.n, 1)  # need initialize
        self.usq = ones(self.n, 1)  # need initialize
        self.iTt = div(1.0, self.Tt)

    def gcall(self):
        self.adq = dae.y[self.a]
        self.usq = mul(dae.y[self.v], cos(dae.y[self.a] - self.adq))
        self.usd = mul(dae.y[self.v], sin(dae.y[self.a] - self.adq))
        iudc = div(1, dae.y[self.v1] - dae.y[self.v2])

        dae.g[self.uref] = mul(self.PV + self.VV, dae.y[self.uref] - self.uref0)
        dae.g[self.pref] = mul(self.PV + self.PQ, dae.y[self.pref] - self.pcref0)
        dae.g[self.qref] = mul(self.PQ + self.VQ, dae.y[self.qref] - self.qcref0)
        dae.g[self.udcref] = mul(self.VV + self.VQ, dae.y[self.udcref] - self.udcref0)

        dae.g[self.Idref] = mul(self.PQ + self.VQ, div(dae.y[self.qref], self.usq) + dae.x[self.Nd]
                                                   + mul(self.Kp2, dae.y[self.qref] - mul(self.usq, dae.x[self.Id]))) \
                            + mul(self.PV + self.VV, dae.x[self.Nd] + mul(self.Kp3, dae.y[self.uref] - self.usq)) \
                            - dae.y[self.Idref]

        dae.g[self.Iqref] = mul(self.PQ + self.PV, div(dae.y[self.pref], self.usq) + dae.x[self.Nq]
                                                   + mul(self.Kp4, dae.y[self.pref] - mul(self.usq, dae.x[self.Iq]))) \
                            + mul(self.VV + self.VQ, mul(2 * dae.x[self.Idc], (dae.y[self.v1] - dae.y[self.v2]))
                                                    - mul(dae.x[self.Id], dae.x[self.ucd]), iucq) \
                            - dae.y[self.Iqref]
        dae.g[self.Idcy] = mul(self.PQ + self.PV, mul(dae.x[self.ucd], dae.x[self.Id]) + mul(dae.x[self.ucq], dae.x[self.Iq]), iudc, 0.5) \
                           - dae.y[self.Idcy]

        # interface equations
        dae.y[self.a] += mul(self.usq, dae.x[self.Id])
        dae.y[self.v] += mul(self.usq, dae.x[self.Iq])
        dae.y[self.v1] += mul(self.VV + self.VQ, dae.x[self.Idcx]) + mul(self.PQ + self.PV, dae.y[self.Idcy])
        dae.y[self.v2] -= mul(self.VV + self.VQ, dae.x[self.Idcx]) + mul(self.PQ + self.PV, dae.y[self.Idcy])

        dae.add_jac(Gy0, -1, self.Idcy, self.Idcy)
        dae.add_jac(Gx, mul(dae.x[self.Id], 0.5 * iudc), self.Idcy, self.ucd)
        dae.add_jac(Gx, mul(dae.x[self.ucd], 0.5* iudc), self.Idcy, self.Id)
        dae.add_jac(Gx, mul(dae.x[self.Iq], 0.5 * iudc), self.Idcy, self.ucq)
        dae.add_jac(Gx, mul(dae.x[self.ucq], 0.5* iudc), self.Idcy, self.Iq)
        dae.add_jac(Gy, mul(-0.5 * iudc **2, mul(dae.x[self.ucd], dae.x[self.Id]) + mul(dae.x[self.ucq], dae.x[self.Iq])), self.Idcy, self.v1)

    def jac0(self, dae):
        dae.add_jac(Gy0, self.PV + self.VV, self.uref, self.uref)
        dae.add_jac(Gy0, self.PQ + self.PV, self.pref, self.pref)
        dae.add_jac(Gy0, self.PQ + self.VQ, self.qref, self.qref)
        dae.add_jac(Gy0, self.VV + self.VQ, self.udcref, self.udcref)

        dae.add_jac(Gy0, -1.0, self.Idref, self.Idref)
        dae.add_jac(Gx0, 1.0, self.Idref, self.Nd)
        dae.add_jac(Gy0, mul(self.VV + self.VQ, self.Kp3), self.Idref, self.uref)
        dae.add_jac(Gy0, -mul(self.VV + self.VQ, self.Kp3), self.Idref, self.v)

        dae.add_jac(Gy0, -1.0, self.Iqref, self.Iqref)
        dae.add_jac(Gy0, mul(self.PQ + self.PV, div(1.0, self.usq) + self.Kp4, self.Iqref, self.pref))
        dae.add_jac(Gx0, self.PQ + self.PV, self.Iqref, self.Nq)

        dae.add_jac(Fx0, -mul(self.VV + self.VQ, self.iTdc), self.Idcx, self.Idcx)
        dae.add_jac(Fy0, mul(self.VV + self.VQ, self.iTdc, self.Kpdc), self.Idcx, self.udcref)
        dae.add_jac(Fy0, -mul(self.VV + self.VQ, self.iTdc, self.Kpdc), self.Idcx, self.v1)
        dae.add_jac(Fx0, mul(self.VV + self.VQ, self.iTdc), self.Idcx, self.Nq)

    def gycall(self, dae):
        dae.add_jac(Gy, mul(self.PQ + self.PV, div(1.0, self.usq) + self.Kp2), self.Idcref, self.qref)
        dae.add_jac(Gx, -mul(self.PQ + self.PV, self.Kp2, self.usq), self.Idref, self.Id)
        dae.add_jac(Gy, mul(self.PQ + self.PV,
                             -div(dae.y[self.qref], self.usq ** 2) - mul(self.Kp2, dae.x[self.Id])), self.Idref, self.v)

        dae.add_jac(Gy, -mul(self.PQ + self.PV, self.Kp4, dae.x[self.Iq]))
        dae.add_jac(Gx, -mul(self.PQ + self.PV, self.Kp4, dae.y[self.v]))

        iucq = div(1, dae.x[self.ucq])
        dae.add_jac(Gx, mul(self.VV + self.VQ, 2 * (dae.y[self.v1] - dae.y[self.v2]), iucq), self.Iqref, self.Idc)
        dae.add_jac(Gy, mul(self.VV + self.VQ, 2 * dae.x[self.Idc], iucq), self.Iqref, self.v1)
        dae.add_jac(Gx, mul(self.VV + self.VQ, -dae.x[self.ucd], iucq), self.Iqref, self.Id)
        dae.add_jac(Gx, mul(self.VV + self.VQ, -dae.x[self.Iq], iucq,), self.Iqref, self.ucd)
        dae.add_jac(Gx, mul(self.VV + self.VQ, mul(2 * dae.x[self.Idc], (dae.y[self.v1] - dae.y[self.v2])) - mul(dae.x[self.Id], dae.x[self.ucd]), - iucq ** 2))

    def fcall(self, dae):
        dae.f[self.Id] = - mul(self.rsh, self.iLsh, dae.x[self.Id]) + dae.x[self.Iq] + mul(self.iLsh, dae.x[self.ucd] - self.usd)
        dae.f[self.Iq] = - mul(self.rsh, self.ishL, dae.x[self.Iq]) - dae.x[self.Id] + mul(self.iLsh, dae.x[self.ucq] - self.usq)

        dae.f[self.Md] = mul(self.Ki1, dae.y[self.Idref] - dae.x[self.Id])
        dae.f[self.Mq] = mul(self.Ki1, dae.y[self.Iqref] - dae.x[self.Iq])

        dae.f[self.ucd] = -mul(self.Kp1, self.iTt, dae.x[self.Id]) - mul(self.xsh, self.iTt, dae.x[self.Iq]) \
                          - mul(self.iTt, dae.x[self.ucd]) + mul(self.iTt, dae.x[self.Md]) \
                          + mul(self.Kp1, self.iTt, dae.y[self.Idref]) + mul(self.iTt, self.usd)

        dae.f[self.ucq] = -mul(self.Kp1, self.iTt, dae.x[self.Iq]) + mul(self.xsh, self.iTt, dae.x[self.Id]) \
                          - mul(self.iTt, dae.x[self.ucq]) + mul(self.iTt, dae.x[self.Mq]) \
                          + mul(self.Kp1, self.iTt, dae.y[self.Iqref]) + mul(self.iTt, self.usq)

        dae.f[self.Nd] = mul(self.Ki2, dae.y[self.qref] - mul(self.usq, dae.x[self.Id]), self.PQ + self.VQ) \
                         + mul(self.Ki3, dae.y[self.uref] - self.usq, self.PV + self.VV)  # Q or Vac control

        dae.f[self.Nq] = mul(self.Ki4, dae.y[self.pref] - mul(self.usq, dae.x[self.Iq]), self.PQ + self.PV) \
                         + mul(self.Kidc, dae.y[self.udcref] - (dae.y[self.v1] - dae.y[self.v2]), self.VV + self.VQ)

        dae.f[self.Idcx] = mul(self.VV + self.VQ, self.iTdc, -dae.x[self.Idcx] + mul(self.Kpdc, dae.y[self.udcref] - (dae.y[self.v1] - dae.y[self.v2])) + dae.x[self.Nq]) \
                           + mul(self.PQ + self.PV, dae.x[self.Idcx])

    def fcall_jac(self, dae):
        dae.add_jac(Fx0, -mul(self.rsh, self.iLsh), self.Id, self.Id)
        dae.add_jac(Fx0, 1.0, self.Id, self.Iq)

        dae.add_jac(Fx0, self.PV + self.PQ, self.Idcx, self.Idcx)
