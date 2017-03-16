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
        self._fnamey.extend(['\\theta_{sh}', 'V_{sh}', 'P_{sh}', 'Q_{sh}', 'P_{dc}', 'I_{sh}'])
        self._mandatory.extend(['bus', 'control'])
        self._service.extend(['Zsh', 'Ysh', 'glim', 'ylim', 'vdcref', 'R',
                              'PQ', 'PV', 'vV', 'vQ'])
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
        self.vV = zeros(self.n, 1)
        self.vQ = zeros(self.n, 1)
        for idx, cc in enumerate(self.control):
            if cc not in ['PQ', 'PV', 'vV', 'vQ']:
                raise KeyError('VSC {0} control parameter {1} is invalid.'.format(self.names[idx], cc))
            self.__dict__[cc][idx] = 1

    def init0(self, dae):
        # behind-transformer AC theta_sh and V_sh - must assign first
        dae.y[self.ash] = dae.y[self.a] + 1e-6
        dae.y[self.vsh] = mul(self.v0, 1 - self.vV) + mul(self.vref0, self.vV) + 1e-6

        Vm = polar(dae.y[self.v], dae.y[self.a] * 1j)
        Vsh = polar(dae.y[self.vsh], dae.y[self.ash] * 1j)  # Initial value for Vsh
        IshC = conj(div(Vsh - Vm, self.Zsh))
        Ssh = mul(Vsh, IshC)

        # PQ PV and V control initials on converters
        dae.y[self.psh] = mul(self.pref0, self.PQ + self.PV)
        dae.y[self.qsh] = mul(self.qref0, self.PQ)
        dae.y[self.v1] = dae.y[self.v2] + mul(dae.y[self.v1], 1 - self.vV) + mul(self.vdcref0, self.vV)

        # PV and V control on AC buses
        dae.y[self.v] = mul(dae.y[self.v], 1 - self.PV - self.vV) + mul(self.vref0, self.PV + self.vV)

        # Converter current initial
        dae.y[self.Ish] = abs(IshC)

        # Converter dc power output
        dae.y[self.pdc] = mul(Vsh, IshC).real() + \
                          (self.k0 + mul(self.k1, dae.y[self.Ish]) + mul(self.k2, mul(dae.y[self.Ish], dae.y[self.Ish])))

    def gcall(self, dae):
        if sum(self.u) == 0:
            return

        Vm = polar(dae.y[self.v], dae.y[self.a])
        Vsh = polar(dae.y[self.vsh], dae.y[self.ash])
        Ish = mul(self.Ysh, Vsh - Vm)
        IshC = conj(Ish)
        Ssh = mul(Vm, IshC)

        # check the Vsh and Ish limits during PF iterations
        vupper = list(abs(Vsh) - self.vshmax)
        vlower = list(abs(Vsh) - self.vshmin)
        iupper = list(abs(IshC) - self.Ishmax)
        # check for Vsh and Ish limit violations
        if self.system.SPF.iter >= self.system.SPF.ipv2pq:
            for i in range(self.n):
                if self.u[i] and (vupper[i] > 0 or vlower[i] <0 or iupper[i] > 0):
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
        dae.g[self.a] -= mul(self.u, dae.y[self.psh])  # active power load
        dae.g[self.v] -= mul(self.u, dae.y[self.qsh])  # reactive power load

        # DC interfaces - current
        above = list(dae.y[self.v1] - self.vhigh)
        below = list(dae.y[self.v1] - self.vlow)
        above = matrix([1 if i > 0 else 0 for i in above])
        below = matrix([1 if i < 0 else 0 for i in below])
        self.R = mul(above or below, self.K)
        self.vdcref = mul(self.droop, above, self.vhigh) + mul(self.droop, below, self.vlow)
        dae.g[self.v1] -= div(mul(self.u, dae.y[self.pdc]), dae.y[self.v1] - dae.y[self.v2])  # current injection
        dae.g += spmatrix(div(mul(self.u, dae.y[self.pdc]), dae.y[self.v1] - dae.y[self.v2]), self.v2, [0]*self.n, (dae.m, 1), 'd')  # negative current injection

        dae.g[self.ash] = mul(self.u, Ssh.real() - dae.y[self.psh])  # (2)
        dae.g[self.vsh] = mul(self.u, Ssh.imag() - dae.y[self.qsh])  # (3)

        # PQ, PV or V control
        dae.g[self.psh] = mul(dae.y[self.psh] + mul(self.R, dae.y[self.v1] - self.vdcref) - self.pref0, self.PQ + self.PV, self.u) + mul((dae.y[self.v1] - dae.y[self.v2]) - self.vdcref0, self.vV + self.vQ, self.u)  # (12), (15)
        dae.g[self.qsh] = mul(dae.y[self.qsh] - self.qref0, self.PQ + self.vQ, self.u) + mul(dae.y[self.v] - self.vref0, self.PV + self.vV, self.u)  # (13), (16)

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
                    self.switch(comp, 'Q')
                else:
                    idx = self.psh[comp]
                    self.switch(comp, 'P')

                self.system.DAE.factorize = True
                dae.g[idx] = dae.y[yidx] - ylim
                if idx not in self.glim:
                    self.glim.append(idx)
                if yidx not in self.ylim:
                    self.ylim.append(yidx)

        dae.g[self.Ish] = mul(self.u, abs(IshC) - dae.y[self.Ish])  # (10)

        dae.g[self.pdc] = mul(self.u,
                              mul(Vsh, IshC).real() - dae.y[self.pdc] + (self.k0 + mul(self.k1, dae.y[self.Ish]) + mul(self.k2, dae.y[self.Ish] ** 2))
                              )

    def switch(self, idx, control):
        """Switch a single control of <idx>"""
        old = None
        new = None
        if control == 'Q':
            if self.PQ[idx] == 1:
                old = 'PQ'
                new = 'PV'
            elif self.vQ[idx] == 1:
                old = 'vQ'
                new = 'vV'
        elif control == 'P':
            if self.PQ[idx] == 1:
                old = 'PQ'
                new = 'vQ'
            elif self.PV[idx] == 1:
                old = 'PV'
                new = 'vV'
        elif control == 'V':
            if self.PV[idx] == 1:
                old = 'PV'
                new = 'PQ'
            elif self.vV[idx] == 1:
                old = 'vV'
                new = 'vQ'
        elif control == 'v':
            if self.vQ[idx] == 1:
                old = 'vQ'
                new = 'PQ'
            elif self.vV[idx] == 1:
                old = 'vV'
                new = 'PV'
        if old and new:
            self.__dict__[old][idx] = 0
            self.__dict__[new][idx] = 1

    def gycall(self, dae):
        if sum(self.u) == 0:
            return
        Zsh = self.rsh + 1j * self.xsh
        iZsh = div(self.u, abs(Zsh))
        Vh = polar(dae.y[self.v], dae.y[self.a] * 1j)
        Vsh = polar(dae.y[self.vsh], dae.y[self.ash] * 1j)
        Ish = div(Vsh - Vh + 1e-6, Zsh)
        iIsh = div(self.u, Ish)

        gsh = div(self.u, Zsh).real()
        bsh = div(self.u, Zsh).imag()
        V = dae.y[self.v]
        theta = dae.y[self.a]
        Vsh = dae.y[self.vsh]
        thetash = dae.y[self.ash]
        Vdc = dae.y[self.v1] - dae.y[self.v2]
        iVdc2 = div(self.u, Vdc ** 2)

        dae.add_jac(Gy, -div(self.u, Vdc), self.v1, self.pdc)
        dae.add_jac(Gy, mul(dae.y[self.pdc], iVdc2), self.v1, self.v1)
        dae.add_jac(Gy, -mul(dae.y[self.pdc], iVdc2), self.v1, self.v2)

        dae.add_jac(Gy, div(self.u, Vdc), self.v2, self.pdc)
        dae.add_jac(Gy, -mul(dae.y[self.pdc], iVdc2), self.v2, self.v1)
        dae.add_jac(Gy, mul(dae.y[self.pdc], iVdc2), self.v2, self.v2)

        dae.add_jac(Gy, -2*mul(gsh, V) + mul(gsh, Vsh, cos(theta - thetash)) + mul(bsh, Vsh, sin(theta - thetash)), self.ash, self.v)
        dae.add_jac(Gy, mul(gsh, V, cos(theta - thetash)) + mul(bsh, V, sin(theta - thetash)), self.ash, self.vsh)
        dae.add_jac(Gy, -mul(gsh, V, Vsh, sin(theta - thetash)) + mul(bsh, V, Vsh, cos(theta - thetash)), self.ash, self.a)
        dae.add_jac(Gy, mul(gsh, V, Vsh, sin(theta - thetash)) - mul(bsh, V, Vsh, cos(theta - thetash)), self.ash, self.ash)

        dae.add_jac(Gy, 2*mul(bsh, V) + mul(gsh, Vsh, sin(theta - thetash)) - mul(bsh, Vsh, cos(theta - thetash)), self.vsh, self.v)
        dae.add_jac(Gy, mul(gsh, V, sin(theta - thetash)) - mul(bsh, V, cos(theta - thetash)) + 1e-6, self.vsh, self.vsh)
        dae.add_jac(Gy, mul(gsh, V, Vsh, cos(theta - thetash)) + mul(bsh, V, Vsh, sin(theta - thetash)), self.vsh, self.a)
        dae.add_jac(Gy, -mul(gsh, V, Vsh, cos(theta - thetash)) - mul(bsh, V, Vsh, sin(theta - thetash)), self.vsh, self.ash)

        dae.add_jac(Gy, 0.5 * mul(self.u, 2*V - 2*mul(Vsh, cos(theta - thetash)), abs(iIsh), abs(iZsh) ** 2), self.Ish, self.v)
        dae.add_jac(Gy, 0.5 * mul(self.u, 2*Vsh - 2*mul(V, cos(theta - thetash)), abs(iIsh), abs(iZsh) ** 2), self.Ish, self.vsh)
        dae.add_jac(Gy, 0.5 * mul(self.u, 2*V, Vsh, sin(theta-thetash), abs(iIsh), abs(iZsh) ** 2), self.Ish, self.a)
        dae.add_jac(Gy, 0.5 * mul(self.u, 2*V, Vsh, - sin(theta - thetash), abs(iIsh), abs(iZsh) ** 2), self.Ish, self.ash)

        dae.add_jac(Gy, -2 * mul(self.u, self.k2, dae.y[self.Ish]), self.pdc, self.Ish)

        dae.add_jac(Gy, mul(2 * gsh, Vsh) - mul(gsh, V, cos(theta - thetash)) + mul(bsh, V, sin(theta - thetash)), self.pdc, self.vsh)
        dae.add_jac(Gy, -mul(gsh, Vsh, cos(theta - thetash)) + mul(bsh, Vsh, sin(theta - thetash)), self.pdc, self.v)
        dae.add_jac(Gy, mul(gsh, V, Vsh, sin(theta - thetash)) + mul(bsh, V, Vsh, cos(theta - thetash)), self.pdc, self.a)
        dae.add_jac(Gy, -mul(gsh, V, Vsh, sin(theta - thetash)) - mul(bsh, V, Vsh, cos(theta - thetash)), self.pdc, self.ash)

        for gidx, yidx in zip(self.glim, self.ylim):
            dae.set_jac(Gy, 0.0, [gidx] * dae.m, range(dae.m))
            dae.set_jac(Gy, 1.0, [gidx], [yidx])
            dae.set_jac(Gy, 1e-6, [gidx], [gidx])

    def jac0(self, dae):
        dae.add_jac(Gy0, 1e-6, self.v1, self.v1)
        dae.add_jac(Gy0, 1e-6, self.v2, self.v2)
        dae.add_jac(Gy0, 1e-6, self.ash, self.ash)
        dae.add_jac(Gy0, 1e-6, self.vsh, self.vsh)

        dae.add_jac(Gy0, -self.u, self.ash, self.psh)
        dae.add_jac(Gy0, -self.u, self.vsh, self.qsh)

        dae.add_jac(Gy0, mul(self.u, self.PQ + self.PV) + 1e-6, self.psh, self.psh)
        dae.add_jac(Gy0, mul(self.u, self.vV), self.psh, self.v1)
        dae.add_jac(Gy0, -mul(self.u, self.vV), self.psh, self.v2)
        dae.add_jac(Gy0, mul(self.PV + self.PQ, self.u, self.R), self.psh, self.v1)

        dae.add_jac(Gy0, mul(self.u, self.PQ) + 1e-6, self.qsh, self.qsh)
        dae.add_jac(Gy0, mul(self.u, self.PV + self.vV), self.qsh, self.v)

        dae.add_jac(Gy0, -self.u, self.a, self.psh)
        dae.add_jac(Gy0, -self.u, self.v, self.qsh)

        dae.add_jac(Gy0, -self.u + 1e-6, self.Ish, self.Ish)

        dae.add_jac(Gy0, -self.u + 1e-6, self.pdc, self.pdc)
        dae.add_jac(Gy0, mul(self.u, self.k1), self.pdc, self.Ish)

    def disable(self, idx):
        """Disable an element and reset the outputs"""
        if idx not in self.int.keys():
            self.message('Element index {0} does not exist.'.format(idx))
            return
        self.u[self.int[idx]] = 0
        # self.system.DAE.y[self.psh] = 0
        # self.system.DAE.y[self.qsh] = 0
        # self.system.DAE.y[self.pdc] = 0
        # self.system.DAE.y[self.Ish] = 0


class VSCDyn(DCBase):
    """Shunt-connected dynamic VSC model for transient simulation"""
    def __init__(self, system, name):
        super(VSCDyn, self).__init__(system, name)
        self._group = 'AC/DC'
        self._name = 'VSCDyn'
        self._ac = {'bus': ['a', 'v']}
        self._data.update({'vsc': None,
                           'Kp1': 0.2,
                           'Ki1': 0.5,
                           'Kp2': 2,
                           'Ki2': 1,
                           'Kp3': 2,
                           'Ki3': 1,
                           'Kp4': 2,
                           'Ki4': 1,
                           'Kpdc': 1,
                           'Kidc': 2,
                           'Tt': 0.02,
                           'Tdc': 0.02,
                           'Cdc': 0.1,
                           })
        self._params.extend(['vsc', 'Kp1', 'Ki1', 'Kp2', 'Ki2', 'Kp3', 'Ki3', 'Kp4', 'Ki4', 'Kpdc', 'Kidc',
                             'Tt', 'Tdc', 'Cdc'])
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
                            'Tdc': 'dc voltage time constant',
                            'Cdc': 'dc interface shunt capacitor',
                            })
        self._algebs.extend(['vref', 'qref', 'pref', 'vdcref', 'Idref', 'Iqref', 'Idcy', 'ICdc'])
        self._states.extend(['Id', 'Iq', 'Md', 'Mq', 'ucd', 'ucq', 'Nd', 'Nq', 'Idcx', 'vCdc'])
        self._service.extend(['rsh', 'xsh', 'iLsh', 'wn', 'usd', 'usq', 'iTt', 'iTdc', 'PQ', 'PV', 'vV', 'vQ', 'adq',
                              'pref0', 'qref0', 'vref0', 'vdcref0'])
        self._mandatory.extend(['vsc'])
        self._zeros.extend(['Tt', 'Tdc', 'Cdc'])
        self._fnamey.extend(['U^{ref}', 'Q^{ref}', 'P^{ref}', 'U_{dc}^{ref}', 'I_d^{ref}', 'I_q^{ref}', 'I_{dcy}', 'I^C_{dc}'])
        self._fnamex.extend(['I_d', 'I_q', 'M_d', 'M_q', 'u_c^d', 'u_c^q', 'N_d', 'N_q', 'I_{dcx}', 'U^C_{dc}'])
        self.calls.update({'init1': True, 'gcall': True,
                           'fcall': True,
                           'gycall': True, 'fxcall': True,
                           'jac0': True,
                           })
        self._mandatory.remove('node1')
        self._mandatory.remove('node2')
        self._mandatory.remove('Vdcn')
        self._ac = {}
        self._dc = {}
        self._inst_meta()

    def setup(self):
        super(VSCDyn, self).setup()

    def init1(self, dae):
        self.copy_param('VSC', src='u', dest='u0', fkey=self.vsc)
        self.copy_param('VSC', src='rsh', fkey=self.vsc)
        self.copy_param('VSC', src='xsh', fkey=self.vsc)
        self.copy_param('VSC', src='PQ', fkey=self.vsc)
        self.copy_param('VSC', src='PV', fkey=self.vsc)
        self.copy_param('VSC', src='vV', fkey=self.vsc)
        self.copy_param('VSC', src='vQ', fkey=self.vsc)
        self.copy_param('VSC', src='a', fkey=self.vsc)
        self.copy_param('VSC', src='v', fkey=self.vsc)
        self.copy_param('VSC', src='ash', fkey=self.vsc)
        self.copy_param('VSC', src='vsh', fkey=self.vsc)
        self.copy_param('VSC', src='v1', fkey=self.vsc)
        self.copy_param('VSC', src='v2', fkey=self.vsc)

        self.copy_param('VSC', src='psh', dest='pref0', fkey=self.vsc)  # copy indices
        self.copy_param('VSC', src='qsh', dest='qref0', fkey=self.vsc)
        self.copy_param('VSC', src='pdc', dest='pdc', fkey=self.vsc)

        self.u = aandb(self.u, self.u0)
        self.PQ = mul(self.u, self.PQ)
        self.PV = mul(self.u, self.PV)
        self.vV = mul(self.u, self.vV)
        self.vQ = mul(self.u, self.vQ)

        self.pref0 = dae.y[self.pref0]
        self.qref0 = dae.y[self.qref0]
        self.vref0 = mul(self.PV + self.vV, dae.y[self.v])
        self.vdcref0 = mul(self.vV + self.vQ, dae.y[self.v1] - dae.y[self.v2])

        self.wn = ones(self.n, 1)  # in pu
        self.iLsh = div(1.0, self.xsh)
        self.iTt = div(1.0, self.Tt)
        self.iTdc = div(1.0, self.Tdc)
        self.usd = zeros(self.n, 1)
        self.usq = dae.y[self.v]
        k1 = div(self.rsh, self.xsh) ** 2
        k2 = div(self.rsh, self.xsh ** 2)

        dae.x[self.ucd] = mul(dae.y[self.vsh], -sin(dae.y[self.ash] - dae.y[self.a]))
        dae.x[self.ucq] = mul(dae.y[self.vsh], cos(dae.y[self.ash] - dae.y[self.a]))

        dae.y[self.pref] = mul(self.PV + self. PQ, self.pref0)
        dae.y[self.qref] = mul(self.PQ + self.vQ, self.qref0)
        dae.y[self.vref] = mul(self.PV + self.vV, self.vref0)
        dae.y[self.vdcref] = mul(self.vV + self.vQ, self.vdcref0)

        dae.y[self.Idref] = div(mul(self.u, self.qref0), self.usq)
        dae.x[self.Id] = div(mul(self.u, self.qref0), self.usq)

        dae.y[self.Iqref] = div(mul(self.u, self.pref0), self.usq)
        dae.x[self.Iq] = div(mul(self.u, self.pref0), self.usq)

        dae.x[self.Md] = mul(self.rsh, dae.x[self.Id])
        dae.x[self.Mq] = mul(self.rsh, dae.x[self.Iq])

        Idc = div(mul(dae.x[self.ucd], dae.x[self.Id]) + mul(dae.x[self.ucq], dae.x[self.Iq]),  dae.y[self.v1] - dae.y[self.v2])
        dae.y[self.Idcy] = mul(self.PQ + self.PV, Idc)
        dae.x[self.Idcx] = mul(self.vQ + self.vV, Idc)

        dae.x[self.Nd] = mul(self.PV + self.vV, div(self.qref0, self.usq))
        dae.x[self.Nq] = mul(self.vQ + self.vV, dae.x[self.Idcx])

        dae.y[self.ICdc] = zeros(self.n, 1)
        dae.x[self.vCdc] = dae.y[self.v1]

        for idx in self.vsc:
            self.system.VSC.disable(idx)

    def gcall(self, dae):
        self.adq = dae.y[self.a]
        self.usq = mul(dae.y[self.v], cos(dae.y[self.a] - self.adq))
        self.usd = mul(dae.y[self.v], sin(dae.y[self.a] - self.adq))
        iudc = div(1, dae.y[self.v1] - dae.y[self.v2])
        iucq = div(1, dae.x[self.ucq])

        # 1 - vref(1): y0[vref]
        dae.g[self.vref] = mul(self.PV + self.vV, dae.y[self.vref] - self.vref0)

        # 2 - pref(1): y0[pref]
        dae.g[self.pref] = mul(self.PV + self.PQ, dae.y[self.pref] - self.pref0)

        # 3 - qref(1): y0[qref]
        dae.g[self.qref] = mul(self.PQ + self.vQ, dae.y[self.qref] - self.qref0)

        # 4 - vdcref(1): y0[vdcref]
        dae.g[self.vdcref] = mul(self.vV + self.vQ, dae.y[self.vdcref] - self.vdcref0)

        # 5 - Idref(8): y[qref], y[v], x[Id]  |  y0[vref], y0[v]  |  x0[Nd], y0[Idref]
        dae.g[self.Idref] = mul(self.PQ + self.vQ, div(dae.y[self.qref], self.usq) + mul(self.Kp2, dae.y[self.qref] - mul(self.usq, dae.x[self.Id]))) \
                            + mul(self.PV + self.vV, mul(self.Kp3, dae.y[self.vref] - self.usq)) \
                            + dae.x[self.Nd] - dae.y[self.Idref]

        # 6 - Iqref(11): y[pref], y[v], x0[Nq], x[Iq]  |  x[Idcx], y[v1], y[v2], x[Id], x[ucd], x[ucq] | y0[Iqref]
        Iqref1 = mul(self.PQ + self.PV, div(dae.y[self.pref], self.usq) + dae.x[self.Nq] + mul(self.Kp4, dae.y[self.pref] - mul(self.usq, dae.x[self.Iq])))
        Iqref2 = mul(self.vV + self.vQ, mul(dae.x[self.Idcx], (dae.y[self.v1] - dae.y[self.v2])) - mul(dae.x[self.Id], dae.x[self.ucd]), iucq)
        dae.g[self.Iqref] = Iqref1 + Iqref2 - dae.y[self.Iqref]

        # 7 - Idcy(7): x[ucd], x[Id], x[ucq], x[Iq], y[v1], y[v2], y0[Idcy]
        dae.g[self.Idcy] = mul(self.PQ + self.PV, mul(dae.x[self.ucd], dae.x[self.Id]) + mul(dae.x[self.ucq], dae.x[self.Iq]), iudc) - dae.y[self.Idcy]

        # interface equations
        # 8 - a(2): y[v], x[Id]
        dae.g[self.a] -= mul(self.u, self.usq, dae.x[self.Iq])

        # 9 - v(2): y[v], x[Iq]
        dae.g[self.v] -= mul(self.u, self.usq, dae.x[self.Id])

        # 10 - v1: x0[Idcx]  |  y0[Idcy]
        Idc = mul(self.vV + self.vQ, dae.x[self.Idcx]) + mul(self.PQ + self.PV, dae.y[self.Idcy])
        dae.g[self.v1] -= Idc + dae.y[self.ICdc]

        # 11 - v2: x0[Idcx]  |  y0[Idcy]
        dae.g += spmatrix(Idc, self.v2, [0] * self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(dae.y[self.ICdc], self.v2, [0] * self.n, (dae.m, 1), 'd')

        # 11.1
        dae.g[self.ICdc] = mul(self.u, dae.y[self.v1] - dae.x[self.vCdc])

    def jac0(self, dae):
        # 1 [1], 2[1], 3[1], 4[1]
        dae.add_jac(Gy0, self.PV + self.vV + 1e-6, self.vref, self.vref)
        dae.add_jac(Gy0, self.PQ + self.PV + 1e-6, self.pref, self.pref)
        dae.add_jac(Gy0, self.PQ + self.vQ + 1e-6, self.qref, self.qref)
        dae.add_jac(Gy0, self.vV + self.vQ + 1e-6, self.vdcref, self.vdcref)

        # 5 [vref], [v] [Idref], [Nd]
        dae.add_jac(Gy0, mul(self.PV + self.vV, self.Kp3), self.Idref, self.vref)
        dae.add_jac(Gy0, -mul(self.PV + self.vV, self.Kp3), self.Idref, self.v)
        dae.add_jac(Gy0, -1.0, self.Idref, self.Idref)
        dae.add_jac(Gx0, 1.0, self.Idref, self.Nd)

        # 6 [Nq], [Iqref]
        dae.add_jac(Gx0, self.PQ + self.PV, self.Iqref, self.Nq)
        dae.add_jac(Gy0, -1.0, self.Iqref, self.Iqref)

        # 7 - [Idcy]
        dae.add_jac(Gy0, -self.u + 1e-6, self.Idcy, self.Idcy)

        # 10 - [Idcx], [Idcy]
        dae.add_jac(Gx0, -(self.vV + self.vQ), self.v1, self.Idcx)
        dae.add_jac(Gy0, -(self.PQ + self.PV), self.v1, self.Idcy)
        dae.add_jac(Gy0, -self.u, self.v1, self.ICdc)

        # 11 - [Idcx], [Idcy]
        dae.add_jac(Gx0, self.vV + self.vQ, self.v2, self.Idcx)
        dae.add_jac(Gy0, self.PQ + self.PV, self.v2, self.Idcy)

        # 11.1
        dae.add_jac(Gy0, self.u, self.ICdc, self.v1)
        dae.add_jac(Gx0, -self.u, self.ICdc, self.vCdc)
        dae.add_jac(Gy0, 1e-6, self.ICdc, self.ICdc)

        # 12 - [Id], [Iq], [ucd]
        dae.add_jac(Fx0, -mul(self.rsh, self.iLsh, self.u), self.Id, self.Id)
        dae.add_jac(Fx0, self.u, self.Id, self.Iq)
        dae.add_jac(Fx0, mul(self.u, self.iLsh), self.Id, self.ucd)

        # 13 - [Iq], [Id], [ucq]
        dae.add_jac(Fx0, -mul(self.u, self.rsh, self.iLsh), self.Iq, self.Iq)
        dae.add_jac(Fx0, -self.u, self.Iq, self.Id)
        dae.add_jac(Fx0, mul(self.u, self.iLsh), self.Iq, self.ucq)

        # 14 - [Idref], [Id]
        dae.add_jac(Fy0, self.Ki1, self.Md, self.Idref)
        dae.add_jac(Fx0, -self.Ki1, self.Md, self.Id)
        # dae.add_jac(Fx0, 1e-6, self.Md, self.Md)

        # 15 - [Iqref], [Iq]
        dae.add_jac(Fy0, self.Ki1, self.Mq, self.Iqref)
        dae.add_jac(Fx0, -self.Ki1, self.Mq, self.Iq)
        # dae.add_jac(Fx0, 1e-6, self.Mq, self.Mq)

        # 16 - [Id], [Iq], [ucd], [Md], [Idref]
        dae.add_jac(Fx0, -mul(self.Kp1, self.iTt), self.ucd, self.Id)
        dae.add_jac(Fx0, -mul(self.xsh, self.iTt), self.ucd, self.Iq)
        dae.add_jac(Fx0, -self.iTt, self.ucd, self.ucd)
        dae.add_jac(Fx0, self.iTt, self.ucd, self.Md)
        dae.add_jac(Fy0, mul(self.Kp1, self.iTt), self.ucd, self.Idref)

        # 17 - [Iq], [Id], [ucq], [Mq], [Iqref], [v]
        dae.add_jac(Fx0, -mul(self.Kp1, self.iTt), self.ucq, self.Iq)
        dae.add_jac(Fx0, mul(self.xsh, self.iTt), self.ucq, self.Id)
        dae.add_jac(Fx0, -self.iTt, self.ucq, self.ucq)
        dae.add_jac(Fx0, self.iTt, self.ucq, self.Mq)
        dae.add_jac(Fy0, mul(self.Kp1, self.iTt), self.ucq, self.Iqref)
        dae.add_jac(Fy0, self.iTt, self.ucq, self.v)

        # 18 - [qref], [vref], [v]
        dae.add_jac(Fy0, mul(self.PQ + self.vQ, self.Ki2), self.Nd, self.qref)
        dae.add_jac(Fy0, mul(self.PV + self.vV, self.Ki3), self.Nd, self.vref)
        dae.add_jac(Fy0, -mul(self.PV + self.vV, self.Ki3), self.Nd, self.v)
        # dae.add_jac(Fx0, 1e-6, self.Nd, self.Nd)

        # 19 - [pref], [vdcref], [v1], [v2]
        dae.add_jac(Fy0, mul(self.PQ + self.PV, self.Ki4), self.Nq, self.pref)
        dae.add_jac(Fy0, mul(self.vV + self.vQ, self.Kidc), self.Nq, self.vdcref)
        dae.add_jac(Fy0, -mul(self.vV + self.vQ, self.Kidc), self.Nq, self.v1)
        dae.add_jac(Fy0, mul(self.vV + self.vQ, self.Kidc), self.Nq, self.v2)
        # dae.add_jac(Fx0, 1e-6, self.Nq, self.Nq)

        # 20 - [Idcx], [vdcref], [v1], [Nq], [Idcx]
        dae.add_jac(Fx0, -mul(self.vV + self.vQ, self.iTdc), self.Idcx, self.Idcx)
        dae.add_jac(Fy0, mul(self.vV + self.vQ, self.iTdc, self.Kpdc), self.Idcx, self.vdcref)
        dae.add_jac(Fy0, -mul(self.vV + self.vQ, self.iTdc, self.Kpdc), self.Idcx, self.v1)
        dae.add_jac(Fy0, mul(self.vV + self.vQ, self.iTdc, self.Kpdc), self.Idcx, self.v2)
        dae.add_jac(Fx0, mul(self.vV + self.vQ, self.iTdc), self.Idcx, self.Nq)
        # dae.add_jac(Fx0, mul(self.PQ + self.PV, self.u + 1e-6), self.Idcx, self.Idcx)

        # 21.1
        dae.add_jac(Fy0, -div(self.u, self.Cdc), self.vCdc, self.ICdc)

    def gycall(self, dae):
        iudc = div(1, dae.y[self.v1] - dae.y[self.v2])
        iucq = div(1, dae.x[self.ucq])

        # 5 [qref], [v]
        dae.add_jac(Gy, mul(self.PQ + self.vQ, div(1.0, self.usq) + self.Kp2), self.Idref, self.qref)
        dae.add_jac(Gy, mul(self.PQ + self.vQ,
                             -div(dae.y[self.qref], self.usq ** 2) - mul(self.Kp2, dae.x[self.Id])), self.Idref, self.v)

        # 6 [pref], [v], [v1] [v2]
        dae.add_jac(Gy, mul(self.PQ + self.PV, div(1.0, self.usq) + self.Kp4), self.Iqref, self.pref)
        dae.add_jac(Gy, -mul(self.PQ + self.PV, self.Kp4, dae.x[self.Iq]), self.Iqref, self.v)
        dae.add_jac(Gy, mul(self.vV + self.vQ, dae.x[self.Idcx], iucq), self.Iqref, self.v1)
        dae.add_jac(Gy, -mul(self.vV + self.vQ, dae.x[self.Idcx], iucq), self.Iqref, self.v2)

        # 7 [v1], [v2]
        dae.add_jac(Gy, mul(self.PQ + self.PV, -iudc **2, mul(dae.x[self.ucd], dae.x[self.Id]) + mul(dae.x[self.ucq], dae.x[self.Iq])), self.Idcy, self.v1)
        dae.add_jac(Gy, mul(self.PQ + self.PV, iudc **2, mul(dae.x[self.ucd], dae.x[self.Id]) + mul(dae.x[self.ucq], dae.x[self.Iq])), self.Idcy, self.v2)

        # 8 [v]
        dae.add_jac(Gy, mul(self.u, dae.x[self.Id]), self.a, self.v)

        # 9 [v]
        dae.add_jac(Gy, mul(self.u, dae.x[self.Iq]), self.v, self.v)

    def fcall(self, dae):
        # 12 - Id(3): x0[Id], x0[Iq], x0[ucd]
        dae.f[self.Id] = mul(self.u, - mul(self.rsh, self.iLsh, dae.x[self.Id]) + dae.x[self.Iq] + mul(self.iLsh, dae.x[self.ucd] - self.usd))

        # 13 - Iq(3): x0[Iq], x0[Id], x0[ucq]
        dae.f[self.Iq] = mul(self.u, - mul(self.rsh, self.iLsh, dae.x[self.Iq]) - dae.x[self.Id] + mul(self.iLsh, dae.x[self.ucq] - self.usq))

        # 14 - Md(2): y0[Idref], x0[Id]
        dae.f[self.Md] = mul(self.u, self.Ki1, dae.y[self.Idref] - dae.x[self.Id])

        # 15 - Mq(2): y0[Iqref], x0[Iq]
        dae.f[self.Mq] = mul(self.u, self.Ki1, dae.y[self.Iqref] - dae.x[self.Iq])

        # 16 - ucd(5): x0[Id], x0[Iq], x0[ucd], x0[Md], y0[Idref]
        dae.f[self.ucd] = mul(self.u,
                              -mul(self.Kp1, self.iTt, dae.x[self.Id]) - mul(self.xsh, self.iTt, dae.x[self.Iq]) \
                              - mul(self.iTt, dae.x[self.ucd]) + mul(self.iTt, dae.x[self.Md]) \
                              + mul(self.Kp1, self.iTt, dae.y[self.Idref]) + mul(self.iTt, self.usd))

        # 17 - ucq(6): x0[Iq], x0[Id], x0[ucq], x0[Mq], y0[Iqref], y0[v]
        dae.f[self.ucq] = mul(self.u,
                              -mul(self.Kp1, self.iTt, dae.x[self.Iq]) + mul(self.xsh, self.iTt, dae.x[self.Id]) \
                              - mul(self.iTt, dae.x[self.ucq]) + mul(self.iTt, dae.x[self.Mq]) \
                              + mul(self.Kp1, self.iTt, dae.y[self.Iqref]) + mul(self.iTt, self.usq))

        # 18 - Nd(5): y0[qref], y[v], x[Id]  |  y0[vref], y0[v]
        dae.f[self.Nd] = mul(self.Ki2, dae.y[self.qref] - mul(self.usq, dae.x[self.Id]), self.PQ + self.vQ) \
                         + mul(self.Ki3, dae.y[self.vref] - self.usq, self.PV + self.vV)  # Q or Vac control

        # 19 - Nq(6): y0[pref], x[Iq], y[v]  |  y0[vdcref], y0[v1], y0[v2]
        dae.f[self.Nq] = mul(self.Ki4, dae.y[self.pref] - mul(self.usq, dae.x[self.Iq]), self.PQ + self.PV) + mul(self.Kidc, dae.y[self.vdcref] - (dae.y[self.v1] - dae.y[self.v2]), self.vV + self.vQ)

        # 20 - Idcx(5): x0[Idcx], y0[vdcref], y0[v1], x0[Nq]  |  x0[Idcx]
        dae.f[self.Idcx] = mul(self.vV + self.vQ, self.iTdc, dae.x[self.Nq] - dae.x[self.Idcx] + mul(self.Kpdc, dae.y[self.vdcref] - (dae.y[self.v1] - dae.y[self.v2])) ) \
                           + mul(self.PQ + self.PV, dae.x[self.Idcx])

        # 21.1
        dae.f[self.vCdc] = -div(dae.y[self.ICdc], self.Cdc)

        # self.reset_offline()

    def fxcall(self, dae):
        iudc = div(1, dae.y[self.v1] - dae.y[self.v2])
        iucq = div(1, dae.x[self.ucq])

        # 5 - [Id]
        dae.add_jac(Gx, -mul(self.PQ + self.vQ, self.Kp2, self.usq), self.Idref, self.Id)

        # 6 - [Iq], [Idcx], [Id], [ucd], [ucq]
        dae.add_jac(Gx, -mul(self.PQ + self.PV, self.Kp4, dae.y[self.v]), self.Iqref, self.Iq)
        dae.add_jac(Gx, mul(self.vV + self.vQ, (dae.y[self.v1] - dae.y[self.v2]), iucq), self.Iqref, self.Idcx)
        dae.add_jac(Gx, mul(self.vV + self.vQ, -dae.x[self.ucd], iucq), self.Iqref, self.Id)
        dae.add_jac(Gx, mul(self.vV + self.vQ, -dae.x[self.Id], iucq), self.Iqref, self.ucd)
        dae.add_jac(Gx, mul(self.vV + self.vQ, mul(dae.x[self.Idcx], (dae.y[self.v1] - dae.y[self.v2])) - mul(dae.x[self.Id], dae.x[self.ucd]), - iucq ** 2), self.Iqref, self.ucq)

        # 7 [ucd], [Id], [ucq], [Iq]
        dae.add_jac(Gx, mul(self.PQ + self.PV, dae.x[self.Id], iudc), self.Idcy, self.ucd)
        dae.add_jac(Gx, mul(self.PQ + self.PV, dae.x[self.ucd], iudc), self.Idcy, self.Id)
        dae.add_jac(Gx, mul(self.PQ + self.PV, dae.x[self.Iq], iudc), self.Idcy, self.ucq)
        dae.add_jac(Gx, mul(self.PQ + self.PV, dae.x[self.ucq], iudc), self.Idcy, self.Iq)

        # 8 [Iq]
        dae.add_jac(Gx, mul(self.u, self.usq), self.a, self.Iq)

        # 9 [Iq]
        dae.add_jac(Gx, mul(self.u, self.usq), self.v, self.Iq)

        # 18 [Id], [v]
        dae.add_jac(Fx, -mul(self.PQ + self.vQ, self.Ki2, self.usq), self.Nd, self.Id)
        dae.add_jac(Fy, -mul(self.PQ + self.vQ, self.Ki2, dae.x[self.Id]), self.Nd, self.v)

        # 19 [Iq], [v]
        dae.add_jac(Fx, -mul(self.PQ + self.PV, self.Ki4, self.usq), self.Nq, self.Iq)
        dae.add_jac(Fy, -mul(self.PQ + self.PV, self.Ki4, dae.x[self.Iq]), self.Nq, self.v)
