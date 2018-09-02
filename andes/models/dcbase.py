from cvxopt import matrix, spmatrix, spdiag, mul, div  # NOQA
from .base import ModelBase
from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA


class Node(ModelBase):
    """ DC node class"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'Topology'
        self._name = 'Node'
        self.param_remove('Sn')
        self.param_remove('Vn')
        self._params.extend([
            'Vdcn',
            'Idcn',
            'voltage',
        ])
        self._descr.update({
            'Vdcn': 'DC voltage rating',
            'Idcn': 'DC current rating',
            'voltage': 'Initial nodal voltage guess',
            'area': 'Area code',
            'region': 'Region code',
            'xcoord': 'x coordinate',
            'ycoord': 'y coordinate'
        })
        self._data.update({
            'Vdcn': 100.0,
            'Idcn': 10.0,
            'area': 0,
            'region': 0,
            'voltage': 1.0,
            'xcoord': None,
            'ycoord': None,
        })
        self._units.update({
            'Vdcn': 'kV',
            'Idcn': 'kA',
            'area': 'na',
            'region': 'na',
            'voltage': 'pu',
            'xcoord': 'deg',
            'ycoord': 'deg',
        })
        self.calls.update({
            'init0': True,
            'pflow': True,
            'jac0': True,
        })
        self._mandatory = ['Vdcn']
        self._zeros = ['Vdcn', 'Idcn']
        # self.v = list()
        self._algebs.extend(['v'])
        self._fnamey.extend(['V_{dc}'])
        self._init()

    def elem_add(self, idx=None, name=None, **kwargs):
        super().elem_add(idx, name, **kwargs)

    def _varname(self):
        if not self.n:
            return
        self.system.varname.append(
            listname='unamey',
            xy_idx=self.v,
            var_name='Vdc',
            element_name=self.name)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=self.v,
            var_name='V_{dc}',
            element_name=self.name)

    def setup(self):
        self._param_to_matrix()

    def init0(self, dae):
        dae.y[self.v] = matrix(self.voltage, (self.n, 1), 'd')

    def jac0(self, dae):
        if self.n is 0:
            return
        dae.add_jac(Gy0, -1e-6, self.v, self.v)


class DCBase(ModelBase):
    """Two-terminal DC device base"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'DCBasic'
        self._params.remove('Sn')
        self._params.remove('Vn')
        self._data.update({
            'node1': None,
            'node2': None,
            'Vdcn': 100.0,
            'Idcn': 10.0,
        })
        self._params.extend(['Vdcn', 'Idcn'])
        self._descr.update({
            'Vdcn': 'DC voltage rating',
            'Idcn': 'DC current rating',
            'node1': 'DC node 1 idx',
            'node2': 'DC node 2 idx',
        })
        self._units.update({
            'Vdcn': 'kV',
            'Idcn': 'A',
        })
        self._dc = {
            'node1': 'v1',
            'node2': 'v2',
        }
        self._mandatory.extend(['node1', 'node2', 'Vdcn'])

    @property
    def v12(self):
        return self.system.dae.y[self.v1] - self.system.dae.y[self.v2]


class R(DCBase):
    """DC Resistence line class"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'R'
        self._params.extend(['R'])
        self._data.update({
            'R': 0.01,
        })
        self._r.extend(['R'])
        self.calls.update({
            'pflow': True,
            'gcall': True,
            'jac0': True,
        })
        self._algebs.extend(['Idc'])
        self._fnamey = ['I_{dc}']
        self._init()

    def gcall(self, dae):
        dae.g[self.Idc] = div(self.v12, self.R) + dae.y[self.Idc]
        dae.g -= spmatrix(dae.y[self.Idc], self.v1, [0] * self.n, (dae.m, 1),
                          'd')
        dae.g += spmatrix(dae.y[self.Idc], self.v2, [0] * self.n, (dae.m, 1),
                          'd')

    def jac0(self, dae):
        dae.add_jac(Gy0, -self.u, self.v1, self.Idc)
        dae.add_jac(Gy0, self.u, self.v2, self.Idc)
        dae.add_jac(Gy0, div(self.u, self.R), self.Idc, self.v1)
        dae.add_jac(Gy0, -div(self.u, self.R), self.Idc, self.v2)
        dae.add_jac(Gy0, self.u - 1e-6, self.Idc, self.Idc)


class L(DCBase):
    """Pure inductive line"""

    def __init__(self, system, name):
        super(L, self).__init__(system, name)
        self._name = 'L'
        self._data.update({'L': 0.001})
        self._params.extend(['L'])
        self._r.extend(['L'])
        self._algebs.extend(['Idc'])
        self._fnamey.extend(['I_{dc}'])
        self._states.extend(['IL'])
        self._fnamex.extend(['I_L'])
        self._service.extend(['iL'])
        self.calls.update({
            'pflow': True,
            'init0': True,
            'gcall': True,
            'fcall': True,
            'jac0': True,
        })
        self._init()

    def servcall(self, dae):
        self.iL = div(self.u, self.L)

    def init0(self, dae):
        self.servcall(dae)

    def gcall(self, dae):
        dae.g[self.Idc] = dae.x[self.IL] + dae.y[self.Idc]
        dae.g -= spmatrix(dae.y[self.Idc], self.v1, [0] * self.n, (dae.m, 1),
                          'd')
        dae.g += spmatrix(dae.y[self.Idc], self.v2, [0] * self.n, (dae.m, 1),
                          'd')

    def fcall(self, dae):
        dae.f[self.IL] = mul(self.v12, self.iL)

    def jac0(self, dae):
        dae.add_jac(Gx0, self.u, self.Idc, self.IL)
        dae.add_jac(Gy0, self.u, self.Idc, self.Idc)
        dae.add_jac(Fy0, self.iL, self.IL, self.v1)
        dae.add_jac(Fy0, -self.iL, self.IL, self.v2)
        dae.add_jac(Gy0, -self.u, self.v1, self.Idc)
        dae.add_jac(Gy0, self.u, self.v2, self.Idc)


class C(DCBase):
    """Pure capacitive line"""

    def __init__(self, system, name):
        super(C, self).__init__(system, name)
        self._name = 'C'
        self._data.update({'C': 0.001})
        self._params.extend(['C'])
        self._g.extend(['C'])
        self._algebs.extend(['Idc'])
        self._fnamey.extend(['I_{dc}'])
        self._states.extend(['vC'])
        self._fnamex.extend(['vC'])
        self._service.extend(['iC'])
        self.calls.update({
            'pflow': True,
            'init0': True,
            'gcall': True,
            'fcall': True,
            'jac0': True,
        })
        self._init()

    def servcall(self, dae):
        self.iC = div(self.u, self.C)

    def init0(self, dae):
        self.servcall(dae)

    def gcall(self, dae):
        dae.g[self.Idc] = dae.x[self.vC] - self.v12
        dae.g -= spmatrix(dae.y[self.Idc], self.v1, [0] * self.n, (dae.m, 1),
                          'd')
        dae.g += spmatrix(dae.y[self.Idc], self.v2, [0] * self.n, (dae.m, 1),
                          'd')

    def fcall(self, dae):
        dae.f[self.vC] = -mul(dae.y[self.Idc], self.iC)

    def jac0(self, dae):
        dae.add_jac(Gx0, self.u, self.Idc, self.vC)
        dae.add_jac(Gy0, -self.u, self.Idc, self.v1)
        dae.add_jac(Gy0, self.u, self.Idc, self.v2)
        dae.add_jac(Gy0, 1e-6, self.Idc, self.Idc)

        dae.add_jac(Gy0, -self.u, self.v1, self.Idc)
        dae.add_jac(Gy0, self.u, self.v2, self.Idc)

        dae.add_jac(Fy0, -self.iC, self.vC, self.Idc)


class RLs(DCBase):
    """DC Resistive and Inductive line"""

    def __init__(self, system, name):
        super(RLs, self).__init__(system, name)
        self._name = 'RLs'
        self._params.extend(['R', 'L'])
        self._data.update({
            'R': 0.01,
            'L': 0.001,
        })
        self._params.extend(['R', 'L'])
        self._r.extend(['R', 'L'])
        self._algebs.extend(['Idc'])
        self._fnamey.extend(['I_{dc}'])
        self._states.extend(['IL'])
        self._fnamex.extend(['I_L'])
        self.calls.update({
            'pflow': True,
            'init0': True,
            'gcall': True,
            'fcall': True,
            'jac0': True,
        })
        self._service.extend(['iR', 'iL'])
        self._init()

    def base(self):
        super(RLs, self).base()

    def servcall(self, dae):
        self.iR = div(self.u, self.R)
        self.iL = div(self.u, self.L)

    def init0(self, dae):
        self.servcall(dae)
        dae.x[self.IL] = mul(self.v12, self.iR)
        dae.y[self.Idc] = -dae.x[self.IL]

    def gcall(self, dae):
        dae.g[self.Idc] = mul(self.u, dae.x[self.IL] + dae.y[self.Idc])
        dae.g -= spmatrix(
            mul(self.u, dae.y[self.Idc]), self.v1, [0] * self.n, (dae.m, 1),
            'd')
        dae.g += spmatrix(
            mul(self.u, dae.y[self.Idc]), self.v2, [0] * self.n, (dae.m, 1),
            'd')

    def fcall(self, dae):
        dae.f[self.IL] = mul(self.v12 - mul(self.R, dae.x[self.IL], self.u),
                             self.iL)

    def jac0(self, dae):
        dae.add_jac(Gx0, self.u, self.Idc, self.IL)
        dae.add_jac(Gy0, self.u + 1e-6, self.Idc, self.Idc)
        dae.add_jac(Gy0, -self.u, self.v1, self.Idc)
        dae.add_jac(Gy0, self.u, self.v2, self.Idc)
        dae.add_jac(Fx0, -mul(self.R, self.iL, self.u) + 1e-6, self.IL,
                    self.IL)
        dae.add_jac(Fy0, mul(self.u, self.iL), self.IL, self.v1)
        dae.add_jac(Fy0, -mul(self.u, self.iL), self.IL, self.v2)


class RCp(DCBase):
    """RC parallel line"""

    def __init__(self, system, name):
        super(RCp, self).__init__(system, name)
        self._name = 'RCp'
        self._params.extend(['R', 'C'])
        self._data.update({
            'R': 0.01,
            'C': 0.001,
        })
        self._params.extend(['R', 'C'])
        self._r.extend(['R'])
        self._g.extend(['C'])
        self._algebs.extend(['Idc'])
        self._fnamey.extend(['I_{dc}'])
        self._states.extend(['vC'])
        self._fnamex.extend(['v_C'])
        self.calls.update({
            'pflow': True,
            'init0': True,
            'gcall': True,
            'fcall': True,
            'jac0': True,
        })
        self._service.extend(['iR', 'iC'])
        self._init()

    def servcall(self, dae):
        self.iR = div(self.u, self.R)
        self.iC = div(self.u, self.C)

    def init0(self, dae):
        self.servcall(dae)
        dae.x[self.vC] = self.v12
        dae.y[self.Idc] = -mul(self.v12, self.iR)

    def gcall(self, dae):
        dae.g[self.Idc] = dae.x[self.vC] - self.v12
        dae.g -= spmatrix(dae.y[self.Idc], self.v1, [0] * self.n, (dae.m, 1),
                          'd')
        dae.g += spmatrix(dae.y[self.Idc], self.v2, [0] * self.n, (dae.m, 1),
                          'd')

    def fcall(self, dae):
        dae.f[self.vC] = -mul(dae.y[self.Idc] + mul(dae.x[self.vC], self.iR),
                              self.iC)

    def jac0(self, dae):
        dae.add_jac(Gx0, self.u, self.Idc, self.vC)
        dae.add_jac(Gy0, -self.u, self.Idc, self.v1)
        dae.add_jac(Gy0, self.u, self.Idc, self.v2)
        dae.add_jac(Gy0, 1e-6, self.Idc, self.Idc)

        dae.add_jac(Gy0, -self.u, self.v1, self.Idc)
        dae.add_jac(Gy0, self.u, self.v2, self.Idc)

        dae.add_jac(Fy0, -self.iC, self.vC, self.Idc)
        dae.add_jac(Fx0, -mul(self.iR, self.iC), self.vC, self.vC)


class RLCp(DCBase):
    """RLC parallel line"""

    def __init__(self, system, name):
        super(RLCp, self).__init__(system, name)
        self._name = 'RLCp'
        self._params.extend(['R', 'L', 'C'])
        self._data.update({
            'R': 0.01,
            'L': 0.001,
            'C': 0.001,
        })
        self._params.extend(['R', 'L', 'C'])
        self._r.extend(['R', 'L'])
        self._g.extend(['C'])
        self._algebs.extend(['Idc'])
        self._fnamey.extend(['I_{dc}'])
        self._states.extend(['IL', 'vC'])
        self._fnamex.extend(['I_L', 'v_C'])
        self.calls.update({
            'pflow': True,
            'init0': True,
            'gcall': True,
            'fcall': True,
            'jac0': True,
        })
        self._service.extend(['iR', 'iL', 'iC'])
        self._init()

    def servcall(self, dae):
        self.iR = div(self.u, self.R)
        self.iL = div(self.u, self.L)
        self.iC = div(self.u, self.C)

    def init0(self, dae):
        self.servcall(dae)
        dae.x[self.vC] = self.v12
        dae.y[self.Idc] = -mul(self.v12, self.iR)

    def gcall(self, dae):
        dae.g[self.Idc] = dae.x[self.vC] - self.v12
        dae.g -= spmatrix(dae.y[self.Idc], self.v1, [0] * self.n, (dae.m, 1),
                          'd')
        dae.g += spmatrix(dae.y[self.Idc], self.v2, [0] * self.n, (dae.m, 1),
                          'd')

    def fcall(self, dae):
        dae.f[self.IL] = mul(dae.x[self.vC], self.iL)
        dae.f[self.vC] = -mul(
            dae.y[self.Idc] + mul(dae.x[self.vC], self.iR) + dae.x[self.IL],
            self.iC)

    def jac0(self, dae):
        dae.add_jac(Gx0, self.u, self.Idc, self.vC)
        dae.add_jac(Gy0, -self.u, self.Idc, self.v1)
        dae.add_jac(Gy0, self.u, self.Idc, self.v2)
        dae.add_jac(Gy0, 1e-6, self.Idc, self.Idc)

        dae.add_jac(Gy0, -self.u, self.v1, self.Idc)
        dae.add_jac(Gy0, self.u, self.v2, self.Idc)

        dae.add_jac(Fx0, self.iL, self.IL, self.vC)

        dae.add_jac(Fy0, -self.iC, self.vC, self.Idc)
        dae.add_jac(Fx0, -mul(self.iR, self.iC), self.vC, self.vC)
        dae.add_jac(Fx0, -self.iC, self.vC, self.IL)


class RCs(DCBase):
    """RC series line"""

    def __init__(self, system, name):
        super(RCs, self).__init__(system, name)
        self._name = 'RCs'
        self._data.update({'R': 0.01, 'C': 0.001})
        self._params.extend(['R', 'C'])
        self._r.extend(['R'])
        self._g.extend(['C'])
        self._algebs.extend(['Idc'])
        self._fnamey.extend(['I_{dc}'])
        self._states.extend(['vC'])
        self._fnamex.extend(['vC'])
        self._service.extend(['iC', 'iR'])
        self.calls.update({
            'pflow': True,
            'init0': True,
            'gcall': True,
            'fcall': True,
            'jac0': True,
        })
        self._init()

    def servcall(self, dae):
        self.iC = div(self.u, self.C)
        self.iR = div(self.u, self.R)

    def init0(self, dae):
        self.servcall(dae)

    def gcall(self, dae):
        dae.g[self.Idc] = dae.x[self.vC] - self.v12 - mul(
            dae.y[self.Idc], self.R)
        dae.g -= spmatrix(dae.y[self.Idc], self.v1, [0] * self.n, (dae.m, 1),
                          'd')
        dae.g += spmatrix(dae.y[self.Idc], self.v2, [0] * self.n, (dae.m, 1),
                          'd')

    def fcall(self, dae):
        dae.f[self.vC] = -mul(dae.y[self.Idc], self.iC)

    def jac0(self, dae):
        dae.add_jac(Gx0, self.u, self.Idc, self.vC)
        dae.add_jac(Gy0, -self.u, self.Idc, self.v1)
        dae.add_jac(Gy0, self.u, self.Idc, self.v2)
        dae.add_jac(Gy0, 1e-6 - self.R, self.Idc, self.Idc)

        dae.add_jac(Gy0, -self.u, self.v1, self.Idc)
        dae.add_jac(Gy0, self.u, self.v2, self.Idc)

        dae.add_jac(Fy0, -self.iC, self.vC, self.Idc)


class RLCs(DCBase):
    """RLC series"""

    def __init__(self, system, name):
        super(RLCs, self).__init__(system, name)
        self._name = 'RLCs'
        self._params.extend(['R', 'L', 'C'])
        self._data.update({'R': 0.01, 'L': 0.001, 'C': 0.001})
        self._params.extend(['R', 'L', 'C'])
        self._r.extend(['R', 'L'])
        self._g.extend(['C'])
        self._algebs.extend(['Idc'])
        self._fnamey.extend(['I_{dc}'])
        self._states.extend(['IL', 'vC'])
        self._fnamex.extend(['I_L', 'v_C'])
        self.calls.update({
            'pflow': True,
            'init0': True,
            'gcall': True,
            'fcall': True,
            'jac0': True,
        })
        self._service.extend(['iR', 'iL', 'iC'])
        self._init()

    def servcall(self, dae):
        self.iR = div(self.u, self.R)
        self.iL = div(self.u, self.L)
        self.iC = div(self.u, self.C)

    def init0(self, dae):
        self.servcall(dae)
        dae.x[self.vC] = self.v12

    def gcall(self, dae):
        dae.g[self.Idc] = dae.x[self.IL] + dae.y[self.Idc]
        dae.g -= spmatrix(dae.y[self.Idc], self.v1, [0] * self.n, (dae.m, 1),
                          'd')
        dae.g += spmatrix(dae.y[self.Idc], self.v2, [0] * self.n, (dae.m, 1),
                          'd')

    def fcall(self, dae):
        dae.f[self.IL] = mul(
            self.v12 - mul(self.R, dae.x[self.IL]) - dae.x[self.vC], self.iL)
        dae.f[self.vC] = mul(dae.x[self.IL], self.iC)

    def jac0(self, dae):
        dae.add_jac(Gx0, self.u, self.Idc, self.IL)
        dae.add_jac(Gy0, self.u, self.Idc, self.Idc)

        dae.add_jac(Gy0, -self.u, self.v1, self.Idc)
        dae.add_jac(Gy0, self.u, self.v2, self.Idc)

        dae.add_jac(Fx0, -mul(self.R, self.iL), self.IL, self.IL)
        dae.add_jac(Fy0, self.iL, self.IL, self.v1)
        dae.add_jac(Fy0, -self.iL, self.IL, self.v2)
        dae.add_jac(Fx0, -self.iL, self.IL, self.vC)

        dae.add_jac(Fx0, self.iC, self.vC, self.IL)


class Ground(DCBase):
    """DC Ground node"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self.param_remove('node1')
        self.param_remove('node2')
        self.param_remove('v')
        self._data.update({
            'node': None,
            'voltage': 0.0,
        })
        self._algebs.extend(['I'])
        self._unamey = ['I']
        self._fnamey = ['I']

        self._dc = {'node': 'v'}
        self._params.extend(['voltage'])
        self._mandatory.extend(['node'])
        self.calls.update({
            'init0': True,
            'jac0': True,
            'gcall': True,
            'pflow': True,
        })

        self._init()

    def init0(self, dae):
        dae.y[self.v] = self.voltage

    def gcall(self, dae):
        dae.g[self.v] -= dae.y[self.I]
        dae.g[self.I] = self.voltage - dae.y[self.v]

    def jac0(self, dae):
        dae.add_jac(Gy0, -self.u, self.v, self.I)
        dae.add_jac(Gy0, self.u - 1 - 1e-6, self.I, self.I)
        dae.add_jac(Gy0, -self.u, self.I, self.v)


class DCgen(DCBase):
    """DC generator to impose active power injection"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'DCgen'
        self._params.extend(['P', 'Sn'])
        self._data.update({
            'P': 0.0,
        })
        self._powers.extend(['P'])
        self.calls.update({
            'pflow': True,
            'gcall': True,
            'stagen': True,
        })
        self._init()

    def gcall(self, dae):
        dae.g -= spmatrix(
            div(mul(self.u, self.P), self.v12), self.v1, [0] * self.n,
            (dae.m, 1), 'd')
        dae.g -= spmatrix(-div(mul(self.u, self.P), self.v12), self.v2,
                          [0] * self.n, (dae.m, 1), 'd')

    def disable_gen(self, idx):
        self.u[self.uid[idx]] = 0
        self.system.dae.factorize = True
