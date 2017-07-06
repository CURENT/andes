from cvxopt import matrix, spmatrix, spdiag, mul, div
from .base import ModelBase
from ..consts import *


class Node(ModelBase):
    """ DC node class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'Topology'
        self._name = 'Node'
        self.remove_param('Sn')
        self.remove_param('Vn')
        self._params.extend(['Vdcn',
                             'Idcn',
                             'voltage',
                             ])
        self._descr.update({'Vdcn': 'DC voltage rating',
                            'Idcn': 'DC current rating',
                            'voltage': 'Initial nodal voltage guess',
                            'area': 'Area code',
                            'region': 'Region code',
                            'xcoord': 'x coordinate',
                            'ycoord': 'y coordinate'
                           })
        self._data.update({'Vdcn': 100.0,
                           'Idcn': 10.0,
                           'area': 0,
                           'region': 0,
                           'voltage': 1.0,
                           'xcoord': None,
                           'ycoord': None,
                           })
        self._units.update({'Vdcn': 'kV',
                            'Idcn': 'kA',
                            'area': 'na',
                            'region': 'na',
                            'voltage': 'pu',
                            'xcoord': 'deg',
                            'ycoord': 'deg',
                            })
        self.calls.update({'init0': True, 'pflow': True,
                           'jac0': True,
                           })
        self._mandatory = ['Vdcn']
        self._zeros = ['Vdcn', 'Idcn']
        self.v = []
        self._inst_meta()

    def add(self, idx=None, name=None, **kwargs):
        super().add(idx, name, **kwargs)

    def _varname(self):
        self.system.VarName.append(listname='unamey', xy_idx=self.v, var_name='Vdc', element_name=self.name)
        self.system.VarName.append(listname='fnamey', xy_idx=self.v, var_name='V_{dc}', element_name=self.name)

    def setup(self):
        self.v = list(range(self.system.DAE.m, self.system.DAE.m + self.n))
        self.system.DAE.m += self.n
        self._param2matrix()

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
        self._data.update({'node1': None,
                           'node2': None,
                           'Vdcn': 100.0,
                           'Idcn': 10.0,
                           })
        self._params.extend(['Vdcn', 'Idcn'])
        self._descr.update({'Vdcn': 'DC voltage rating',
                            'Idcn': 'DC current rating',
                            'node1': 'DC node 1 idx',
                            'node2': 'DC node 2 idx',
                            })
        self._units.update({'Vdcn': 'kV',
                           'Idcn': 'A',
                           })
        self._dc = {'node1': 'v1',
                    'node2': 'v2',
                    }
        self._mandatory.extend(['node1', 'node2', 'Vdcn'])

    @property
    def v12(self):
        return self.system.DAE.y[self.v1] - self.system.DAE.y[self.v2]


class RLine(DCBase):
    """DC Resistence line class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'RLine'
        self._params.extend(['R'])
        self._data.update({'R': 0.1,
                           })
        self._r.extend('R')
        self.calls.update({'pflow': True,
                           'gcall': True,
                           'jac0': True,
                           })
        self._algebs.extend(['I'])
        self._unamey = ['I']
        self._fnamey = ['I']
        self._inst_meta()
        self.Y = []

    def gcall(self, dae):
        dae.g[self.I] = div(dae.y[self.v1] - dae.y[self.v2], self.R) + dae.y[self.I]
        dae.g -= spmatrix(dae.y[self.I], self.v1, [0] * self.n, (dae.m, 1), 'd')
        dae.g += spmatrix(dae.y[self.I], self.v2, [0] * self.n, (dae.m, 1), 'd')

    def jac0(self, dae):
        dae.add_jac(Gy0, -self.u, self.v1, self.I)
        dae.add_jac(Gy0, self.u, self.v2, self.I)
        dae.add_jac(Gy0, div(self.u, self.R), self.I, self.v1)
        dae.add_jac(Gy0, -div(self.u, self.R), self.I, self.v2)
        dae.add_jac(Gy0, self.u - 1e-6, self.I, self.I)


class LLine(DCBase):
    """Pure inductive line"""
    def __init__(self, system, name):
        super(LLine, self).__init__(system, name)
        self._name = 'LLine'
        self._data.update({'L': 0.1})
        self._params.extend(['L'])
        self._r.extend(['L'])
        self._algebs.extend(['Idc'])
        self._fnamey.extend(['I_{dc}'])
        self._states.extend(['IL'])
        self._fnamex.extend(['I_L'])
        self._service.extend(['iL'])
        self.calls.update({'pflow': True, 'init0': True,
                           'gcall': True, 'fcall': True,
                           'jac0': True, 'fxcall': True,
                           })
        self._inst_meta()

    def servcall(self, dae):
        self.iL = div(self.u, self.L)

    def init0(self, dae):
        self.servcall(dae)
        # dae.


class RLLine(DCBase):
    """DC Resistive and Inductive line"""
    def __init__(self, system, name):
        super(RLLine, self).__init__(system, name)
        self._name = 'RLLine'
        self._params.extend(['R', 'L'])
        self._data.update({'R': 0.1,
                           'L': 0.1,
                           })
        self._params.extend(['R', 'L'])
        self._r.extend(['R', 'L'])
        self._algebs.extend(['Idc'])
        self._fnamey.extend(['I_{dc}'])
        self._states.extend(['IL'])
        self._fnamex.extend(['I_L'])
        self.calls.update({'pflow': True, 'init0': True,
                           'gcall': True, 'fcall': True,
                           'jac0': True, 'fxcall': True,
                           })
        self._service.extend(['iR', 'iL'])
        self._inst_meta()

    def servcall(self, dae):
        self.iR = div(1, self.R)
        self.iL = div(1, self.L)

    def init0(self, dae):
        self.servcall(dae)
        dae.x[self.IL] = mul(self.v12, self.iR)
        dae.y[self.Idc] = - dae.x[self.IL]

    def gcall(self, dae):
        dae.g[self.Idc] = dae.x[self.IL] + dae.y[self.Idc]

    def fcall(self, dae):
        dae.f[self.IL] = mul(self.v12 - mul(self.R, dae.x[self.IL]), self.iL)

    def jac0(self, dae):
        dae.add_jac(Gx0, 1, self.Idc, self.IL)
        dae.add_jac(Gy0, 1, self.Idc, self.Idc)
        dae.add_jac(Fx0, -mul(self.R, self.iL), self.IL, self.IL)

    def fxcall(self, dae):
        dae.add_jac(Fy, 1, self.IL, self.v1)
        dae.add_jac(Fy, -1, self.IL, self.v2)


class Ground(DCBase):
    """DC Ground node"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self.remove_param('node1')
        self.remove_param('node2')

        self._data.update({'node': None,
                           'voltage': 0.0,
                           })
        self._algebs.extend(['I'])
        self._unamey = ['I']
        self._fnamey = ['I']

        self._dc = {'node': 'v'}
        self._params.extend(['voltage'])
        self._mandatory.extend(['node'])
        self.calls.update({'init0': True,
                           'jac0': True,
                           'gcall': True,
                           'pflow': True,
                           })

        self._inst_meta()

    def init0(self, dae):
        dae.y[self.v] = self.voltage

    def gcall(self, dae):
        dae.g[self.v] -= dae.y[self.I]
        dae.g[self.I] = self.voltage - dae.y[self.v]

    def jac0(self, dae):
        dae.add_jac(Gy0, -self.u, self.v, self.I)
        dae.add_jac(Gy0, self.u - 1 - 1e-6, self.I, self.I)
        dae.add_jac(Gy0, -self.u, self.I, self.v)
