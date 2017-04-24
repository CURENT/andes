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
        self.system.VarName.append(listname='unamey', xy_idx=self.v, var_name='Vdc', element_name=self.names)
        self.system.VarName.append(listname='fnamey', xy_idx=self.v, var_name='V_{dc}', element_name=self.names)

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
        self._dc = {'node1': 'v1',
                    'node2': 'v2',
                    }
        self._mandatory.extend(['node1', 'node2', 'Vdcn'])


class RLine(DCBase):
    """DC Resistence line class"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._name = 'RLine'
        self._params.extend(['R'])
        self._data.update({'R': 1.0,
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
        if self.n > 1:
            self.message('Multiple DC Ground defined!', ERROR)
        dae.y[self.v] = self.voltage

    def gcall(self, dae):
        dae.g[self.v] -= dae.y[self.I]
        dae.g[self.I] = self.voltage - dae.y[self.v]

    def jac0(self, dae):
        dae.add_jac(Gy0, -self.u, self.v, self.I)
        dae.add_jac(Gy0, self.u - 1 - 1e-6, self.I, self.I)
        dae.add_jac(Gy0, -self.u, self.I, self.v)
