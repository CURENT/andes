from cvxopt import matrix, mul, spmatrix
from .base import ModelBase
from ..consts import *

class Shunt(ModelBase):
    """Static shunt device"""
    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'StaticVar'
        self._name = 'Shunt'
        self._data.update({'bus': None,
                           'g': 0,
                           'b': 0,
                           'fn': 60.0,
                           })
        self._units.update({'bus': 'na',
                            'g': 'pu',
                            'b': 'pu',
                            'fn': 'Hz',
                            })
        self._params.extend(['g',
                             'b',
                             'fn',
                             ])
        self._descr.update({'bus': 'idx of connected bus',
                            'g': 'shunt conductance (real part)',
                            'b': 'shunt susceptance (positive as capatance)',
                            'fn': 'rated frequency',
                            })
        self._ac = {'bus': ['a', 'v']}
        self.calls.update({'gcall': True, 'gycall': True,
                           'pflow': True, 'shunt': True,
                           })

        self._y = ['g', 'b']
        self._inst_meta()

    def full_y(self, Y):
        """Add self(shunt) into full Jacobian Y"""
        if not self.n:
            return
        Ysh = matrix(self.g, (self.n, 1), 'd') + 1j*matrix(self.b, (self.n, 1),'d')
        uYsh = mul(self.u, Ysh)
        Y += spmatrix(uYsh, self.a, self.a, Y.size, 'z')

    def gcall(self, dae):
        vc2 = mul(self.u, dae.y[self.v] ** 2)
        dae.g += spmatrix(mul(vc2, self.g), self.a, [0] * self.n, (dae.m, 1), 'd')
        dae.g -= spmatrix(mul(vc2, self.b), self.v, [0] * self.n, (dae.m, 1), 'd')

    def gycall(self, dae):
        dV2 = mul(self.u, 2 * dae.y[self.v])
        dPdV = mul(dV2, matrix(self.g, (self.n, 1), 'd'))
        dQdV = -mul(dV2, matrix(self.b, (self.n, 1), 'd'))

        dae.add_jac(Gy, dPdV, self.a, self.v)
        dae.add_jac(Gy, dQdV, self.v, self.v)

