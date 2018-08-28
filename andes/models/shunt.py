from cvxopt import matrix, mul, spmatrix
from scipy.sparse import coo_matrix
import numpy as np

from .base import ModelBase
from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA


class Shunt(ModelBase):
    """Static shunt device"""

    def __init__(self, system, name):
        super().__init__(system, name)
        self._group = 'StaticVar'
        self._name = 'Shunt'
        self._data.update({
            'bus': None,
            'g': 0,
            'b': 0,
            'fn': 60.0,
        })
        self._units.update({
            'bus': 'na',
            'g': 'pu',
            'b': 'pu',
            'fn': 'Hz',
        })
        self._params.extend([
            'g',
            'b',
            'fn',
        ])
        self._descr.update({
            'bus': 'idx of connected bus',
            'g': 'shunt conductance (real part)',
            'b': 'shunt susceptance (positive as capatance)',
            'fn': 'rated frequency',
        })
        self._ac = {'bus': ['a', 'v']}
        self.calls.update({
            'gcall': True,
            'gycall': True,
            'pflow': True,
            'shunt': True,
        })

        self._y = ['g', 'b']
        self._init()

    def full_y(self, Y):
        """Add self(shunt) into full Jacobian Y"""
        if not self.n:
            return
        Ysh = matrix(self.g,
                     (self.n, 1), 'd') + 1j * matrix(self.b, (self.n, 1), 'd')
        uYsh = mul(self.u, Ysh)
        Y += spmatrix(uYsh, self.a, self.a, Y.size, 'z')

    def gcall(self, dae):
        # scipy.sparse.coo_matrix speed up for large matrices ======

        v2 = mul(self.u, dae.y[self.v]**2)
        p_inj = mul(v2, self.g)
        q_inj = -mul(v2, self.b)

        if self.n <= 400:
            for a, v, p, q in zip(self.a, self.v, p_inj, q_inj):
                dae.g[a] += p
                dae.g[v] += q
        else:
            p = np.array(p_inj).reshape((-1))
            q = np.array(q_inj).reshape((-1))

            p = coo_matrix(
                (p, (self.a, np.zeros(self.n))), shape=(dae.m, 1)).toarray()
            q = coo_matrix(
                (q, (self.v, np.zeros(self.n))), shape=(dae.m, 1)).toarray()

            dae.g += matrix(p)
            dae.g += matrix(q)

    def gycall(self, dae):
        dV2 = mul(self.u, 2 * dae.y[self.v])
        dPdV = mul(dV2, matrix(self.g, (self.n, 1), 'd'))
        dQdV = -mul(dV2, matrix(self.b, (self.n, 1), 'd'))

        dae.add_jac(Gy, dPdV, self.a, self.v)
        dae.add_jac(Gy, dQdV, self.v, self.v)
