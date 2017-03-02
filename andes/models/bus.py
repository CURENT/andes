from .base import ModelBase
from cvxopt import matrix, spmatrix
from cvxopt import uniform


class Bus(ModelBase):
    """AC bus model"""
    def __init__(self, system, name):
        """constructor of an AC bus object"""
        super().__init__(system, name)
        self._group = 'Topology'
        self._data.pop('Sn')
        self._data.update({'Vn': 110.0,
                           'voltage': 1.0,
                           'angle': 0.0,
                           'vmax': 1.1,
                           'vmin': 0.9,
                           'area': 0,
                           'region': 0,
                           'owner': 0,
                           'xcoord': None,
                           'ycoord': None,
                           })
        self._params = ['u',
                        'Vn',
                        'voltage',
                        'angle',
                        'vmax',
                        'vmin',]
        self._descr.update({'voltage': 'voltage magnitude in p.u.',
                            'angle': 'voltage angle in radian',
                            'vmax': 'maximum voltage in p.u.',
                            'vmin': 'minimum voltage in p.u.',
                            'area': 'area code',
                            'region': 'region code',
                            'owner': 'owner code',
                            'xcoord': 'x coordinate',
                            'ycoord': 'y coordinate',
                            })
        self._service = ['Pg',
                         'Qg',
                         'Pl',
                         'Ql']
        self._zeros = ['Vn']
        self._mandatory = ['Vn']
        self.calls.update({'init0': True,
                           'pflow': True,
                           })
        self._inst_meta()
        self.a = list()
        self.v = list()

    def setup(self):
        """set up bus class after data parsing - manually assign angle and voltage indices"""
        if not self.n:
            self.system.Log.error('Powersystem instance contains no <Bus> element.')
        self.a = list(range(0, self.n))
        self.v = list(range(self.n, 2*self.n))
        self.system.DAE.m = 2*self.n
        self._list2matrix()

    def _varname(self):
        """customize varname for bus class"""
        if not self.addr:
            self.system.Log.error('Unable to assign Varname before allocating address')
            return
        self.system.VarName.append(listname='unamey', xy_idx=self.a, var_name='theta', element_name=self.a)
        self.system.VarName.append(listname='unamey', xy_idx=self.v, var_name='vm', element_name=self.a)
        self.system.VarName.append(listname='fnamey', xy_idx=self.a, var_name='\\theta', element_name=self.a)
        self.system.VarName.append(listname='fnamey', xy_idx=self.v, var_name='V', element_name=self.a)


