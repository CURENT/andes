from .base import ModelBase
from cvxopt import matrix, spmatrix, uniform
from ..consts import *


class Bus(ModelBase):
    """AC bus model"""
    def __init__(self, system, name):
        """constructor of an AC bus object"""
        super().__init__(system, name)
        self._group = 'Topology'
        self._data.pop('Sn')
        self._data.update({'voltage': 1.0,
                           'angle': 0.0,
                           'vmax': 1.1,
                           'vmin': 0.9,
                           'area': 0,
                           'zone': 0,
                           'region': 0,
                           'owner': 0,
                           'xcoord': None,
                           'ycoord': None,
                           })
        self._units.update({'voltage': 'pu',
                            'angle': 'rad',
                            'vmax': 'pu',
                            'vmin': 'pu',
                            'xcoord': 'deg',
                            'ycoord': 'deg',
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
                            'zone': 'zone code',
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
        self.islanded_buses = list()
        self.island_sets = list()

    def setup(self):
        """Set up bus class after data parsing - manually assign angle and voltage indices"""
        self.a = list(range(0, self.n))
        self.v = list(range(self.n, 2*self.n))
        self.system.DAE.m = 2*self.n
        self._param2matrix()

    def _varname(self):
        """Customize varname for bus class"""
        self.system.VarName.append(listname='unamey', xy_idx=self.a, var_name='theta', element_name=self.name)
        self.system.VarName.append(listname='unamey', xy_idx=self.v, var_name='vm', element_name=self.name)
        self.system.VarName.append(listname='fnamey', xy_idx=self.a, var_name='\\theta', element_name=self.name)
        self.system.VarName.append(listname='fnamey', xy_idx=self.v, var_name='V', element_name=self.name)

    def _varname_inj(self):
        """Customize varname for bus injections"""
        # Bus Pi
        if not self.n:
            return
        m = self.system.DAE.m
        xy_idx = range(m, self.n + m)
        self.system.VarName.append(listname='unamey', xy_idx=xy_idx, var_name='P', element_name=self.name)
        self.system.VarName.append(listname='fnamey', xy_idx=xy_idx, var_name='P', element_name=self.name)

        # Bus Qi
        xy_idx = range(m + self.n, m + 2*self.n)
        self.system.VarName.append(listname='unamey', xy_idx=xy_idx, var_name='Q', element_name=self.name)
        self.system.VarName.append(listname='fnamey', xy_idx=xy_idx, var_name='Q', element_name=self.name)

    def init0(self, dae):
        """Set bus Va and Vm initial values"""
        if not self.system.SPF.flatstart:
            dae.y[self.a] = self.angle + 1e-10*uniform(self.n)
            dae.y[self.v] = self.voltage
        else:
            dae.y[self.a] = matrix(0.0, (self.n, 1), 'd') + 1e-10*uniform(self.n)
            dae.y[self.v] = matrix(1.0, (self.n, 1), 'd')

    def gisland(self,dae):
        """Reset g(x) for islanded buses and areas"""
        if not (self.islanded_buses and self.island_sets):
            return

        a, v = list(), list()

        # for islanded areas without a slack bus
        for island in self.island_sets:
            nosw = 1
            for item in self.system.SW.bus:
                if self.int[item] in island:
                    nosw = 0
                    break
            if nosw:
                self.islanded_buses += island
                self.island_sets.remove(island)

        a = self.islanded_buses
        v = [self.n + item for item in a]
        dae.g[a] = 0
        dae.g[v] = 0

    def gyisland(self,dae):
        """Reset gy(x) for islanded buses and areas"""
        if self.system.Bus.islanded_buses:
            a = self.system.Bus.islanded_buses
            v = [self.system.Bus.n + item for item in a]
            dae.set_jac(Gy, 1e-6, a, a)
            dae.set_jac(Gy, 1e-6, v, v)


