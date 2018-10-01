from .base import ModelBase
from cvxopt import matrix, spmatrix, uniform  # NOQA
from ..consts import *  # NOQA
from ..consts import Fx0, Fy0, Gx0, Gy0  # NOQA
from ..consts import Fx, Fy, Gx, Gy  # NOQA


class Bus(ModelBase):
    """ac bus model for defining topology definition
    """

    def define(self):
        self._name = 'Bus'
        self._group = 'Topology'
        self._category = 'Topology'

        self.param_remove('Sn')

        self.param_define(
            'voltage',
            1.0,
            unit='pu',
            descr='initial voltage magnitude',
            nonzero=True)
        self.param_define(
            'angle', 0.0, unit='rad', descr='initial voltage phase angle')
        self.param_define('vmax', 1.1, unit='pu', descr='voltage upper limit')
        self.param_define('vmin', 0.9, unit='pu', descr='voltage upper limit')
        self.param_define('area', 0, descr='area code', tomatrix=False)
        self.param_define('zone', 0, descr='zone code', tomatrix=False)
        self.param_define('region', 0, descr='region code', tomatrix=False)
        self.param_define('owner', 0, descr='owner code', tomatrix=False)
        self.param_define('xcoord', 0, descr='x coordinate', tomatrix=False)
        self.param_define('ycoord', 0, descr='y coordinate', tomatrix=False)

        self.param_alter('Vn', mandatory=True)

        self.var_define('a', 'y', '\\theta', descr='bus voltage phase angle')
        self.var_define('v', 'y', 'V', descr='bus voltage magnitude')

        self.service_define('Pg', matrix)
        self.service_define('Qg', matrix)
        self.service_define('Pl', matrix)
        self.service_define('Ql', matrix)
        self.service_define('islanded_buses', list)
        self.service_define('island_sets', list)

        self._config['address_group_by'] = 'variable'

        self.calls.update({
            'init0': True,
            'pflow': True,
        })

        self._init()

    def _varname_inj(self):
        """Customize varname for bus injections"""
        # Bus Pi
        if not self.n:
            return
        m = self.system.dae.m
        xy_idx = range(m, self.n + m)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='P',
            element_name=self.name)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='P',
            element_name=self.name)

        # Bus Qi
        xy_idx = range(m + self.n, m + 2 * self.n)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='Q',
            element_name=self.name)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='Q',
            element_name=self.name)

    def init0(self, dae):
        """Set bus Va and Vm initial values"""
        if not self.system.pflow.config.flatstart:
            dae.y[self.a] = self.angle + 1e-10 * uniform(self.n)
            dae.y[self.v] = self.voltage
        else:
            dae.y[self.a] = matrix(0.0,
                                   (self.n, 1), 'd') + 1e-10 * uniform(self.n)
            dae.y[self.v] = matrix(1.0, (self.n, 1), 'd')

    def gisland(self, dae):
        """Reset g(x) for islanded buses and areas"""
        if not (self.islanded_buses and self.island_sets):
            return

        a, v = list(), list()

        # for islanded areas without a slack bus
        for island in self.island_sets:
            nosw = 1
            for item in self.system.SW.bus:
                if self.uid[item] in island:
                    nosw = 0
                    break
            if nosw:
                self.islanded_buses += island
                self.island_sets.remove(island)

        a = self.islanded_buses
        v = [self.n + item for item in a]
        dae.g[a] = 0
        dae.g[v] = 0

    def gyisland(self, dae):
        """Reset gy(x) for islanded buses and areas"""
        if self.system.Bus.islanded_buses:
            a = self.system.Bus.islanded_buses
            v = [self.system.Bus.n + item for item in a]
            dae.set_jac(Gy, 1e-6, a, a)
            dae.set_jac(Gy, 1e-6, v, v)


class BusOld(ModelBase):
    """AC bus model"""

    def __init__(self, system, name):
        """constructor of an AC bus object"""
        super().__init__(system, name)
        self._group = 'Topology'
        self._data.pop('Sn')
        self._data.update({
            'voltage': 1.0,
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
        self._units.update({
            'voltage': 'pu',
            'angle': 'rad',
            'vmax': 'pu',
            'vmin': 'pu',
            'xcoord': 'deg',
            'ycoord': 'deg',
        })
        self._params = [
            'u',
            'Vn',
            'voltage',
            'angle',
            'vmax',
            'vmin',
        ]
        self._descr.update({
            'voltage': 'voltage magnitude in p.u.',
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
        self._service = ['Pg', 'Qg', 'Pl', 'Ql']
        self._zeros = ['Vn']
        self._mandatory = ['Vn']
        self._algebs.extend(['a', 'v'])
        self._fnamey.extend(['\\theta', 'V'])
        self.calls.update({
            'init0': True,
            'pflow': True,
        })
        self._init()
        self.islanded_buses = list()
        self.island_sets = list()
        self._config['address_group_by'] = 'variable'

    def _varname_inj(self):
        """Customize varname for bus injections"""
        # Bus Pi
        if not self.n:
            return
        m = self.system.dae.m
        xy_idx = range(m, self.n + m)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='P',
            element_name=self.name)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='P',
            element_name=self.name)

        # Bus Qi
        xy_idx = range(m + self.n, m + 2 * self.n)
        self.system.varname.append(
            listname='unamey',
            xy_idx=xy_idx,
            var_name='Q',
            element_name=self.name)
        self.system.varname.append(
            listname='fnamey',
            xy_idx=xy_idx,
            var_name='Q',
            element_name=self.name)

    def init0(self, dae):
        """Set bus Va and Vm initial values"""
        if not self.system.pflow.config.flatstart:
            dae.y[self.a] = self.angle + 1e-10 * uniform(self.n)
            dae.y[self.v] = self.voltage
        else:
            dae.y[self.a] = matrix(0.0,
                                   (self.n, 1), 'd') + 1e-10 * uniform(self.n)
            dae.y[self.v] = matrix(1.0, (self.n, 1), 'd')

    def gisland(self, dae):
        """Reset g(x) for islanded buses and areas"""
        if not (self.islanded_buses and self.island_sets):
            return

        a, v = list(), list()

        # for islanded areas without a slack bus
        for island in self.island_sets:
            nosw = 1
            for item in self.system.SW.bus:
                if self.uid[item] in island:
                    nosw = 0
                    break
            if nosw:
                self.islanded_buses += island
                self.island_sets.remove(island)

        a = self.islanded_buses
        v = [self.n + item for item in a]
        dae.g[a] = 0
        dae.g[v] = 0

    def gyisland(self, dae):
        """Reset gy(x) for islanded buses and areas"""
        if self.system.Bus.islanded_buses:
            a = self.system.Bus.islanded_buses
            v = [self.system.Bus.n + item for item in a]
            dae.set_jac(Gy, 1e-6, a, a)
            dae.set_jac(Gy, 1e-6, v, v)
