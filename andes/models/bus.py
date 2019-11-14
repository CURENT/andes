from cvxopt import matrix, uniform  # NOQA

from andes.core.var import Algeb
from andes.core.param import NumParam, DataParam  # NOQA
from .base import ModelBase
from .base import ModelData, Model

from ..consts import Gy  # NOQA


class Bus(ModelBase):
    """
    AC bus model for defining topology definition
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
        if (not self.islanded_buses) and (not self.island_sets):
            return

        a, v = list(), list()

        # for islanded areas without a slack bus
        # TODO: fix for islanded sets without sw
        # for island in self.island_sets:
        #     nosw = 1
        #     for item in self.system.SW.bus:
        #         if self.uid[item] in island:
        #             nosw = 0
        #             break
        #     if nosw:
        #         self.islanded_buses += island
        #         self.island_sets.remove(island)

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


class BusData(ModelData):
    """
    Class for Bus data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Vn = NumParam(default=110, info="AC voltage rating", unit='kV', non_zero=True)
        self.angle = NumParam(default=0, info="initial voltage phase angle", unit='rad')

        self.area = DataParam(default=None, info="Area code")
        self.region = DataParam(default=None, info="Region code")
        self.owner = DataParam(default=None, info="Owner code")
        self.xcoord = DataParam(default=0, info='x coordinate')
        self.ycoord = DataParam(default=0, info='y coordinate')

        self.vmax = NumParam(default=1.1, info="Voltage upper limit")
        self.vmin = NumParam(default=0.9, info="Voltage lower limit")
        self.voltage = NumParam(default=1.0, info="initial voltage magnitude", non_zero=True)


class BusNew(Model, BusData):
    """
    Bus model constructed from the NewModelBase
    """
    group = 'StaticLoad'
    category = 'Load'

    def __init__(self, system, name=None):
        Model.__init__(self, system=system, name=name)
        BusData.__init__(self)

        self.flags['collate'] = False
        self.flags['pflow'] = True

        self.a = Algeb(name='a', tex_name=r'\theta', info='voltage angle', unit='radian')
        self.v = Algeb(name='v', tex_name='V', info='voltage magnitude', unit='pu')

        # optional initial values
        self.a.v_init = 'angle'
        self.v.v_init = 'voltage'
