import logging
from andes.core.model import Model, ModelData
from andes.core.param import IdxParam, DataParam, NumParam  # NOQA
from andes.core.var import Algeb
logger = logging.getLogger(__name__)


class BusData(ModelData):
    """
    Class for Bus data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Vn = NumParam(default=110, info="AC voltage rating", unit='kV', non_zero=True, tex_name=r'V_n')
        self.vmax = NumParam(default=1.1, info="Voltage upper limit", tex_name=r'V_{max}', unit='p.u.')
        self.vmin = NumParam(default=0.9, info="Voltage lower limit", tex_name=r'V_{min}', unit='p.u.')

        self.v0 = NumParam(default=1.0, info="initial voltage magnitude", non_zero=True, tex_name=r'V_0',
                           unit='p.u.')
        self.a0 = NumParam(default=0, info="initial voltage phase angle", unit='rad', tex_name=r'\theta_0')

        self.xcoord = DataParam(default=0, info='x coordinate (longitude)')
        self.ycoord = DataParam(default=0, info='y coordinate (latitude)')

        self.area = IdxParam(model='Area', default=None, info="Area code")
        self.region = IdxParam(model='Region', default=None, info="Region code")
        self.owner = IdxParam(model='Owner', default=None, info="Owner code")


class Bus(Model, BusData):
    """
    AC Bus model developed using the symbolic framework
    """

    def __init__(self, system=None, config=None):
        BusData.__init__(self)
        Model.__init__(self, system=system, config=config)

        self.group = 'ACTopology'
        self.category = ['TransNode']

        self.flags.update({'collate': False,
                           'pflow': True})

        self.a = Algeb(name='a', tex_name=r'\theta', info='voltage angle', unit='rad')
        self.v = Algeb(name='v', tex_name='V', info='voltage magnitude', unit='p.u.')

        # optional initial values
        self.a.v_str = 'a0'
        self.v.v_str = 'v0'
