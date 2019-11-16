import logging
from andes.core.model import Model, ModelData, ModelConfig  # NOQA
from andes.core.param import DataParam, NumParam, ExtParam  # NOQA
from andes.core.var import Algeb, State, ExtAlgeb  # NOQA
from andes.core.limiter import Comparer, SortedLimiter  # NOQA
from andes.core.service import Service  # NOQa
logger = logging.getLogger(__name__)


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


class Bus(Model, BusData):
    """
    Bus model constructed from the NewModelBase
    """
    group = 'StaticLoad'
    category = 'Load'

    def __init__(self, system, name=None, config=None):
        Model.__init__(self, system=system, name=name, config=config)
        BusData.__init__(self)

        self.flags['collate'] = False
        self.flags['pflow'] = True

        self.a = Algeb(name='a', tex_name=r'\theta', info='voltage angle', unit='radian')
        self.v = Algeb(name='v', tex_name='V', info='voltage magnitude', unit='pu')

        # optional initial values
        self.a.v_init = 'angle'
        self.v.v_init = 'voltage'
