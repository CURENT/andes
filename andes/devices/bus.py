import logging
from andes.core.model import Model, ModelData  # NOQA
from andes.core.param import IdxParam, DataParam, NumParam  # NOQA
from andes.core.var import Algeb, Calc  # NOQA
logger = logging.getLogger(__name__)


class BusData(ModelData):
    """
    Class for Bus data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Vn = NumParam(default=110, info="AC voltage rating", unit='kV', non_zero=True, tex_name=r'V_n')
        self.vmax = NumParam(default=1.1, info="Voltage upper limit", tex_name=r'v_{max}')
        self.vmin = NumParam(default=0.9, info="Voltage lower limit", tex_name=r'v_{min}')

        self.v0 = NumParam(default=1.0, info="initial voltage magnitude", non_zero=True, tex_name=r'v_0')
        self.a0 = NumParam(default=0, info="initial voltage phase angle", unit='rad', tex_name=r'\theta_0')

        self.xcoord = DataParam(default=0, info='x coordinate')
        self.ycoord = DataParam(default=0, info='y coordinate')

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
        self.category = ['Node']

        self.flags.update({'collate': False,
                           'pflow': True})

        self.a = Algeb(name='a', tex_name=r'\theta', info='voltage angle', unit='radian')
        self.v = Algeb(name='v', tex_name='V', info='voltage magnitude', unit='pu')

        # --- TO BE REMOVED EXAMPLE ---
        self.a_times_v = Calc()
        self.a_times_v.e_str = 'a * v'
        # ------------------------------
        # optional initial values
        self.a.v_init = 'a0'
        self.v.v_init = 'v0'
