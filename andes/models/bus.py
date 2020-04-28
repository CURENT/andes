import logging
from collections import OrderedDict
from andes.core.model import Model, ModelData
from andes.core.param import IdxParam, DataParam, NumParam
from andes.core.var import Algeb
logger = logging.getLogger(__name__)


class BusData(ModelData):
    """
    Class for Bus data
    """
    def __init__(self):
        super().__init__()
        self.Vn = NumParam(default=110,
                           info="AC voltage rating",
                           unit='kV',
                           non_zero=True,
                           tex_name=r'V_n',
                           )
        self.vmax = NumParam(default=1.1,
                             info="Voltage upper limit",
                             tex_name=r'V_{max}',
                             unit='p.u.',
                             )
        self.vmin = NumParam(default=0.9,
                             info="Voltage lower limit",
                             tex_name=r'V_{min}',
                             unit='p.u.',
                             )

        self.v0 = NumParam(default=1.0,
                           info="initial voltage magnitude",
                           non_zero=True,
                           tex_name=r'V_0',
                           unit='p.u.',
                           )
        self.a0 = NumParam(default=0,
                           info="initial voltage phase angle",
                           unit='rad',
                           tex_name=r'\theta_0',
                           )

        self.xcoord = DataParam(default=0,
                                info='x coordinate (longitude)',
                                )
        self.ycoord = DataParam(default=0,
                                info='y coordinate (latitude)',
                                )

        self.area = IdxParam(model='Area',
                             default=None,
                             info="Area code",
                             )
        self.zone = IdxParam(model='Region',
                             default=None,
                             info="Zone code",
                             )
        self.owner = IdxParam(model='Owner',
                              default=None,
                              info="Owner code",
                              )


class Bus(Model, BusData):
    """
    AC Bus model.

    Power balance equation have the form of ``load - injection = 0``.
    Namely, load is positively summed, while injections are negative.
    """

    def __init__(self, system=None, config=None):
        BusData.__init__(self)
        Model.__init__(self, system=system, config=config)

        self.config.add(OrderedDict((('flat_start', 0),
                                     )))
        self.config.add_extra("_help",
                              flat_start="flat start for voltages",
                              )
        self.config.add_extra("_alt",
                              flat_start=(0, 1),
                              )
        self.config.add_extra("_tex",
                              flat_start="z_{flat}",
                              )

        self.group = 'ACTopology'
        self.category = ['TransNode']

        self.flags.update({'collate': False,
                           'pflow': True})

        self.a = Algeb(name='a',
                       tex_name=r'\theta',
                       info='voltage angle',
                       unit='rad',
                       )
        self.v = Algeb(name='v',
                       tex_name='V',
                       info='voltage magnitude',
                       unit='p.u.',
                       )

        # initial values
        self.a.v_str = 'flat_start*1e-8 + ' \
                       '(1-flat_start)*a0'
        self.v.v_str = 'flat_start*1 + ' \
                       '(1-flat_start)*v0'
