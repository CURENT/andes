import logging
from collections import OrderedDict

import numpy as np

from andes.core.model import Model, ModelData
from andes.core.param import DataParam, IdxParam, NumParam
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

        # island information
        self.n_islanded_buses = 0
        self.island_sets = list()
        self.islanded_buses = list()  # list of lists containing bus uid of islands
        self.islands = list()         # same as the above
        self.islanded_a = np.array([])
        self.islanded_v = np.array([])

        # config
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
                       is_output=True,
                       )
        self.v = Algeb(name='v',
                       tex_name='V',
                       info='voltage magnitude',
                       unit='p.u.',
                       is_output=True,
                       )

        # initial values
        self.a.v_str = 'flat_start*1e-8 + ' \
                       '(1-flat_start)*a0'
        self.v.v_str = 'flat_start*1 + ' \
                       '(1-flat_start)*v0'

    def set(self, src, idx, attr, value):
        super().set(src=src, idx=idx, attr=attr, value=value)
        _check_conn_status(system=self.system, src=src, attr=attr)


def _check_conn_status(system, src, attr):
    """
    Helper function to determine if connectivity update is needed.

    Parameters
    ----------
    system : System
        The system object.
    src : str
        Name of the model property
    attr : str
        The internal attribute of the property to get.
    """
    # Check if connectivity update is required
    if src == 'u' and attr == 'v':
        if system.is_setup:
            system.conn.record()  # Record connectivity once setup is confirmed

        if not system.TDS.initialized:
            # Log a warning if Power Flow needs resolution before EIG or TDS
            if system.PFlow.converged:
                logger.warning('Bus connectivity is touched, resolve PFlow before running EIG or TDS!')
            system.PFlow.converged = False  # Flag Power Flow as not converged
