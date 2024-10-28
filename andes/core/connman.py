"""
Module for Connectivity Manager.
"""

import logging
from collections import OrderedDict

from andes.utils.func import list_flatten
from andes.shared import np

logger = logging.getLogger(__name__)


# connectivity dependencies for each model
# TODO: this is not an exhaustive list
# TODO: DC Topologies are not included yet
bus_deps = OrderedDict([
    ('ACLine', ['bus1', 'bus2']),
    ('ACShort', ['bus1', 'bus2']),
    ('FreqMeasurement', ['bus']),
    ('Interface', ['bus']),
    ('Motor', ['bus']),
    ('PhasorMeasurement', ['bus']),
    ('StaticACDC', ['bus']),
    ('StaticGen', ['bus']),
    ('StaticLoad', ['bus']),
    ('StaticShunt', ['bus']),
])


class ConnMan:
    """
    Define a Connectivity Manager class for System.

    Connectivity Manager is used to automatically **turn off**
    attached devices when a `Bus` is turned off.

    Attributes
    ----------
    system: system object
        System object to manage the connectivity.
    busu0: ndarray
        Last recorded bus connection status.
    is_changed: bool
        Flag to indicate if bus connectivity is changed.
    changes: dict
        Dictionary to record bus connectivity changes ('on' and 'off').
    """

    def __init__(self, system=None):
        """
        Initialize the connectivity manager.

        Parameters
        ----------
        system: system object
            System object to manage the connectivity.
        """
        self.system = system
        self.busu0 = None           # placeholder for Bus.u.v
        self.is_changed = False     # flag to indicate if bus connectivity is changed
        self.changes = {'on': None, 'off': None}    # dict of bus connectivity changes

    def init(self):
        """
        Initialize the connectivity.

        `ConnMan` is initialized in `System.setup()`, where all buses are considered online
        by default. This method records the initial bus connectivity.
        """
        self.busu0 = np.ones(self.system.Bus.n, dtype=bool)
        # NOTE: 'on' means th or 'off'e bus is previous offline and now online
        #       'off' means the bus is previous online and now offline
        #       The bool value for each bus indicates if the bus is 'on'
        self.changes['on'] = np.zeros(self.system.Bus.n, dtype=bool)
        self.changes['off'] = np.logical_and(self.busu0 == 1, self.system.Bus.u.v == 0)

        if np.any(self.changes['off']):
            self.is_changed = True
        return self.changes

    def record(self):
        """
        Record the bus connectivity in-place.

        This method should be called if `Bus.set()` or `Bus.alter()` is called.
        """
        self.changes['on'][...] = np.logical_and(self.busu0 == 0, self.system.Bus.u.v == 1)
        self.changes['off'][...] = np.logical_and(self.busu0 == 1, self.system.Bus.u.v == 0)

        if np.any(self.changes['on']):
            onbus_idx = [self.system.Bus.idx.v[i] for i in np.nonzero(self.changes["on"])[0]]
            logger.warning(f'Bus turned on: {onbus_idx}')

        if np.any(self.changes['off']):
            offbus_idx = [self.system.Bus.idx.v[i] for i in np.nonzero(self.changes["off"])[0]]
            logger.warning(f'Bus turned off: {offbus_idx}')

        # update busu0
        self.busu0[...] = self.system.Bus.u.v
        return self.changes

    def act(self):
        """
        Update the connectivity.
        """
        if not self.is_changed:
            logger.debug('Connectivity is not need to be updated.')
            return None

        # --- action ---
        logger.warning('Entering connectivity update.')
        logger.warning('-> System connectivity update results:')

        offbus_idx = [self.system.Bus.idx.v[i] for i in np.nonzero(self.changes["off"])[0]]
        for grp_name, src_list in bus_deps.items():
            devices = []
            for src in src_list:
                grp_devs = self.system.__dict__[grp_name].find_idx(keys=src, values=offbus_idx,
                                                                   allow_none=True, allow_all=True,
                                                                   default=None)
                grp_devs_flat = list_flatten(grp_devs)
                if grp_devs_flat != [None]:
                    devices.append(grp_devs_flat)

            devices_flat = list_flatten(devices)

            if len(devices_flat) > 0:
                self.system.__dict__[grp_name].set(src='u', attr='v',
                                                   idx=devices_flat, value=0)
                logger.warning(f'In <{grp_name}>, turn off {devices_flat}')

        self.is_changed = False     # reset the action flag

        self.system.connectivity(info=True)
        return None
