"""
Module for Connectivity Manager.
"""

import logging
from collections import OrderedDict

from andes.utils.func import list_flatten
from andes.shared import np

logger = logging.getLogger(__name__)


# connectivity dependencies of `Bus`
# NOTE: only include PFlow models and measurements models
# cause online status of dynamic models are expected to be handled by their
# corresponding static models
# TODO: DC Topologies are not included yet, `Node`, etc
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
    attached devices when a ``Bus`` is turned off **after** system
    setup and **before** TDS initializtion.

    Attributes
    ----------
    system: system object
        System object to manage the connectivity.
    busu0: ndarray
        Last recorded bus connection status.
    is_needed: bool
        Flag to indicate if connectivity update is needed.
    changes: dict
        Dictionary to record bus connectivity changes ('on' and 'off').
        'on' means the bus is previous offline and now online.
        'off' means the bus is previous online and now offline.
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
        self.busu0 = None               # placeholder for Bus.u.v
        self.is_needed = False          # flag to indicate if check is needed
        self.changes = {'on': None, 'off': None}    # dict of bus connectivity changes

    def init(self):
        """
        Initialize the connectivity.

        `ConnMan` is initialized in `System.setup()`, where all buses are considered online
        by default. This method records the initial bus connectivity.
        """
        # NOTE: here, we expect all buses are online before the system setup
        self.busu0 = np.ones(self.system.Bus.n, dtype=int)
        self.changes['on'] = np.zeros(self.system.Bus.n, dtype=int)
        self.changes['off'] = np.logical_and(self.busu0 == 1, self.system.Bus.u.v == 0).astype(int)

        if np.any(self.changes['off']):
            self.is_needed = True

        self.act()

        return True

    def _update(self):
        """
        Helper function for in-place update of bus connectivity.
        """
        self.changes['on'][...] = np.logical_and(self.busu0 == 0, self.system.Bus.u.v == 1)
        self.changes['off'][...] = np.logical_and(self.busu0 == 1, self.system.Bus.u.v == 0)
        self.busu0[...] = self.system.Bus.u.v

    def record(self):
        """
        Record the bus connectivity in-place.

        This method should be called if `Bus.set()` or `Bus.alter()` is called.
        """
        self._update()

        if np.any(self.changes['on']):
            onbus_idx = [self.system.Bus.idx.v[i] for i in np.nonzero(self.changes["on"])[0]]
            logger.warning(f'Bus turned on: {onbus_idx}')
            self.is_needed = True
            if len(onbus_idx) > 0:
                raise NotImplementedError('Turning on bus after system setup is not supported yet!')

        if np.any(self.changes['off']):
            offbus_idx = [self.system.Bus.idx.v[i] for i in np.nonzero(self.changes["off"])[0]]
            logger.warning(f'Bus turned off: {offbus_idx}')
            self.is_needed = True

        return self.changes

    def act(self):
        """
        Update the connectivity.
        """
        if not self.is_needed:
            logger.debug('No need to update connectivity.')
            return True

        if self.system.TDS.initialized:
            raise NotImplementedError('Bus connectivity update during TDS is not supported yet!')

        # --- action ---
        offbus_idx = [self.system.Bus.idx.v[i] for i in np.nonzero(self.changes["off"])[0]]

        # skip if no bus is turned off
        if len(offbus_idx) == 0:
            return True

        logger.warning('Entering connectivity update.')
        logger.warning(f'Following bus(es) are turned off: {offbus_idx}')

        logger.warning('-> System connectivity update results:')
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

        self.is_needed = False      # reset the action flag
        self._update()              # update but not record
        self.system.connectivity(info=True)
        return True
