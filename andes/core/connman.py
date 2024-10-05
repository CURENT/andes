"""
Module for Connectivity Manager.
"""

import logging

logger = logging.getLogger(__name__)


class ConnMan:
    """
    Define a Connectivity Manager class for System
    """

    def __init__(self, system=None):
        """
        Initialize the connectivity manager.

        system: system object
        """
        self.system = system
        self.busu0 = system.Bus.u.v.copy()  # a copy of Bus.u.v for internal use

    def act(self):
        """
        Update the connectivity.
        """
        if not self.is_act:
            logger.debug('Connectivity is not need to be updated.')
            return None

        if True:
            self._disconnect()
        else:
            self._connect()
        self.system.connectivity(info=True)
        return None

    def _disconnect(self):
        """
        Disconnect involved devices.
        """
        pass
        
    def _connect(self):
        """
        Connect involved devices.
        """
        pass
