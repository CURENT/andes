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
        self.busu0 = None           # placeholder for Bus.u.v
        self.is_act = False         # flag for act, True to act

    def init(self):
        """
        Initialize the connectivity.
        """
        self.busu0 = self.system.Bus.u.v.copy()
        return None

    def record(self):
        """
        Record the bus connectivity in-place.
        """
        self.busu0[...] = self.system.Bus.u
        return None

    def act(self):
        """
        Update the connectivity.
        """
        if not self.is_act:
            logger.debug('Connectivity is not need to be updated.')
            return None

        # --- action ---
        pass

        self.system.connectivity(info=True)
        return None
