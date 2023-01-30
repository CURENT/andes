"""
Base class for ANDES calculation routines.
"""

from typing import Optional

# from andes.linsolvers.solverbase import Solver
from andes.core import Config


class BaseRoutine:
    """
    Base routine class.

    Provides references to system, config, and solver.
    """

    def __init__(self, system=None):
        self.system = system

        self.config: Optional[Config] = None

        # self.solver = Solver(sparselib=self.config.sparselib)

        self.exec_time = 0.0  # recorded time to execute the routine in seconds

    @property
    def class_name(self):
        return self.__class__.__name__

    def doc(self, max_width=78, export='plain'):
        """
        Routine documentation interface.
        """
        return self.config.doc(max_width, export)

    def init(self):
        """
        Routine initialization interface.
        """
        pass

    def run(self, **kwargs):
        """
        Routine main entry point.
        """
        raise NotImplementedError

    def summary(self, **kwargs):
        """
        Summary interface
        """
        raise NotImplementedError

    def report(self, **kwargs):
        """
        Report interface.
        """
        raise NotImplementedError

    def create_config(self, name, config_obj=None):

        config = Config(name)

        if config_obj is not None:
            config.load(config_obj)

        return config

    def register_config(self, config_manager):
        config_manager.register(self.class_name, self.create_config)

    def set_config(self, config_manager):
        self.config = config_manager._store[self.class_name]
