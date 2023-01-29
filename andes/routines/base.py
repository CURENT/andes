"""
Base class for ANDES calculation routines.
"""

from andes.linsolvers.solverbase import Solver
from andes.core import Config
from collections import OrderedDict


def create_config_base_routine(name, config_obj=None):
    config = Config(name)

    if config_obj is not None:
        config.load(config_obj)

    config.add(OrderedDict((('sparselib', 'klu'),
                            ('linsolve', 0),
                            )))

    config.add_extra("_help",
                     sparselib="linear sparse solver name",
                     linsolve="solve symbolic factorization each step (enable when KLU segfaults)",
                     )
    config.add_extra("_alt",
                     sparselib=("klu", "umfpack", "spsolve", "cupy"),
                     linsolve=(0, 1),
                     )

    return config


class BaseRoutine:
    """
    Base routine class.

    Provides references to system, config, and solver.
    """

    def __init__(self, system=None):
        self.system = system

        self.config = create_config_base_routine("BaseRoutine")
        self.solver = Solver(sparselib=self.config.sparselib)
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
