"""
Base class for ANDES calculation routines.
"""

from andes.linsolvers.solverbase import Solver
from andes.core import Config
from collections import OrderedDict


def check_conn_before_init(func):
    """
    A decorator that ensures the connection is active before calling the `init` or `run`
    method of a `BaseRoutine` derived class.

    This decorator calls the `act` method on `self.system.conn` to ensure the connection
    is active before proceeding with the initialization.

    Parameters
    ----------
    func : function
        The `init` method of a `BaseRoutine` derived class.

    Returns
    -------
    function
        The wrapped function with connection check.
    """

    def wrapper(self, *args, **kwargs):
        self.system.conn.act()
        return func(self, *args, **kwargs)
    return wrapper


class BaseRoutine:
    """
    Base routine class.

    Provides references to system, config, and solver.
    """

    def __init__(self, system=None, config=None):
        self.system = system
        self.config = Config(self.class_name)

        if config is not None:
            self.config.load(config)

        self.config.add(OrderedDict((('sparselib', 'klu'),
                                     ('linsolve', 0),
                                     )))
        self.config.add_extra("_help",
                              sparselib="linear sparse solver name",
                              linsolve="solve symbolic factorization each step (enable when KLU segfaults)",
                              )
        self.config.add_extra("_alt",
                              sparselib=("klu", "umfpack", "spsolve", "cupy"),
                              linsolve=(0, 1),
                              )

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
