from andes.core.solver import Solver
from andes.core import Config
from collections import OrderedDict


class BaseRoutine(object):
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
        self.config.add_extra("_alt", sparselib=("klu", "umfpack", "spsolve", "cupy"),
                              linsolve=(0, 1),
                              )

        self.solver = Solver(sparselib=self.config.sparselib)

    @property
    def class_name(self):
        return self.__class__.__name__

    def doc(self, max_width=78, export='plain'):
        return self.config.doc(max_width, export)

    def init(self):
        pass

    def run(self, **kwargs):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError
