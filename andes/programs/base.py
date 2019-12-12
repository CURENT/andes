from andes.common.solver import Solver
from andes.common.config import Config
from collections import OrderedDict


class ProgramBase(object):

    def __init__(self, system=None, config=None):
        self.system = system
        self.config = Config(self.class_name)

        if config is not None:
            self.config.load(config)

        self.config.add(OrderedDict((('sparselib', 'klu'), )))

        self.solver = Solver(sparselib=self.config.sparselib)

    @property
    def class_name(self):
        return self.__class__.__name__
