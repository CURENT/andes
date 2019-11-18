from andes.common.solver import Solver
from andes.core.model import ModelConfig
from collections import OrderedDict


class ProgramBase(object):

    def __init__(self, system=None, config=None):
        self.system = system
        self.config = ModelConfig()

        if config is not None:
            self.set_config(config)

        self.config.add(sparselib='klu')

        self.solver = Solver(sparselib=self.config.sparselib)

    @staticmethod
    def class_name(self):
        return self.__class__.__name__

    def set_config(self, config):
        if self.class_name in config:
            config_section = config[self.class_name]
            self.config.add(**OrderedDict(config_section))
