from . import ConfigBase
from ..utils.cached import cached


class Eig(ConfigBase):
    def __init__(self, **kwargs):
        self.neig = 1
        self.method = 1
        self.map = 1
        self.matrix = 4
        self.report = ''
        self.eigs = ''
        self.pf = ''
        self.plot = True
        super(Eig, self).__init__(**kwargs)

    @cached
    def config_descr(self):
        descriptions = {}
        return descriptions
