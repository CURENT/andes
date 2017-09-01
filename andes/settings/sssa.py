from ..settings.base import SettingsBase
from ..utils.cached import cached


class SSSA(SettingsBase):

    def __init__(self):
        self.neig = 1
        self.method = 1
        self.map = 1
        self.matrix = 4
        self.report = ''
        self.eigs = ''
        self.pf = ''
        self.plot = True

    @cached
    def doc_help(self):
        descriptions = {

        }
        return descriptions
