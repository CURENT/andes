from ..settings.base import SettingsBase
from ..utils.cached import cached


class CPF(SettingsBase):

    def __init__(self):
        self.method = 'perpendicular intersection'
        self.single_slack = False
        self.reactive_limits = False
        self.nump = 1000
        self.mu_init = 1.0
        self.hopf = False
        self.step = 0.1

    @cached
    def doc_help(self):
        descriptions = {'method': 'method for CPF routine analysis',
                        'single_slack': 'use single slack bus mode',
                        'reactive_limits': 'consider reactive power limits',
                        }
        return descriptions
