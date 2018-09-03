from . import ConfigBase
from ..utils.cached import cached


class CPF(ConfigBase):
    def __init__(self, **kwargs):
        self.method = 'perpendicular intersection'
        self.single_slack = False
        self.reactive_limits = False
        self.nump = 1000
        self.mu_init = 1.0
        self.hopf = False
        self.step = 0.1
        super(CPF, self).__init__(**kwargs)

    @cached
    def config_descr(self):
        descriptions = {
            'method': 'method for CPF routine analysis',
            'single_slack': 'use single slack bus mode',
            'reactive_limits': 'consider reactive power limits',
        }
        return descriptions
