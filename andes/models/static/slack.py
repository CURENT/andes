"""
Slack generator model for steady state.
"""

from collections import OrderedDict

from andes.core import NumParam, ExtParam, SortedLimiter, Algeb
from andes.models.static.pv import PVData, PVModel


class SlackData(PVData):
    def __init__(self):
        super().__init__()
        self.a0 = NumParam(default=0.0,
                           info="reference angle set point",
                           tex_name=r'\theta_0')


class Slack(SlackData, PVModel):
    """
    Slack generator.
    """

    def __init__(self, system=None, config=None):
        SlackData.__init__(self)
        PVModel.__init__(self, system, config)

        self.config.add(OrderedDict((('av2pv', 0),
                                     )))
        self.config.add_extra("_help",
                              av2pv="convert Slack to PV in PFlow at P limits",
                              )
        self.config.add_extra("_alt",
                              av2pv=(0, 1),
                              )
        self.config.add_extra("_tex",
                              av2pv="z_{av2pv}",
                              )
        self.busa0 = ExtParam(model='Bus', src='a0', indexer=self.bus,
                              export=False, tex_name=r'\theta_{0bus}',
                              )
        self.a.v_setter = True
        self.a.v_str = 'u * a0 + (1-u) * busa0'

        delattr(self, 'p')
        self.services.pop('p')

        self.p = Algeb(info='actual active power generation',
                       unit='p.u.',
                       tex_name=r'p',
                       diag_eps=True,
                       v_str='u * p0',
                       )

        self.p.e_str = "u*(plim_zi * (a0-a) + " \
                       "plim_zl * (pmin-p) + " \
                       "plim_zu * (pmax-p))"
        self.a.e_str = '-u * p'

        self.plim = SortedLimiter(u=self.p, lower=self.pmin, upper=self.pmax,
                                  enable=self.config.av2pv)
