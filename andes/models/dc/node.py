"""
DC node model.
"""

from collections import OrderedDict

from andes.core import ModelData, NumParam, DataParam, IdxParam, Model, Algeb


class NodeData(ModelData):
    def __init__(self):
        """
        DC Node data.
        """
        super().__init__()
        self.Vdcn = NumParam(default=100,
                             info='DC voltage rating',
                             unit='kV',
                             non_zero=True,
                             tex_name='V_{dcn}',
                             )
        self.Idcn = NumParam(default=1,
                             info='DC current rating',
                             unit='kA',
                             non_zero=True,
                             tex_name='I_{dcn}',
                             )
        self.v0 = NumParam(default=1.0,
                           info="initial voltage magnitude",
                           tex_name=r'V_{dc0}',
                           unit='p.u.',
                           )
        self.xcoord = DataParam(default=0,
                                info='x coordinate (longitude)',
                                )
        self.ycoord = DataParam(default=0,
                                info='y coordinate (latitude)',
                                )
        self.area = IdxParam(model='Area',
                             default=None,
                             info="Area code",
                             )
        self.zone = IdxParam(model='Region',
                             default=None,
                             info="Zone code",
                             )
        self.owner = IdxParam(model='Owner',
                              default=None,
                              info="Owner code",
                              )


class Node(NodeData, Model):
    """
    DC Node model.

    A DC Node is like an AC Bus. DC devices need to be connected to
    Nodes to inject power flow.
    """

    def __init__(self, system, config):
        NodeData.__init__(self)
        Model.__init__(self, system=system, config=config)

        self.config.add(OrderedDict((('flat_start', 0),
                                     )))
        self.config.add_extra("_help",
                              flat_start="flat start for voltages",
                              )
        self.config.add_extra("_alt",
                              flat_start=(0, 1),
                              )
        self.config.add_extra("_tex",
                              flat_start="z_{flat}",
                              )

        self.group = 'DCTopology'
        self.category = ['TransNode']
        self.flags.update({'pflow': True})

        self.v = Algeb(name='v',
                       tex_name='V_{dc}',
                       info='voltage magnitude',
                       unit='p.u.',
                       diag_eps=True,
                       )

        self.v.v_str = 'flat_start*1 + ' \
                       '(1-flat_start)*v0'
