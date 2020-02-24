import logging
from collections import OrderedDict
from andes.core.model import Model, ModelData
from andes.core.param import IdxParam, DataParam, NumParam
from andes.core.var import Algeb, ExtAlgeb, State  # NOQA
logger = logging.getLogger(__name__)


class NodeData(ModelData):
    def __init__(self):
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
                           non_zero=True,
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
    def __init__(self, system, config):
        NodeData.__init__(self)
        Model.__init__(self, system=system, config=config)

        self.config.add(OrderedDict((('flat_start', 0.0),
                                     )))

        self.group = 'DCTopology'
        self.category = ['TransNode']
        self.flags.update({'pflow': True})

        self.v = Algeb(name='v',
                       tex_name='V_{dc}',
                       info='voltage magnitude',
                       unit='p.u.',
                       # diag_eps=1e-6,
                       )

        self.v.v_str = 'flat_start * 1 + ' \
                       '(1 - flat_start) * v0'


class DC2Term(ModelData, Model):
    def __init__(self, system, config):
        ModelData.__init__(self)
        self.node1 = IdxParam(default=None,
                              tex_name='node_1',
                              info='Node 1 index',
                              mandatory=True,
                              model='Node',
                              )
        self.node2 = IdxParam(default=None,
                              tex_name='node_2',
                              info='Node 2 index',
                              mandatory=True,
                              model='Node',
                              )
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

        Model.__init__(self, system, config)
        self.v1 = ExtAlgeb(model='Node',
                           src='v',
                           indexer=self.node1,
                           )
        self.v2 = ExtAlgeb(model='Node',
                           src='v',
                           indexer=self.node2,
                           )


class R(DC2Term):
    """
    Resistive dc line
    """
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.R = NumParam(unit='p.u.',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='u * (v2 - v1) / R - Idc',
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'
