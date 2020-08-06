import logging
from collections import OrderedDict
from andes.core.model import Model, ModelData
from andes.core.param import IdxParam, DataParam, NumParam
from andes.core.var import Algeb, ExtAlgeb, State  # NOQA
logger = logging.getLogger(__name__)


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


class DC2Term(ModelData, Model):
    """Two-terminal DC device template"""
    def __init__(self, system, config):
        ModelData.__init__(self)
        self.node1 = IdxParam(default=None,
                              info='Node 1 index',
                              mandatory=True,
                              model='Node',
                              )
        self.node2 = IdxParam(default=None,
                              info='Node 2 index',
                              mandatory=True,
                              model='Node',
                              )
        self.Vdcn1 = NumParam(default=100,
                              info='DC voltage rating on node 1',
                              unit='kV',
                              non_zero=True,
                              tex_name='V_{dcn1}',
                              )
        self.Vdcn2 = NumParam(default=100,
                              info='DC voltage rating on node 2',
                              unit='kV',
                              non_zero=True,
                              tex_name='V_{dcn2}',
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
                           info='DC voltage on node 1',
                           )
        self.v2 = ExtAlgeb(model='Node',
                           src='v',
                           indexer=self.node2,
                           info='DC voltage on node 2',
                           )


class ACDC2Term(ModelData, Model):
    """AC to two-terminal DC device template"""
    def __init__(self, system, config):
        ModelData.__init__(self)
        self.bus = IdxParam(model='Bus',
                            info="idx of connected bus",
                            mandatory=True,
                            )
        self.node1 = IdxParam(default=None,
                              info='Node 1 index',
                              mandatory=True,
                              model='Node',
                              )
        self.node2 = IdxParam(default=None,
                              info='Node 2 index',
                              mandatory=True,
                              model='Node',
                              )
        self.Vn = NumParam(default=110.0,
                           info="AC voltage rating",
                           non_zero=True,
                           tex_name=r'V_n',
                           )
        self.Vdcn1 = NumParam(default=100,
                              info='DC voltage rating on node 1',
                              unit='kV',
                              non_zero=True,
                              tex_name='V_{dcn1}',
                              )
        self.Vdcn2 = NumParam(default=100,
                              info='DC voltage rating on node 2',
                              unit='kV',
                              non_zero=True,
                              tex_name='V_{dcn2}',
                              )
        self.Idcn = NumParam(default=1,
                             info='DC current rating',
                             unit='kA',
                             non_zero=True,
                             tex_name='I_{dcn}',
                             )

        Model.__init__(self, system, config)
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          info='AC bus voltage phase',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          info='AC bus voltage magnitude',
                          )
        self.v1 = ExtAlgeb(model='Node',
                           src='v',
                           indexer=self.node1,
                           info='DC node 1 voltage',
                           )
        self.v2 = ExtAlgeb(model='Node',
                           src='v',
                           indexer=self.node2,
                           info='DC node 2 voltage',
                           )


class Ground(ModelData, Model):
    """
    Ground model that sets the voltage of the connected DC node.
    """

    def __init__(self, system, config):
        ModelData.__init__(self)
        self.node = IdxParam(default=None,
                             info='Node index',
                             mandatory=True,
                             model='Node',
                             )
        self.voltage = NumParam(default=0.0,
                                tex_name='V_0',
                                info='Ground voltage (typically 0)',
                                unit='p.u.',
                                )
        Model.__init__(self, system, config)
        self.flags.update({'pflow': True})
        self.group = 'DCLink'
        self.v = ExtAlgeb(model='Node',
                          src='v',
                          indexer=self.node,
                          e_str='-Idc',
                          )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Ficticious current injection from ground',
                         e_str='u * (v - voltage)',
                         v_str='0',
                         diag_eps=True,
                         )
        self.v.e_str = '-Idc'


class R(DC2Term):
    """
    Resistive dc line
    """
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'
        self.R = NumParam(unit='p.u.',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         v_str='u * (v2 - v1) / R',
                         e_str='u * (v2 - v1) / R - Idc',
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class L(DC2Term):
    """
    Inductive dc line
    """
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.L = NumParam(unit='p.u.',
                          info='DC line inductance',
                          non_zero=True,
                          default=0.001,
                          r=True,
                          )
        self.IL = State(tex_name='I_L',
                        info='Inductance current',
                        unit='p.u.',
                        v_str='0',
                        e_str='-u * (v1 - v2) / L',
                        )
        self.v1.e_str = '-IL'
        self.v2.e_str = '+IL'


class C(DC2Term):
    """
    Capacitive dc branch
    """
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.C = NumParam(unit='p.u.',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        v_str='0',
                        e_str='-u * Idc / C',
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         v_str='0',
                         e_str='u * (vC - (v1 - v2)) + '
                               '(1 - u) * Idc',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RLs(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.L = NumParam(unit='p.u.',
                          tex_name='L',
                          info='DC line inductance',
                          non_zero=True,
                          default=0.001,
                          r=True,
                          )
        self.IL = State(tex_name='I_L',
                        info='Inductance current',
                        unit='p.u.',
                        e_str='u * (v1 - v2 - R * IL) / L',
                        v_str='(v1 - v2) / R',
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='-u * IL - Idc',
                         v_str='-u * (v1 - v2) / R',
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RCp(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.C = NumParam(unit='p.u.',
                          tex_name='C',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        e_str='-u * (Idc - vC/R) / C',
                        v_str='v1 - v2',
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='u * (vC - (v1 - v2)) + '
                               '(1 - u) * Idc',
                         v_str='-(v1 - v2) / R',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RLCp(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.L = NumParam(unit='p.u.',
                          tex_name='L',
                          info='DC line inductance',
                          non_zero=True,
                          default=0.001,
                          r=True,
                          )
        self.C = NumParam(unit='p.u.',
                          tex_name='C',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.IL = State(tex_name='I_L',
                        info='Inductance current',
                        unit='p.u.',
                        v_str='0',
                        e_str='u * vC / L',
                        )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        e_str='-u * (Idc - vC/R - IL) / C',
                        v_str='v1 - v2',
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='u * (vC - (v1 - v2)) + '
                               '(1 - u) * Idc',
                         v_str='-(v1 - v2) / R',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RCs(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.C = NumParam(unit='p.u.',
                          tex_name='C',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        e_str='-u * Idc / C',
                        v_str='v1 - v2',
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='u * (vC - (v1 - v2) - Idc * R) + '
                               '(1 - u) * Idc',
                         v_str='-(v1 - v2) / R',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RLCs(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.L = NumParam(unit='p.u.',
                          tex_name='L',
                          info='DC line inductance',
                          non_zero=True,
                          default=0.001,
                          r=True,
                          )
        self.C = NumParam(unit='p.u.',
                          tex_name='C',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.IL = State(tex_name='I_L',
                        info='Inductance current',
                        unit='p.u.',
                        e_str='u * (v1 - v2 - R * IL - vC) / L',
                        v_str='0',
                        )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        e_str='u * IL / C',
                        v_str='v1 - v2',
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='-IL - Idc',
                         v_str='0',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'
