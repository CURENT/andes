"""
Base classes for AC/DC devices.
"""

from andes.core import ModelData, Model, IdxParam, NumParam, ExtAlgeb


class ACDC2Term(ModelData, Model):
    """
    AC to two-terminal DC device template.
    """

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
