from andes.core import ModelData, Model, IdxParam, NumParam, ExtAlgeb


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
