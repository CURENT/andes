from andes.core import ModelData, Model, IdxParam, NumParam, ExtAlgeb, Algeb


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
                          ename='-Idc',
                          tex_ename='-I_{dc}',
                          )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Ficticious current injection from ground',
                         e_str='u * (v - voltage)',
                         v_str='0',
                         diag_eps=True,
                         )
        self.v.e_str = '-Idc'
