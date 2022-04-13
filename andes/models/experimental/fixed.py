from andes.core import ModelData, IdxParam, Model, ExtAlgeb, ExtService


class FixedGenData(ModelData):
    """
    Data for fixed dynamic generator
    """

    def __init__(self):
        ModelData.__init__(self)

        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            mandatory=True,
                            )


class FixedGenModel(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'Experimental'

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle',
                          e_str='-p0',
                          ename='P',
                          tex_ename='P',
                          )

        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          e_str='-q0',
                          ename='P',
                          tex_ename='P',
                          )

        self.p0 = ExtService(model='StaticGen',
                             src='p',
                             indexer=self.gen,
                             tex_name='P_0',
                             )
        self.q0 = ExtService(model='StaticGen',
                             src='q',
                             indexer=self.gen,
                             tex_name='Q_0',
                             )

    def v_numeric(self, **kwargs):
        """
        Disable the corresponding `StaticGen`s.
        """
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class FixedGen(FixedGenData, FixedGenModel):
    def __init__(self, system, config):
        FixedGenData.__init__(self)
        FixedGenModel.__init__(self, system, config)
