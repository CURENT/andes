"""
Phasor-domain shunt compensator model.
"""

from andes.core import ModelData, IdxParam, NumParam, Model, ExtAlgeb


class ShuntData(ModelData):

    def __init__(self, system=None, name=None):
        super().__init__(system, name)

        self.bus = IdxParam(model='Bus', info="idx of connected bus", mandatory=True)

        self.Sn = NumParam(default=100.0, info="Power rating", non_zero=True, tex_name='S_n')
        self.Vn = NumParam(default=110.0, info="AC voltage rating", non_zero=True, tex_name='V_n')
        self.g = NumParam(default=0.0, info="shunt conductance (real part)", y=True, tex_name='g')
        self.b = NumParam(default=0.0, info="shunt susceptance (positive as capacitive)", y=True, tex_name='b')
        self.fn = NumParam(default=60.0, info="rated frequency", tex_name='f_n')


class ShuntModel(Model):
    """
    Shunt equations.
    """

    def __init__(self, system=None, config=None):
        Model.__init__(self, system, config)
        self.group = 'StaticShunt'
        self.flags.pflow = True
        self.flags.tds = True

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus, tex_name=r'\theta',
                          ename='P',
                          tex_ename='P',
                          )
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name='V',
                          ename='Q',
                          tex_ename='Q',
                          )

        self.a.e_str = 'u * v**2 * g'
        self.v.e_str = '-u * v**2 * b'


class Shunt(ShuntData, ShuntModel):
    """
    Phasor-domain shunt compensator Model.
    """

    def __init__(self, system=None, config=None):
        ShuntData.__init__(self)
        ShuntModel.__init__(self, system, config)
