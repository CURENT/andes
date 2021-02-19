from andes.core import ModelData, IdxParam, NumParam, Model, ExtParam, ExtService, ExtAlgeb, Algeb


class WTARV1Data(ModelData):
    """
    Data for wind turbine aerodynamics with wind velocity initialization.
    """
    def __init__(self):
        ModelData.__init__(self)

        self.rego = IdxParam(mandatory=True,
                             info='Renewable governor idx',
                             )
        self.nblade = NumParam(info='number of blades', default=3.0,
                               )

        self.ngen = NumParam(info='number of wind generator units', default=50,
                             )
        self.npole = NumParam(info='number of poles in generator', default=4,
                              )
        self.R = NumParam(info='rotor radius', default=30.0,
                          unit='m',
                          )
        self.ngb = NumParam(info='gear box ratio', default=5,
                            )
        self.rho = NumParam(info='air density', unit='kg/m3', default=1.20,
                            )


class WTARV1Model(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'RenAerodynamics'

        self.Sn = ExtParam(model='RenGovernor', src='Sn', indexer=self.rego,
                           tex_name='S_n', export=False,
                           )

        self.Pe0 = ExtService(model='RenGovernor',
                              src='Pe0',
                              indexer=self.rego,
                              tex_name='P_{e0}',
                              )

        self.Pmg = ExtAlgeb(model='RenGovernor',
                            src='Pm',
                            indexer=self.rego,
                            )

        self.theta = Algeb(tex_name=r'\theta',
                           info='Pitch angle',
                           unit='rad',
                           )


class WTARV1(WTARV1Data, WTARV1Model):
    """
    Wind turbine aerodynamics model with wind velocity details.

    Work is in progress.
    """

    def __init__(self, system, config):
        WTARV1Data.__init__(self)
        WTARV1Model.__init__(self, system, config)
