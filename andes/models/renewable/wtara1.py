from andes.core import (Algeb, ConstService, ExtAlgeb, ExtService, IdxParam,
                        Model, ModelData, NumParam,)


class WTARA1Data(ModelData):
    """
    Wind turbine aerodynamics model data.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.rego = IdxParam(mandatory=True,
                             info='Renewable governor idx',
                             )

        self.Ka = NumParam(default=1.0, info='Aerodynamics gain',
                           tex_name='K_a',
                           non_negative=True,
                           unit='p.u./deg.'
                           )

        self.theta0 = NumParam(default=0.0, info='Initial pitch angle',
                               tex_name=r'\theta_0',
                               unit='deg.',
                               )
        # TODO: check how to treat `theta0` if pitch controller is provided


class WTARA1Model(Model):
    """
    Wind turbine aerodynamics model equations.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'RenAerodynamics'

        self.theta0r = ConstService(v_str='rad(theta0)',
                                    tex_name=r'\theta_{0r}',
                                    info='Initial pitch angle in radian',
                                    )

        self.theta = Algeb(tex_name=r'\theta',
                           info='Pitch angle',
                           unit='rad',
                           v_str='theta0r',
                           e_str='theta0r - theta',
                           )

        self.Pe0 = ExtService(model='RenGovernor',
                              src='Pe0',
                              indexer=self.rego,
                              tex_name='P_{e0}',
                              )

        self.Pmg = ExtAlgeb(model='RenGovernor',
                            src='Pm',
                            indexer=self.rego,
                            e_str='-Pe0 - (theta - theta0) * theta * Ka + Pe0',
                            ename='Pmg',
                            tex_ename='P_{mg}',
                            )


class WTARA1(WTARA1Data, WTARA1Model):
    """
    Wind turbine aerodynamics model (no wind speed details).
    """

    def __init__(self, system, config):
        WTARA1Data.__init__(self)
        WTARA1Model.__init__(self, system, config)
