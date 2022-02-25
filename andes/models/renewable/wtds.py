from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        IdxParam, Model, ModelData, NumParam, State,)
from andes.core.block import Integrator
from andes.core.var import AliasState


class WTDSData(ModelData):
    """
    Wind turbine governor swing equation model data.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.ree = IdxParam(mandatory=True,
                            info='Renewable exciter idx',
                            )

        self.H = NumParam(default=3.0, tex_name='H_t',
                          info='Total inertia', unit='MWs/MVA',
                          power=True,
                          non_zero=True,
                          non_negative=True,
                          )

        self.D = NumParam(default=1.0, tex_name='D_{shaft}',
                          info='Damping coefficient',
                          unit='p.u.',
                          power=True,
                          )

        self.w0 = NumParam(default=1.0, tex_name=r'\omega_0',
                           info='Default speed if not using a torque model',
                           unit='p.u.',
                           )


class WTDSModel(Model):
    """
    Wind turbine one-mass generator model.

    User-provided reference speed should be specified in parameter `w0`.
    Internally, `w0` is set to the algebraic variable `wr0`.

    See J.H. Chow, J.J. Sanchez-Gasca. "Power System Modeling, Computation, and Control".
    John Wiley & Sons, 2020. pp. 518.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'RenGovernor'

        self.reg = ExtParam(model='RenExciter', src='reg', indexer=self.ree,
                            export=False,
                            )

        self.Sn = ExtParam(model='RenGen', src='Sn', indexer=self.reg,
                           tex_name='S_n', export=False,
                           )

        self.wge = ExtAlgeb(model='RenExciter', src='wg', indexer=self.ree,
                            export=False,
                            e_str='-1.0 + s1_y',
                            ename='wg',
                            tex_ename=r'\omega_g',
                            )

        self.Pe = ExtAlgeb(model='RenGen', src='Pe', indexer=self.reg, export=False,
                           info='Retrieved Pe of RenGen')

        self.Pe0 = ExtService(model='RenGen', src='Pe', indexer=self.reg, tex_name='P_{e0}',
                              )

        self.H2 = ConstService(v_str='2 * H', tex_name='2H')

        self.Pm = Algeb(tex_name='P_m',
                        info='Mechanical power',
                        e_str='Pe0 - Pm',
                        v_str='Pe0',
                        )

        self.wr0 = Algeb(tex_name=r'\omega_{r0}',
                         unit='p.u.',
                         v_str='w0',
                         e_str='w0 - wr0',
                         info='speed set point',
                         )

        # `s1_y` is `w_m`
        self.s1 = Integrator(u='(Pm - Pe) / wge - D * (s1_y - wr0)',
                             T=self.H2,
                             K=1.0,
                             y0='wr0',
                             )

        # make two alias states, `wt` and `wg`, pointing to `s1_y`
        self.wt = AliasState(self.s1_y, tex_name=r'\omega_t')

        self.wg = AliasState(self.s1_y, tex_name=r'\omega_g')

        self.s3_y = State(info='Unused state variable',
                          tex_name='y_{s3}',
                          )

        self.Kshaft = ConstService(v_str='1.0', tex_name='K_{shaft}',
                                   info='Dummy Kshaft',
                                   )


class WTDS(WTDSData, WTDSModel):
    """
    Custom wind turbine model with a single swing-equation.

    This model is used to simulate the mechanical swing
    of the combined machine and turbine mass. The speed output
    is ``s1_y`` which will be fed to ``RenExciter.wg``.

    ``PFLAG`` needs to be set to ``1`` in exciter to consider
    speed for Pref.
    """

    def __init__(self, system, config):
        WTDSData.__init__(self)
        WTDSModel.__init__(self, system, config)
