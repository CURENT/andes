from andes.core import ModelData, IdxParam, NumParam, Model, ExtParam, ExtAlgeb, ExtService, ConstService, Algeb
from andes.core.block import Integrator
from andes.core.var import AliasState


class WTDTA1Data(ModelData):
    """
    Data for WTDTA1 wind drive-train model.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.ree = IdxParam(mandatory=True,
                            info='Renewable exciter idx',
                            )

        self.Ht = NumParam(default=3.0, tex_name='H_t',
                           info='Turbine inertia', unit='MWs/MVA',
                           power=True,
                           non_zero=True,
                           )

        self.Hg = NumParam(default=3.0, tex_name='H_g',
                           info='Generator inertia', unit='MWs/MVA',
                           power=True,
                           non_zero=True,
                           )

        self.Dshaft = NumParam(default=1.0, tex_name='D_{shaft}',
                               info='Damping coefficient',
                               unit='p.u. (gen base)',
                               power=True,
                               )

        self.Kshaft = NumParam(default=1.0, tex_name='K_{shaft}',
                               info='Spring constant',
                               unit='p.u. (gen base)',
                               power=True,
                               )

        self.w0 = NumParam(default=1.0, tex_name=r'\omega_0',
                           info='Default speed if not using a torque model',
                           unit='p.u.',
                           )


class WTDTA1Model(Model):
    """
    WTDTA1 model equations
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
                            e_str='-1.0 + s2_y'
                            )

        self.Pe = ExtAlgeb(model='RenGen', src='Pe', indexer=self.reg, export=False,
                           info='Retrieved Pe of RenGen')

        self.Pe0 = ExtService(model='RenGen', src='Pe', indexer=self.reg, tex_name='P_{e0}',
                              )

        self.Ht2 = ConstService(v_str='2 * Ht', tex_name='2H_t')

        self.Hg2 = ConstService(v_str='2 * Hg', tex_name='2H_g')

        self.wr0 = Algeb(tex_name=r'\omega_{r0}',
                         unit='p.u.',
                         v_str='w0',
                         e_str='w0 - wr0',
                         info='speed set point',
                         )

        self.Pm = Algeb(tex_name='P_m',
                        info='Mechanical power',
                        e_str='Pe0 - Pm',
                        v_str='Pe0',
                        )

        # `s1_y` is `wt`
        self.s1 = Integrator(u='(Pm / s1_y) - (Kshaft * s3_y ) - pd',
                             T=self.Ht2,
                             K=1.0,
                             y0='wr0',
                             )

        self.wt = AliasState(self.s1_y, tex_name=r'\omega_t')

        # `s2_y` is `wg`
        self.s2 = Integrator(u='-(Pe / s2_y) + (Kshaft * s3_y ) + pd',
                             T=self.Hg2,
                             K=1.0,
                             y0='wr0',
                             )

        self.wg = AliasState(self.s2_y, tex_name=r'\omega_g')

        # TODO: `s3_y` needs to be properly reinitialized with the new `wr0`
        self.s3 = Integrator(u='s1_y - s2_y',
                             T=1.0,
                             K=1.0,
                             y0='Pe0 / wr0 / Kshaft',
                             )

        self.pd = Algeb(tex_name='P_d', info='Output after damping',
                        v_str='0.0',
                        e_str='Dshaft * (s1_y - s2_y) - pd',
                        )


class WTDTA1(WTDTA1Data, WTDTA1Model):
    """
    WTDTA wind turbine drive-train model.

    User-provided reference speed should be specified in parameter `w0`.
    Internally, `w0` is set to the algebraic variable `wr0`.
    """

    def __init__(self, system, config):
        WTDTA1Data.__init__(self)
        WTDTA1Model.__init__(self, system, config)
