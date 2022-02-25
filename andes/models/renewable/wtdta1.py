from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        IdxParam, Model, ModelData, NumParam,)
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

        self.H = NumParam(default=3.0, tex_name='H_t',
                          info='Total inertia constant', unit='MWs/MVA',
                          power=True,
                          non_zero=True,
                          non_negative=True,
                          )

        self.DAMP = NumParam(default=0.0, tex_name='Damp',
                             info='Damp coefficient',
                             unit='p.u. (gen base)',
                             power=True,
                             )

        self.Htfrac = NumParam(default=0.5, tex_name='D_{shaft}',
                               info='Turbine inertia fraction (Hturb/H)',
                               power=True,
                               vrange='[0, 1]',
                               )

        self.Freq1 = NumParam(default=1, tex_name='Freq1',
                              unit='p.u. (Hz)',
                              info='',
                              )

        self.Dshaft = NumParam(default=1.0, tex_name='D_{shaft}',
                               info='Damping coefficient',
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

        self.reg = ExtParam(model='RenExciter', src='reg', indexer=self.ree, vtype=str,
                            export=False,
                            )
        self.Sn = ExtParam(model='RenGen', src='Sn', indexer=self.reg,
                           tex_name='S_n', export=False,
                           )

        self.wge = ExtAlgeb(model='RenExciter', src='wg', indexer=self.ree,
                            export=False,
                            e_str='-1.0 + s2_y',
                            ename='wg',
                            tex_ename=r'\omega_g',
                            )

        self.Pe = ExtAlgeb(model='RenGen', src='Pe', indexer=self.reg, export=False,
                           info='Retrieved Pe of RenGen')

        self.Pe0 = ExtService(model='RenGen', src='Pe', indexer=self.reg, tex_name='P_{e0}',
                              )

        self.Ht2 = ConstService(v_str='2 * (Htfrac * H)', tex_name='2H_t')

        self.Hg2 = ConstService(v_str='2 * H * (1 - Htfrac)', tex_name='2H_g')

        # (2*pi*Freq1)**2 is considered in p.u., which is Freq1**2 here
        self.Kshaft = ConstService(v_str='Ht2 * Hg2 * 0.5 * Freq1 * Freq1 / H', tex_name='K_{shaft}')

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
        self.s1 = Integrator(u='(Pm / s1_y) - s3_y - pd',
                             T=self.Ht2,
                             K=1.0,
                             y0='wr0',
                             )

        self.wt = AliasState(self.s1_y, tex_name=r'\omega_t')

        # `s2_y` is `wg`
        self.s2 = Integrator(u='-(Pe / s2_y) + s3_y - DAMP * (s2_y - w0) + pd',
                             T=self.Hg2,
                             K=1.0,
                             y0='wr0',
                             )

        self.wg = AliasState(self.s2_y, tex_name=r'\omega_g')

        # TODO: `s3_y` needs to be properly reinitialized with the new `wr0`
        self.s3 = Integrator(u='s1_y - s2_y',
                             T=1.0,
                             K=self.Kshaft,
                             y0='Pe0 / wr0',
                             )

        self.pd = Algeb(tex_name='P_d', info='Output after damping',
                        e_str='Dshaft * (s1_y - s2_y) - pd',
                        v_str='0',
                        )


class WTDTA1(WTDTA1Data, WTDTA1Model):
    """
    WTDTA wind turbine drive-train model.

    User-provided reference speed should be specified in parameter `w0`.
    Internally, `w0` is set to the algebraic variable `wr0`.

    <PSS/E parser notice>

    In PSSE doc, `Freq1` is said to be Hz, but ANDES apply per unit here.
    If the value of `Freq1` is given in Hz, it is suggested to convert it to
    per unit in the dyr file.
    """

    def __init__(self, system, config):
        WTDTA1Data.__init__(self)
        WTDTA1Model.__init__(self, system, config)
