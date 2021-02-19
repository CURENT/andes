from andes.core import ModelData, IdxParam, NumParam, Model, ExtParam, ExtAlgeb, ExtService, ExtState, Algeb
from andes.core.block import PIAWHardLimit, LagAntiWindupRate


class WTPTA1Data(ModelData):
    """
    Pitch control model data.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.rea = IdxParam(mandatory=True,
                            info='Renewable aerodynamics model idx',
                            )

        self.Kiw = NumParam(default=0.1, info='Pitch-control integral gain',
                            tex_name='K_{iw}',
                            unit='p.u.',
                            )

        self.Kpw = NumParam(default=0.0, info='Pitch-control proportional gain',
                            tex_name='K_{pw}',
                            unit='p.u.',
                            )

        self.Kic = NumParam(default=0.1, info='Pitch-compensation integral gain',
                            tex_name='K_{ic}',
                            unit='p.u.',
                            )

        self.Kpc = NumParam(default=0.0, info='Pitch-compensation proportional gain',
                            tex_name='K_{pc}',
                            unit='p.u.',
                            )

        self.Kcc = NumParam(default=0.0, info='Gain for P diff',
                            tex_name='K_{cc}',
                            unit='p.u.',
                            )

        self.Tp = NumParam(default=0.3, info='Blade response time const.',
                           tex_name=r'T_{\theta}',
                           unit='s',
                           )

        self.thmax = NumParam(default=30.0, info='Max. pitch angle',
                              tex_name=r'\theta_{max}',
                              unit='deg.',
                              vrange=(27, 30),
                              )
        self.thmin = NumParam(default=0.0, info='Min. pitch angle',
                              tex_name=r'\theta_{min}',
                              unit='deg.',
                              )
        self.dthmax = NumParam(default=5.0, info='Max. pitch angle rate',
                               tex_name=r'\theta_{max}',
                               unit='deg.',
                               vrange=(5, 10),
                               )
        self.dthmin = NumParam(default=-5.0, info='Min. pitch angle rate',
                               tex_name=r'\theta_{min}',
                               unit='deg.',
                               vrange=(-10, -5),
                               )


class WTPTA1Model(Model):
    """
    Pitch control model equations.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'RenPitch'

        self.rego = ExtParam(model='RenAerodynamics', src='rego', indexer=self.rea,
                             export=False,
                             )

        self.ree = ExtParam(model='RenGovernor', src='ree', indexer=self.rego,
                            export=False,
                            )

        self.wt = ExtAlgeb(model='RenGovernor', src='wt', indexer=self.rego,
                           export=False,
                           )

        self.theta0 = ExtService(model='RenAerodynamics', src='theta0', indexer=self.rea,
                                 )

        self.theta = ExtAlgeb(model='RenAerodynamics', src='theta', indexer=self.rea,
                              export=False,
                              e_str='-theta0 + LG_y'
                              )

        self.Pord = ExtState(model='RenExciter', src='Pord', indexer=self.ree,
                             )

        self.Pref = ExtAlgeb(model='RenExciter', src='Pref', indexer=self.ree,
                             )

        self.PIc = PIAWHardLimit(u='Pord - Pref', kp=self.Kpc, ki=self.Kic,
                                 aw_lower=self.thmin, aw_upper=self.thmax,
                                 lower=self.thmin, upper=self.thmax,
                                 tex_name='PI_c',
                                 info='PI for active power diff compensation',
                                 )

        self.wref = Algeb(tex_name=r'\omega_{ref}',
                          info='optional speed reference',
                          e_str='wt - wref',
                          v_str='wt',
                          )

        self.PIw = PIAWHardLimit(u='Kcc * (Pord - Pref) + wt - wref', kp=self.Kpw, ki=self.Kiw,
                                 aw_lower=self.thmin, aw_upper=self.thmax,
                                 lower=self.thmin, upper=self.thmax,
                                 tex_name='PI_w',
                                 info='PI for speed and active power deviation',
                                 )

        self.LG = LagAntiWindupRate(u='PIw_y + PIc_y', T=self.Tp, K=1.0,
                                    lower=self.thmin, upper=self.thmax,
                                    rate_lower=self.dthmin, rate_upper=self.dthmax,
                                    tex_name='LG',
                                    info='Output lag anti-windup rate limiter')

        # remove warning when pitch angle==0
        self.PIc_hl.warn_flags.pop(0)
        self.PIc_aw.warn_flags.pop(0)
        self.PIw_hl.warn_flags.pop(0)
        self.PIw_aw.warn_flags.pop(0)
        self.LG_lim.warn_flags.pop(0)


class WTPTA1(WTPTA1Data, WTPTA1Model):
    """
    Wind turbine pitch control model.
    """

    def __init__(self, system, config):
        WTPTA1Data.__init__(self)
        WTPTA1Model.__init__(self, system, config)
