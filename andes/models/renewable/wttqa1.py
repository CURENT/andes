import numpy as np

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        ExtState, IdxParam, Lag, Model, ModelData, NumParam,
                        Piecewise, Switcher,)
from andes.core.block import PIAWHardLimit
from andes.core.service import NumSelect


class WTTQA1Data(ModelData):
    """
    Wind turbine torque (Pref) model.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.rep = IdxParam(mandatory=True,
                            info='RenPitch controller idx',
                            )

        self.Kip = NumParam(default=0.1, info='Pref-control integral gain',
                            tex_name='K_{ip}',
                            unit='p.u.',
                            )

        self.Kpp = NumParam(default=0.0, info='Pref-control proportional gain',
                            tex_name='K_{pp}',
                            unit='p.u.',
                            )

        self.Tp = NumParam(default=0.05, info='Pe sensing time const.',
                           tex_name='T_p',
                           unit='s',
                           )

        self.Twref = NumParam(default=30.0, info='Speed reference time const.',
                              tex_name='T_{wref}',
                              unit='s',
                              vrange=(30, 60),
                              )

        self.Temax = NumParam(default=1.2, info='Max. electric torque',
                              tex_name='T_{emax}',
                              unit='p.u.',
                              vrange=(1.1, 1.2),
                              power=True,
                              )

        self.Temin = NumParam(default=0.0, info='Min. electric torque',
                              tex_name='T_{emin}',
                              unit='p.u.',
                              power=True,
                              )

        self.Tflag = NumParam(info='Tflag; 1-power error, 0-speed error',
                              mandatory=True,
                              unit='bool',
                              )

        self.p1 = NumParam(default=0.2, info='Active power point 1',
                           unit='p.u.', tex_name='p_1',
                           power=True,
                           )
        self.sp1 = NumParam(default=0.58, info='Speed power point 1',
                            unit='p.u.', tex_name='s_{p1}',
                            )

        self.p2 = NumParam(default=0.4, info='Active power point 2',
                           unit='p.u.', tex_name='p_2',
                           power=True,
                           )
        self.sp2 = NumParam(default=0.72, info='Speed power point 2',
                            unit='p.u.', tex_name='s_{p2}',
                            )

        self.p3 = NumParam(default=0.6, info='Active power point 3',
                           unit='p.u.', tex_name='p_3',
                           power=True,
                           )
        self.sp3 = NumParam(default=0.86, info='Speed power point 3',
                            unit='p.u.', tex_name='s_{p3}',
                            )

        self.p4 = NumParam(default=0.8, info='Active power point 4',
                           unit='p.u.', tex_name='p_4',
                           power=True,
                           )
        self.sp4 = NumParam(default=1.0, info='Speed power point 4',
                            unit='p.u.', tex_name='s_{p4}',
                            )
        self.Tn = NumParam(default=np.nan, tex_name='T_n',
                           info='Turbine rating. Use Sn from gov if none.',
                           unit='MVA',
                           )


class WTTQA1Model(Model):
    """
    Wind turbine torque Pref model equations.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'RenTorque'

        self.kp1 = ConstService(v_str='(sp2 - sp1) / (p2 - p1)',
                                tex_name='k_{p1}',
                                )
        self.kp2 = ConstService(v_str='(sp3 - sp2) / (p3 - p2)',
                                tex_name='k_{p2}',
                                )
        self.kp3 = ConstService(v_str='(sp4 - sp3) / (p4 - p3)',
                                tex_name='k_{p3}',
                                )

        self.rea = ExtParam(model='RenPitch', src='rea', indexer=self.rep, export=False,
                            )

        self.rego = ExtParam(model='RenAerodynamics', src='rego', indexer=self.rea,
                             export=False,
                             )

        self.ree = ExtParam(model='RenGovernor', src='ree', indexer=self.rego,
                            export=False,
                            )

        self.reg = ExtParam(model='RenExciter', src='reg', indexer=self.ree,
                            export=False,)

        self.Sngo = ExtParam(model='RenGen', src='Sn', indexer=self.reg,
                             tex_name='S_{n,go}', export=False,
                             )

        self.Sn = NumSelect(self.Tn,
                            fallback=self.Sngo,
                            tex_name='S_n',
                            info='Turbine or RenGovernor rating',
                            )

        self.Pe = ExtAlgeb(model='RenGen', src='Pe', indexer=self.reg,
                           tex_name='P_e', export=False,
                           )

        self.s1 = Lag(u=self.Pe, T=self.Tp, K=1.0, tex_name='s_1',
                      info='Pe filter',
                      )

        self.fPe = Piecewise(u=self.s1_y,
                             points=('p1', 'p2', 'p3', 'p4'),
                             funs=('sp1',
                                   f'sp1 + ({self.s1_y.name} - p1) * kp1',
                                   f'sp2 + ({self.s1_y.name} - p2) * kp2',
                                   f'sp3 + ({self.s1_y.name} - p3) * kp3',
                                   'sp4'),
                             tex_name='f_{Pe}',
                             info='Piecewise Pe to wref mapping',
                             )

        # Overwrite `wg` and `wt` initial values in turbine governors
        self.wg = ExtState(model='RenGovernor', src='wg', indexer=self.rego,
                           tex_name=r'\omega_g', export=False,
                           v_str='fPe_y',
                           v_setter=True,
                           )

        self.wt = ExtState(model='RenGovernor', src='wt', indexer=self.rego,
                           tex_name=r'\omega_t', export=False,
                           v_str='fPe_y',
                           v_setter=True,
                           )

        self.s3_y = ExtState(model='RenGovernor', src='s3_y', indexer=self.rego,
                             tex_name='y_{s3}', export=False,
                             v_str='Pref0 / wg',
                             v_setter=True,
                             )

        self.w0 = ExtParam(model='RenGovernor', src='w0', indexer=self.rego,
                           tex_name=r'\omega_0', export=False,
                           )

        self.Kshaft = ExtService(model='RenGovernor', src='Kshaft', indexer=self.rego,
                                 tex_name='K_{shaft}',
                                 )

        self.wr0 = ExtAlgeb(model='RenGovernor', src='wr0', indexer=self.rego,
                            tex_name=r'\omega_{r0}', export=False,
                            info='Retrieved initial w0 from RenGovernor',
                            v_str='fPe_y',
                            e_str='-w0 + fPe_y',  # update `wr0` of RenGovernor
                            v_setter=True,
                            ename='dwr',
                            tex_ename=r'\Delta \omega_r',
                            )

        # `s2_y` is `wref` for WTPTA1
        self.s2 = Lag(u=self.fPe_y, T=self.Twref, K=1.0,
                      tex_name='s_2', info='speed filter',
                      )

        self.SWT = Switcher(u=self.Tflag, options=(0, 1),
                            tex_name='SW_{T}',
                            cache=True,
                            )

        self.Tsel = Algeb(tex_name='T_{sel}',
                          info='Output after Tflag selector',
                          discrete=self.SWT
                          )
        self.Tsel.v_str = 'SWT_s1 * (Pe - Pref0) / wg +' \
                          'SWT_s0 * (s2_y - wg)'
        self.Tsel.e_str = f'{self.Tsel.v_str} - Tsel'

        self.PI = PIAWHardLimit(u=self.Tsel, kp=self.Kpp, ki=self.Kip,
                                aw_lower=self.Temin, aw_upper=self.Temax,
                                lower=self.Temin, upper=self.Temax,
                                tex_name='PI',
                                info='PI controller',
                                x0='Pref0 / fPe_y',
                                )

        # Note:
        #   Reset `wg` of REECA1 to 1.0 becase `wg` has already been multiplied
        #   in the toeque model.
        #   This effectively sets `PFLAG` to 0 if the torque model is connected.

        self.wge = ExtAlgeb(model='RenExciter', src='wg', indexer=self.ree,
                            tex_name=r'\omega_{ge}', export=False,
                            v_str='1.0',
                            e_str='-fPe_y + 1',
                            v_setter=True,
                            ename='dwg',
                            tex_ename=r'\Delta \omega_g',
                            )

        self.Pref0 = ExtService(model='RenExciter', src='p0', indexer=self.ree,
                                tex_name='P_{ref0}',
                                )

        self.Pref = ExtAlgeb(model='RenExciter', src='Pref', indexer=self.ree,
                             tex_name='P_{ref}', export=False,
                             e_str='-Pref0 / wge + PI_y * wg',
                             v_str='PI_y * wg',
                             v_setter=True,
                             ename='Pref',
                             tex_ename='P_{ref}',
                             )


class WTTQA1(WTTQA1Data, WTTQA1Model):
    """
    Wind turbine generator torque (Pref) model.

    PI state freeze following voltage dip has not been implemented.

    Resets `wg` in `REECA1` model to 1.0 when torque model is connected.
    This effectively ignores `PFLAG` of `REECA1`.
    """

    def __init__(self, config, system):
        WTTQA1Data.__init__(self)
        WTTQA1Model.__init__(self, system, config)
