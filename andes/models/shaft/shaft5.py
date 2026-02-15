"""
5-mass torsional shaft model for synchronous generators.
"""

from andes.core import (ModelData, IdxParam, NumParam, Model, State,
                        ExtAlgeb, ExtParam, ExtService, ExtState,
                        ConstService)


class SHAFT5Data(ModelData):
    """
    Data for SHAFT5 model.
    """

    def __init__(self):
        super().__init__()

        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            unique=True,
                            status_parent=True,
                            )

        # --- Inertia constants (M = 2H) for each mass ---
        self.MHP = NumParam(default=0.092,
                            info='HP mass start-up time (2*H_HP)',
                            tex_name='M_{HP}',
                            unit='MWs/MVA',
                            power=True,
                            non_zero=True,
                            non_negative=True,
                            )
        self.MIP = NumParam(default=0.156,
                            info='IP mass start-up time (2*H_IP)',
                            tex_name='M_{IP}',
                            unit='MWs/MVA',
                            power=True,
                            non_zero=True,
                            non_negative=True,
                            )
        self.MLP = NumParam(default=0.858,
                            info='LP mass start-up time (2*H_LP)',
                            tex_name='M_{LP}',
                            unit='MWs/MVA',
                            power=True,
                            non_zero=True,
                            non_negative=True,
                            )
        self.MEX = NumParam(default=0.068,
                            info='EX mass start-up time (2*H_EX)',
                            tex_name='M_{EX}',
                            unit='MWs/MVA',
                            power=True,
                            non_zero=True,
                            non_negative=True,
                            )

        # --- Spring constants between masses ---
        self.KHP = NumParam(default=19.303,
                            info='HP-IP spring constant',
                            tex_name='K_{HP}',
                            unit='p.u.',
                            power=True,
                            non_zero=True,
                            non_negative=True,
                            )
        self.KIP = NumParam(default=34.929,
                            info='IP-LP spring constant',
                            tex_name='K_{IP}',
                            unit='p.u.',
                            power=True,
                            non_zero=True,
                            non_negative=True,
                            )
        self.KLP = NumParam(default=52.038,
                            info='LP-Rotor spring constant',
                            tex_name='K_{LP}',
                            unit='p.u.',
                            power=True,
                            non_zero=True,
                            non_negative=True,
                            )
        self.KEX = NumParam(default=1.55,
                            info='Rotor-EX spring constant',
                            tex_name='K_{EX}',
                            unit='p.u.',
                            power=True,
                            non_zero=True,
                            non_negative=True,
                            )

        # --- Self-damping coefficients ---
        self.DHP = NumParam(default=0.0,
                            info='HP mass self-damping',
                            tex_name='D_{HP}',
                            unit='p.u.',
                            power=True,
                            )
        self.DIP = NumParam(default=0.0,
                            info='IP mass self-damping',
                            tex_name='D_{IP}',
                            unit='p.u.',
                            power=True,
                            )
        self.DLP = NumParam(default=0.0,
                            info='LP mass self-damping',
                            tex_name='D_{LP}',
                            unit='p.u.',
                            power=True,
                            )
        self.DEX = NumParam(default=0.0,
                            info='EX mass self-damping',
                            tex_name='D_{EX}',
                            unit='p.u.',
                            power=True,
                            )

        # --- Mutual damping coefficients ---
        self.D12 = NumParam(default=0.0,
                            info='HP-IP mutual damping',
                            tex_name='D_{12}',
                            unit='p.u.',
                            power=True,
                            )
        self.D23 = NumParam(default=0.0,
                            info='IP-LP mutual damping',
                            tex_name='D_{23}',
                            unit='p.u.',
                            power=True,
                            )
        self.D34 = NumParam(default=0.0,
                            info='LP-Rotor mutual damping',
                            tex_name='D_{34}',
                            unit='p.u.',
                            power=True,
                            )
        self.D45 = NumParam(default=0.0,
                            info='Rotor-EX mutual damping',
                            tex_name='D_{45}',
                            unit='p.u.',
                            power=True,
                            )


class SHAFT5Model(Model):
    """
    5-mass torsional shaft model implementation.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'DynShaft'
        self.flags.update({'tds': True})

        # --- External parameters from generator ---
        self.Sn = ExtParam(model='SynGen',
                           src='Sn',
                           indexer=self.syn,
                           tex_name='S_n',
                           info='Generator power rating',
                           export=False,
                           )
        self.fn = ExtParam(model='SynGen',
                           src='fn',
                           indexer=self.syn,
                           tex_name='f_n',
                           info='Rated frequency',
                           export=False,
                           )

        # --- External services for initialization ---
        self.tm0 = ExtService(model='SynGen',
                              src='tm',
                              indexer=self.syn,
                              tex_name=r'\tau_{m0}',
                              info='Initial mechanical torque',
                              )
        self.delta0_gen = ExtService(model='SynGen',
                                    src='delta',
                                    indexer=self.syn,
                                    tex_name=r'\delta_0',
                                    info='Initial generator rotor angle',
                                    )

        # --- Computed constants ---
        self.Ob = ConstService(v_str='2 * pi * fn',
                               tex_name=r'\Omega_b',
                               info='Base angular frequency',
                               )

        # --- External variables from generator ---
        # omega: add coupling torques to generator's swing equation.
        # This cancels tm and adds spring/damper coupling from LP and EX.
        self.omega = ExtState(
            src='omega',
            model='SynGen',
            indexer=self.syn,
            tex_name=r'\omega',
            info='Generator rotor speed',
            e_str='ue * (-tm - D34 * (omega - wLP) - D45 * (omega - wEX)'
                  ' + KLP * (dLP - delta) + KEX * (dEX - delta))',
        )

        # delta: read-only for coupling in LP and EX equations
        self.delta = ExtState(
            src='delta',
            model='SynGen',
            indexer=self.syn,
            tex_name=r'\delta',
            info='Generator rotor angle',
        )

        # tm: read generator's mechanical torque value for HP mass
        self.tm = ExtAlgeb(
            src='tm',
            model='SynGen',
            indexer=self.syn,
            tex_name=r'\tau_m',
            info='Mechanical torque from generator',
        )

        # ===== HP Mass =====
        self.dHP = State(
            info='HP turbine angle',
            unit='rad',
            tex_name=r'\delta_{HP}',
            v_str='delta0_gen + tm0 / KLP + tm0 / KIP + tm0 / KHP',
            e_str='ue * Ob * (wHP - 1)',
        )
        self.wHP = State(
            info='HP turbine speed',
            unit='p.u.',
            tex_name=r'\omega_{HP}',
            v_str='1',
            e_str='ue * (tm - DHP * (wHP - 1) - D12 * (wHP - wIP)'
                  ' + KHP * (dIP - dHP))',
            t_const=self.MHP,
        )

        # ===== IP Mass =====
        self.dIP = State(
            info='IP turbine angle',
            unit='rad',
            tex_name=r'\delta_{IP}',
            v_str='delta0_gen + tm0 / KLP + tm0 / KIP',
            e_str='ue * Ob * (wIP - 1)',
        )
        self.wIP = State(
            info='IP turbine speed',
            unit='p.u.',
            tex_name=r'\omega_{IP}',
            v_str='1',
            e_str='ue * (-DIP * (wIP - 1) - D12 * (wIP - wHP)'
                  ' - D23 * (wIP - wLP)'
                  ' + KHP * (dHP - dIP) + KIP * (dLP - dIP))',
            t_const=self.MIP,
        )

        # ===== LP Mass =====
        self.dLP = State(
            info='LP turbine angle',
            unit='rad',
            tex_name=r'\delta_{LP}',
            v_str='delta0_gen + tm0 / KLP',
            e_str='ue * Ob * (wLP - 1)',
        )
        self.wLP = State(
            info='LP turbine speed',
            unit='p.u.',
            tex_name=r'\omega_{LP}',
            v_str='1',
            e_str='ue * (-DLP * (wLP - 1) - D23 * (wLP - wIP)'
                  ' - D34 * (wLP - omega)'
                  ' + KIP * (dIP - dLP) + KLP * (delta - dLP))',
            t_const=self.MLP,
        )

        # ===== EX Mass =====
        self.dEX = State(
            info='Exciter mass angle',
            unit='rad',
            tex_name=r'\delta_{EX}',
            v_str='delta0_gen',
            e_str='ue * Ob * (wEX - 1)',
        )
        self.wEX = State(
            info='Exciter mass speed',
            unit='p.u.',
            tex_name=r'\omega_{EX}',
            v_str='1',
            e_str='ue * (-DEX * (wEX - 1) - D45 * (wEX - omega)'
                  ' + KEX * (delta - dEX))',
            t_const=self.MEX,
        )


class SHAFT5(SHAFT5Data, SHAFT5Model):
    """
    5-mass torsional shaft model (HP-IP-LP-Rotor-EX) for synchronous
    generators.

    The model adds torsional dynamics to a synchronous generator by
    introducing 4 additional mass-spring-damper sections connected
    in series with the generator rotor.

    The 5 masses represent:

    - HP: High-pressure turbine section
    - IP: Intermediate-pressure turbine section
    - LP: Low-pressure turbine section
    - Rotor: Generator rotor (existing SynGen omega/delta)
    - EX: Exciter mass

    The mechanical torque (tm) from the governor drives the HP mass.
    The generator's omega equation is modified to remove tm and add
    shaft coupling torques from LP and EX masses.

    At steady state, all mass speeds equal 1.0 p.u. and the spring
    torques balance the mechanical torque through the chain.

    References
    ----------
    Milano, F. (2010). Power System Modelling and Scripting,
    Section 15.1.10.
    """

    def __init__(self, system, config):
        SHAFT5Data.__init__(self)
        SHAFT5Model.__init__(self, system, config)
