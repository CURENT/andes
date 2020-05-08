from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.var import Algeb, State, ExtState, ExtAlgeb
from andes.core.service import ConstService, ExtService
from andes.core.discrete import HardLimiter, DeadBand, AntiWindup
from andes.core.block import LeadLag, LagAntiWindup, IntegratorAntiWindup, Lag


class TGBaseData(ModelData):
    def __init__(self):
        super().__init__()
        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            unique=True,
                            )
        self.wref0 = NumParam(info='Base speed reference',
                              tex_name=r'\omega_{ref0}',
                              default=1.0,
                              unit='p.u.',
                              )


class TGBase(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'TurbineGov'
        self.flags.update({'tds': True})
        self.Sn = ExtParam(src='Sn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='S_n',
                           info='Rated power from generator',
                           unit='MVA',
                           export=False,
                           )
        self.Vn = ExtParam(src='Vn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='V_n',
                           info='Rated voltage from generator',
                           unit='kV',
                           export=False,
                           )

        # Note: changing `tm0` is not allowed in any time!!
        self.tm0 = ExtService(src='tm',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name=r'\tau_{m0}',
                              info='Initial mechanical input')
        self.omega = ExtState(src='omega',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name=r'\omega',
                              info='Generator speed',
                              unit='p.u.'
                              )

        # Note: changing `paux0` is allowed.
        # It is a way how one can input from external programs such as reinforcement learning.
        self.paux0 = ConstService(v_str='0',
                                  tex_name='P_{aux0}',
                                  info='const. auxiliary input')

        self.tm = ExtAlgeb(src='tm',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'\tau_m',
                           e_str='u * (pout - tm0)',
                           info='Mechanical power to generator',
                           )
        self.pout = Algeb(info='Turbine final output power',
                          tex_name='P_{out}',
                          v_str='u*tm0',
                          )
        self.wref = Algeb(info='Speed reference variable',
                          tex_name=r'\omega_{ref}',
                          v_str='wref0',
                          e_str='wref0 - wref',
                          )


class TG2Data(TGBaseData):
    def __init__(self):
        super().__init__()
        self.R = NumParam(info='Speed regulation gain (mach. base default)',
                          tex_name='R',
                          default=0.05,
                          unit='p.u.',
                          ipower=True,
                          )
        self.pmax = NumParam(info='Maximum power output',
                             tex_name='p_{max}',
                             power=True,
                             default=999.0,
                             unit='p.u.',
                             )
        self.pmin = NumParam(info='Minimum power output',
                             tex_name='p_{min}',
                             power=True,
                             default=0.0,
                             unit='p.u.',
                             )
        self.dbl = NumParam(info='Deadband lower limit',
                            tex_name='L_{db}',
                            default=-0.0001,
                            unit='p.u.',
                            )
        self.dbu = NumParam(info='Deadband upper limit',
                            tex_name='U_{db}',
                            default=0.0001,
                            unit='p.u.',
                            )
        self.dbc = NumParam(info='Deadband neutral value',
                            tex_name='C_{db}',
                            default=0.0,
                            unit='p.u.',
                            )

        self.T1 = NumParam(info='Transient gain time',
                           default=0.2,
                           tex_name='T_1',
                           )
        self.T2 = NumParam(info='Governor time constant',
                           default=10.0,
                           tex_name='T_2',
                           )


class TG2(TG2Data, TGBase):
    def __init__(self, system, config):
        TG2Data.__init__(self)
        TGBase.__init__(self, system, config)
        self.config.add({'deadband': 0,
                         'hardlimit': 1})
        self.config.add_extra("_help",
                              deadband="enable input dead band",
                              hardlimit="enable output hard limit"
                              )
        self.config.add_extra("_alt",
                              deadband=(0, 1),
                              hardlimit=(0, 1),
                              )
        self.config.add_extra("_tex",
                              deadband="z_{deadband}",
                              hardlimit="z_{hardlimit}",
                              )

        self.gain = ConstService(v_str='u / R',
                                 tex_name='G',
                                 )

        self.w_d = Algeb(info='Generator speed deviation before dead band (positive for under speed)',
                         tex_name=r'\omega_{dev}',
                         v_str='0',
                         e_str='u*(wref-omega) - w_d',
                         )
        self.w_db = DeadBand(u=self.w_d,
                             center=self.dbc,
                             lower=self.dbl,
                             upper=self.dbu,
                             enable=self.config.deadband,
                             )
        self.w_dm = Algeb(info='Measured speed deviation after dead band',
                          tex_name=r'\omega_{dm}',
                          v_str='0',
                          e_str='(1 - w_db_zi) * w_d + '
                                'w_db_zlr * dbl + '
                                'w_db_zur * dbu - '
                                'w_dm')

        self.w_dmg = Algeb(info='Speed deviation after dead band after gain',
                           tex_name=r'\omega_{dmG}',
                           v_str='0',
                           e_str='gain * w_dm - w_dmg',
                           )
        self.ll = LeadLag(u=self.w_dmg,
                          T1=self.T1,
                          T2=self.T2,
                          )

        self.pnl = Algeb(info='Power output before hard limiter',
                         tex_name='P_{nl}',
                         v_str='tm0',
                         e_str='tm0 + ll_y - pnl',
                         )
        self.plim = HardLimiter(u=self.pnl,
                                lower=self.pmin,
                                upper=self.pmax,
                                enable=self.config.hardlimit,
                                )

        self.pout.e_str = 'pnl * plim_zi + pmax * plim_zu + pmin * plim_zl - pout'


class TGOV1Data(TGBaseData):
    def __init__(self):
        super().__init__()
        self.R = NumParam(info='Speed regulation gain (mach. base default)',
                          tex_name='R',
                          default=0.05,
                          unit='p.u.',
                          ipower=True,
                          )
        self.VMAX = NumParam(info='Maximum valve position',
                             tex_name='V_{max}',
                             unit='p.u.',
                             default=1.2,
                             power=True,
                             )
        self.VMIN = NumParam(info='Minimum valve position',
                             tex_name='V_{min}',
                             unit='p.u.',
                             default=0.0,
                             power=True,
                             )

        self.T1 = NumParam(info='Valve time constant',
                           default=0.1,
                           tex_name='T_1')
        self.T2 = NumParam(info='Lead-lag lead time constant',
                           default=0.2,
                           tex_name='T_2')
        self.T3 = NumParam(info='Lead-lag lag time constant',
                           default=10.0,
                           tex_name='T_3')
        self.Dt = NumParam(info='Turbine damping coefficient',
                           default=0.0,
                           tex_name='D_t',
                           power=True,
                           )


class TGOV1Model(TGBase):
    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        self.gain = ConstService(v_str='u/R',
                                 tex_name='G',
                                 )

        self.pref = Algeb(info='Reference power input',
                          tex_name='P_{ref}',
                          v_str='tm0 * R',
                          e_str='tm0 * R - pref',
                          )

        # `paux` must be zero upon initialization
        self.paux = Algeb(info='Auxiliary power input',
                          tex_name='P_{aux}',
                          v_str='paux0',
                          e_str='paux0 - paux',
                          )

        self.wd = Algeb(info='Generator under speed',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        v_str='0',
                        e_str='(wref - omega) - wd',
                        )
        self.pd = Algeb(info='Pref plus under speed times gain',
                        unit='p.u.',
                        tex_name="P_d",
                        v_str='u * tm0',
                        e_str='u*(wd + pref + paux) * gain - pd')

        self.LAG = LagAntiWindup(u=self.pd,
                                 K=1,
                                 T=self.T1,
                                 lower=self.VMIN,
                                 upper=self.VMAX,
                                 )
        self.LL = LeadLag(u=self.LAG_x,
                          T1=self.T2,
                          T2=self.T3,
                          )
        self.pout.e_str = '(LL_y + Dt * wd) - pout'


class TGOV1ModelAlt(TGBase):
    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        self.pref = Algeb(info='Reference power input',
                          tex_name='P_{ref}',
                          v_str='tm0 * R',
                          e_str='tm0 * R - pref',
                          )
        self.wd = Algeb(info='Generator under speed',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        v_str='0',
                        e_str='u * (wref - omega) - wd',
                        )
        self.pd = Algeb(info='Pref plus under speed times gain',
                        unit='p.u.',
                        tex_name="P_d",
                        v_str='tm0',
                        e_str='(wd + pref) * gain - pd')

        self.LAG_x = State(info='State in lag transfer function',
                           tex_name=r"x'_{LAG}",
                           e_str='LAG_lim_zi * (1 * pd - LAG_x)',
                           t_const=self.T1,
                           v_str='pd',
                           )
        self.LAG_lim = AntiWindup(u=self.LAG_x,
                                  lower=self.VMIN,
                                  upper=self.VMAX,
                                  tex_name='lim_{lag}',
                                  )
        self.LL_x = State(info='State in lead-lag transfer function',
                          tex_name="x'_{LL}",
                          v_str='LAG_x',
                          e_str='(LAG_x - LL_x)',
                          t_const=self.T3
                          )
        self.LL_y = Algeb(info='Lead-lag Output',
                          tex_name='y_{LL}',
                          v_str='LAG_x',
                          e_str='T2 / T3 * (LAG_x - LL_x) + LL_x - LL_y',
                          )

        self.pout.e_str = '(LL_y + Dt * wd) - pout'


class TGOV1(TGOV1Data, TGOV1Model):
    """
    TGOV1 model.
    """
    def __init__(self, system, config):
        TGOV1Data.__init__(self)
        TGOV1Model.__init__(self, system, config)

# Developing a model (use TG2 as an example)
# 0) Find the group class or write a new group class in group.py
# 1) Determine and write the class `TG2Data` derived from ModelData
# 2) Write an empty class for the Model by inheriting `TG2Data` and `Model`
# 3) Implement the `__init__` function
#    a) Call parent class `__init__` methods.
#    b) Set `self.flags` for `pflow` and `tds` if applicable.
#    c) Define external service, algeb and states
#    d) Define base class variables and equations
# 4) Implement the TG2 class variables `pout` and add a limiter


class IEEEG1Data(TGBaseData):

    def __init__(self):
        TGBaseData.__init__(self)
        self.K = NumParam(default=20, tex_name='K',
                          info='Gain (1/R) in mach. base',
                          unit='p.u. (power)', power=True, vrange=(5, 30),
                          )
        self.T1 = NumParam(default=1, tex_name='T_1',
                           info='Gov. lag time const.',
                           vrange=(0, 5),
                           )
        self.T2 = NumParam(default=1, tex_name='T_2',
                           info='Gov. lead time const.',
                           vrange=(0, 10),
                           )
        self.T3 = NumParam(default=0.1, tex_name='T_3',
                           info='Valve controller time const.',
                           vrange=(0.04, 1),
                           )
        # "UO" is "U" and capitalized "o" character
        self.UO = NumParam(default=0.1, tex_name='U_o',
                           info='Max. valve opening rate',
                           unit='p.u./sec', vrange=(0.01, 0.3),
                           )
        self.UC = NumParam(default=-0.1, tex_name='U_c',
                           info='Max. valve closing rate',
                           unit='p.u./sec', vrange=(-0.3, 0),
                           )
        self.PMAX = NumParam(default=5, tex_name='P_{MAX}',
                             info='Max. turbine power',
                             vrange=(0.5, 2), power=True,
                             )
        self.PMIN = NumParam(default=0., tex_name='P_{MIN}',
                             info='Min. turbine power',
                             vrange=(0.0, 0.5), power=True,
                             )

        self.T4 = NumParam(default=0.4, tex_name='T_4',
                           info='Inlet piping/steam bowl time constant',
                           vrange=(0, 1.0),
                           )
        self.K1 = NumParam(default=0.3, tex_name='K_1',
                           info='Fraction of power from HP',
                           vrange=(0, 1.0),
                           )
        self.K2 = NumParam(default=0, tex_name='K_2',
                           info='Fraction of power from LP',
                           vrange=(0,),
                           )
        self.T5 = NumParam(default=8, tex_name='T_5',
                           info='Time constant of 2nd boiler pass',
                           vrange=(0, 10),
                           )
        self.K3 = NumParam(default=0.4, tex_name='K_3',
                           info='Fraction of HP shaft power after 2nd boiler pass',
                           vrange=(0, 0.5),
                           )
        self.K4 = NumParam(default=0, tex_name='K_4',
                           info='Fraction of LP shaft power after 2nd boiler pass',
                           vrange=(0,),
                           )

        self.T6 = NumParam(default=0.5, tex_name='T_6',
                           info='Time constant of 3rd boiler pass',
                           vrange=(0, 10),
                           )
        self.K5 = NumParam(default=0.3, tex_name='K_5',
                           info='Fraction of HP shaft power after 3rd boiler pass',
                           vrange=(0, 0.35),
                           )
        self.K6 = NumParam(default=0, tex_name='K_6',
                           info='Fraction of LP shaft power after 3rd boiler pass',
                           vrange=(0, 0.55),
                           )

        self.T7 = NumParam(default=0.05, tex_name='T_7',
                           info='Time constant of 4th boiler pass',
                           vrange=(0, 10),
                           )
        self.K7 = NumParam(default=0, tex_name='K_7',
                           info='Fraction of HP shaft power after 4th boiler pass',
                           vrange=(0, 0.3),
                           )
        self.K8 = NumParam(default=0, tex_name='K_8',
                           info='Fraction of LP shaft power after 4th boiler pass',
                           vrange=(0, 0.3),
                           )


class IEEEG1Model(TGBase):
    def __init__(self, system, config):
        TGBase.__init__(self, system, config)

        # self.syn2 = IdxParam(model='SynGen',
        #                      info='Optional SynGen idx',
        #                      )
        #
        # self.tm02 = ExtService(src='tm',
        #                        model='SynGen',
        #                        indexer=self.syn2,
        #                        tex_name=r'\tau_{m02}',
        #                        info='Initial mechanical input of syn2')
        #
        # self.tm2 = ExtAlgeb(src='tm',
        #                     model='SynGen',
        #                     indexer=self.syn2,
        #                     tex_name=r'\tau_{m2}',
        #                     # e_str='u * (pout - tm02)',
        #                     info='Mechanical power to syn2',
        #                     )

        self.wd = Algeb(info='Generator under speed',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        v_str='0',
                        e_str='(wref - omega) - wd',
                        )

        self.LL = LeadLag(u=self.wd,
                          T1=self.T2,
                          T2=self.T1,
                          K=self.K,
                          )

        # `P0` == `tm0`
        self.vs = Algeb(info='Valve move speed',
                        v_str='0',
                        e_str='(LL_y + tm0 - IAW_y) / T3 - vs',
                        )

        self.HL = HardLimiter(u=self.vs,
                              lower=self.UC,
                              upper=self.UO,
                              info='Limiter on valve acceleration',
                              )

        self.vsl = Algeb(info='Valve move speed after limiter',
                         v_str='vs * HL_zi + UC * HL_zl + UO * HL_zu',
                         e_str='vs * HL_zi + UC * HL_zl + UO * HL_zu - vsl',
                         )

        self.IAW = IntegratorAntiWindup(u=self.vsl,
                                        K=1,
                                        y0=self.tm0,
                                        lower=self.PMIN,
                                        upper=self.PMAX,
                                        info='Valve position integrator',
                                        )

        self.L4 = Lag(u=self.IAW_y, T=self.T4, K=1)

        self.L5 = Lag(u=self.L4_x, T=self.T5, K=1)

        self.L6 = Lag(u=self.L5_x, T=self.T6, K=1)

        self.L7 = Lag(u=self.L6_x, T=self.T7, K=1)

        self.PHP = Algeb(info='HP output',
                         v_str='K1*L4_x + K3*L5_x + K5*L6_x + K7*L7_x',
                         e_str='K1*L4_x + K3*L5_x + K5*L6_x + K7*L7_x - PHP',
                         )

        self.PLP = Algeb(info='LP output',
                         v_str='K2*L4_x + K4*L5_x + K6*L6_x + K8*L7_x',
                         e_str='K2*L4_x + K4*L5_x + K6*L6_x + K8*L7_x - PLP',
                         )

        self.pout.e_str = 'PHP - pout'


class IEEEG1(IEEEG1Data, IEEEG1Model):
    """
    IEEE Type 1 Speed-Governing Model

    TODO: allow connecting to the second generator

    Notes from PowerWorld documentation:

    https://www.powerworld.com/WebHelp/Content/TransientModels_PDF/Generator/Governor/Governor%20IEEEG1%20and%20IEEEG1_GE.pdf

    ::

        For the IEEEG1 model, if the turbine rating is omitted
        then the MVABase of only the high-pressure generator is used.

    Notes from NEPLAN manual:

    https://www.neplan.ch/wp-content/uploads/2015/08/Nep_TURBINES_GOV.pdf

    ::

        For a tandem-compound turbine the parameters K2, K4, K6,
        and K8 are ignored. For a cross- compound turbine,
        two generators are connected to this turbine-governor model.

        Each generator must be represented in the load flow by data
        on its own MVA base. The values of K1, K3, K5, K7
        must be specified to describe the proportionate
        development of power on the first turbine shaft.
        K2, K4, K6, K8 must describe the second turbine shaft.
        Normally K1 + K3 + K5 + K7 = 1.0 and K2 + K4 + K6 + K8 = 1.0
        (if second generator is present).
    """

    def __init__(self, system, config):
        IEEEG1Data.__init__(self)
        IEEEG1Model.__init__(self, system, config)
