from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.var import Algeb, State, ExtState, ExtAlgeb
from andes.core.service import ConstService, ExtService, NumSelect, FlagValue, ParamCalc, InitChecker
from andes.core.service import PostInitService
from andes.core.discrete import HardLimiter, DeadBandRT, AntiWindup
from andes.core.block import LeadLag, LagAntiWindup, IntegratorAntiWindup, Lag, DeadBand1
import numpy as np


class TGBaseData(ModelData):
    """
    Base data for turbine governors.
    """
    def __init__(self):
        super().__init__()
        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            unique=True,
                            )
        self.Tn = NumParam(info='Turbine power rating. Equal to Sn if not provided.',
                           tex_name='T_n',
                           unit='MVA',
                           default=None,
                           )
        self.wref0 = NumParam(info='Base speed reference',
                              tex_name=r'\omega_{ref0}',
                              default=1.0,
                              unit='p.u.',
                              )


class TGBase(Model):
    """
    Base Turbine Governor model.

    Parameters
    ----------
    add_sn : bool
        True to add ``NumSelect`` Sn; False to add later in custom models.
        This is useful when the governor connects to two generators.
    add_tm0 : bool
        True to add ``ExtService`` ``tm0``.

    """
    def __init__(self, system, config, add_sn=True, add_tm0=True):
        Model.__init__(self, system, config)
        self.group = 'TurbineGov'
        self.flags.update({'tds': True})
        self.Sg = ExtParam(src='Sn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='S_n',
                           info='Rated power from generator',
                           unit='MVA',
                           export=False,
                           )
        if add_sn is True:
            self.Sn = NumSelect(self.Tn,
                                fallback=self.Sg,
                                tex_name='S_n',
                                info='Turbine or Gen rating',
                                )

        self.Vn = ExtParam(src='Vn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='V_n',
                           info='Rated voltage from generator',
                           unit='kV',
                           export=False,
                           )

        # Note: changing the values of `tm0` is not allowed at any time!!
        if add_tm0 is True:
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
                           info='Mechanical power interface to SynGen',
                           )
        # `paux` must be zero upon initialization
        self.paux = Algeb(info='Auxiliary power input',
                          tex_name='P_{aux}',
                          v_str='paux0',
                          e_str='paux0 - paux',
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
        self.w_db = DeadBandRT(u=self.w_d,
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


class TGOV1DBData(TGOV1Data):
    def __init__(self):
        TGOV1Data.__init__(self)
        self.dbL = NumParam(info='Lower bound of deadband',
                            tex_name='db_L',
                            default=0.0,
                            unit='p.u.',
                            )
        self.dbU = NumParam(info='Upper bound of deadband',
                            tex_name='db_U',
                            default=0.0,
                            unit='p.u.',
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
        self.LL = LeadLag(u=self.LAG_y,
                          T1=self.T2,
                          T2=self.T3,
                          )
        self.pout.e_str = '(LL_y + Dt * wd) - pout'


class TGOV1DBModel(TGOV1Model):
    def __init__(self, system, config):
        TGOV1Model.__init__(self, system, config)
        self.DB = DeadBand1(u=self.wd, center=0.0, lower=self.dbL,
                            upper=self.dbU, tex_name='DB',
                            info='deadband for under speed',
                            )
        self.pd.e_str = 'u * (DB_y + pref + paux) * gain - pd'
        self.pout.e_str = '(LL_y + Dt * DB_y) - pout'


class TGOV1ModelAlt(TGBase):
    """
    An alternative implementation of TGOV1 from equations
    (without using Blocks).
    """
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

        self.LAG_y = State(info='State in lag transfer function',
                           tex_name=r"x'_{LAG}",
                           e_str='LAG_lim_zi * (1 * pd - LAG_y)',
                           t_const=self.T1,
                           v_str='pd',
                           )
        self.LAG_lim = AntiWindup(u=self.LAG_y,
                                  lower=self.VMIN,
                                  upper=self.VMAX,
                                  tex_name='lim_{lag}',
                                  )
        self.LL_x = State(info='State in lead-lag transfer function',
                          tex_name="x'_{LL}",
                          v_str='LAG_y',
                          e_str='(LAG_y - LL_x)',
                          t_const=self.T3
                          )
        self.LL_y = Algeb(info='Lead-lag Output',
                          tex_name='y_{LL}',
                          v_str='LAG_y',
                          e_str='T2 / T3 * (LAG_y - LL_x) + LL_x - LL_y',
                          )

        self.pout.e_str = '(LL_y + Dt * wd) - pout'


class TGOV1(TGOV1Data, TGOV1Model):
    """
    TGOV1 turbine governor model.

    Implements the PSS/E TGOV1 model without deadband.
    """
    def __init__(self, system, config):
        TGOV1Data.__init__(self)
        TGOV1Model.__init__(self, system, config)


class TGOV1DB(TGOV1DBData, TGOV1DBModel):
    """
    TGOV1 turbine governor model with speed input deadband.
    """
    def __init__(self, system, config):
        TGOV1DBData.__init__(self)
        TGOV1DBModel.__init__(self, system, config)


class IEEEG1Data(TGBaseData):

    def __init__(self):
        TGBaseData.__init__(self)

        self.syn2 = IdxParam(model='SynGen',
                             info='Optional SynGen idx',
                             )
        self.K = NumParam(default=20, tex_name='K',
                          info='Gain (1/R) in mach. base',
                          unit='p.u. (power)',
                          power=True,
                          vrange=(5, 30),
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
        self.K1 = NumParam(default=0.5, tex_name='K_1',
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
        self.K3 = NumParam(default=0.5, tex_name='K_3',
                           info='Fraction of HP shaft power after 2nd boiler pass',
                           vrange=(0, 0.5),
                           )
        self.K4 = NumParam(default=0.0, tex_name='K_4',
                           info='Fraction of LP shaft power after 2nd boiler pass',
                           vrange=(0,),
                           )

        self.T6 = NumParam(default=0.5, tex_name='T_6',
                           info='Time constant of 3rd boiler pass',
                           vrange=(0, 10),
                           )
        self.K5 = NumParam(default=0.0, tex_name='K_5',
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
        TGBase.__init__(self, system, config, add_sn=False)

        # check if K1-K8 sums up to 1
        self._sumK18 = ConstService(v_str='K1+K2+K3+K4+K5+K6+K7+K8',
                                    info='summation of K1-K8',
                                    tex_name=r"\sum_{i=1}^8 K_i"
                                    )
        self._K18c1 = InitChecker(u=self._sumK18,
                                  info='summation of K1-K8 and 1.0',
                                  equal=1,
                                  )

        # check if  `tm0 * (K2 + k4 + K6 + K8) = tm02 *(K1 + K3 + K5 + K7)
        self._tm0K2 = PostInitService(info='mul of tm0 and (K2+K4+K6+K8)',
                                      v_str='zsyn2*tm0*(K2+K4+K6+K8)',
                                      )
        self._tm02K1 = PostInitService(info='mul of tm02 and (K1+K3+K5+K6)',
                                       v_str='tm02*(K1+K3+K5+K7)',
                                       )
        self._Pc = InitChecker(u=self._tm0K2,
                               info='proportionality of tm0 and tm02',
                               equal=self._tm02K1,
                               )

        self.Sg2 = ExtParam(src='Sn',
                            model='SynGen',
                            indexer=self.syn2,
                            allow_none=True,
                            default=0.0,
                            tex_name='S_{n2}',
                            info='Rated power of Syn2',
                            unit='MVA',
                            export=False,
                            )
        self.Sg12 = ParamCalc(self.Sg, self.Sg2, func=np.add,
                              tex_name="S_{g12}",
                              info='Sum of generator power ratings',
                              )
        self.Sn = NumSelect(self.Tn,
                            fallback=self.Sg12,
                            tex_name='S_n',
                            info='Turbine or Gen rating',
                            )

        self.zsyn2 = FlagValue(self.syn2,
                               value=None,
                               tex_name='z_{syn2}',
                               info='Exist flags for syn2',
                               )

        self.tm02 = ExtService(src='tm',
                               model='SynGen',
                               indexer=self.syn2,
                               tex_name=r'\tau_{m02}',
                               info='Initial mechanical input of syn2',
                               allow_none=True,
                               default=0.0,
                               )
        self.tm012 = ConstService(info='total turbine power',
                                  v_str='tm0 + tm02',
                                  )

        self.tm2 = ExtAlgeb(src='tm',
                            model='SynGen',
                            indexer=self.syn2,
                            allow_none=True,
                            tex_name=r'\tau_{m2}',
                            e_str='zsyn2 * u * (PLP - tm02)',
                            info='Mechanical power to syn2',
                            )

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
                          info='Signal conditioning for wd',
                          )

        # `P0` == `tm0`
        self.vs = Algeb(info='Valve speed',
                        tex_name='V_s',
                        v_str='0',
                        e_str='(LL_y + tm012 + paux - IAW_y) / T3 - vs',
                        )

        self.HL = HardLimiter(u=self.vs,
                              lower=self.UC,
                              upper=self.UO,
                              info='Limiter on valve acceleration',
                              )

        self.vsl = Algeb(info='Valve move speed after limiter',
                         tex_name='V_{sl}',
                         v_str='vs * HL_zi + UC * HL_zl + UO * HL_zu',
                         e_str='vs * HL_zi + UC * HL_zl + UO * HL_zu - vsl',
                         )

        self.IAW = IntegratorAntiWindup(u=self.vsl,
                                        T=1,
                                        K=1,
                                        y0=self.tm012,
                                        lower=self.PMIN,
                                        upper=self.PMAX,
                                        info='Valve position integrator',
                                        )

        self.L4 = Lag(u=self.IAW_y, T=self.T4, K=1,
                      info='first process',)

        self.L5 = Lag(u=self.L4_y, T=self.T5, K=1,
                      info='second (reheat) process',
                      )

        self.L6 = Lag(u=self.L5_y, T=self.T6, K=1,
                      info='third process',
                      )

        self.L7 = Lag(u=self.L6_y, T=self.T7, K=1,
                      info='fourth (second reheat) process',
                      )

        self.PHP = Algeb(info='HP output',
                         tex_name='P_{HP}',
                         v_str='K1*L4_y + K3*L5_y + K5*L6_y + K7*L7_y',
                         e_str='K1*L4_y + K3*L5_y + K5*L6_y + K7*L7_y - PHP',
                         )

        self.PLP = Algeb(info='LP output',
                         tex_name='P_{LP}',
                         v_str='K2*L4_y + K4*L5_y + K6*L6_y + K8*L7_y',
                         e_str='K2*L4_y + K4*L5_y + K6*L6_y + K8*L7_y - PLP',
                         )

        self.pout.e_str = 'PHP - pout'


class IEEEG1(IEEEG1Data, IEEEG1Model):
    """
    IEEE Type 1 Speed-Governing Model.

    If only one generator is connected, its `idx` must
    be given to `syn`, and `syn2` must be left blank.
    Each generator must provide data in its `Sn` base.

    `syn` is connected to the high-pressure output (PHP)
    and the optional `syn2` is connected to the low-
    pressure output (PLP).

    The speed deviation of generator 1 (syn) is measured.
    If the turbine rating `Tn` is not specified, the sum
    of `Sn` of all connected generators will be used.

    Normally, K1 + K2 + ... + K8 = 1.0.
    If the second generator is not connected,
    K1 + K3 + K5 + K7 = 1, and K2 + K4 + K6 + K8 = 0.
    """

    def __init__(self, system, config):
        IEEEG1Data.__init__(self)
        IEEEG1Model.__init__(self, system, config)
