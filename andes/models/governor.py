from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.var import Algeb, State, ExtState, ExtAlgeb
from andes.core.service import ConstService, ExtService
from andes.core.discrete import HardLimiter, DeadBand, AntiWindupLimiter
from andes.core.block import LeadLag, LagAntiWindup


class TGBaseData(ModelData):
    def __init__(self):
        super().__init__()
        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            )
        self.R = NumParam(info='Speed regulation gain under machine base',
                          tex_name='R',
                          default=0.05,
                          unit='p.u.',
                          ipower=True,
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

        self.gain = ConstService(v_str='u / R',
                                 tex_name='G',
                                 )

        self.tm = ExtAlgeb(src='tm',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'\tau_m',
                           e_str='u * (pout - tm0)',
                           info='Mechanical power to generator',
                           )
        self.pout = Algeb(info='Turbine final output power',
                          tex_name='P_{out}',
                          v_str='tm0',
                          )
        self.wref = Algeb(info='Speed reference variable',
                          tex_name=r'\omega_{ref}',
                          v_str='wref0',
                          e_str='wref0 - wref',
                          )


class TG2Data(TGBaseData):
    def __init__(self):
        super().__init__()
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

        self.w_d = Algeb(info='Generator speed deviation before dead band (positive for under speed)',
                         tex_name=r'\omega_{dev}',
                         v_str='0',
                         e_str='u * (wref - omega) - w_d',
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
                        v_str='tm0',
                        e_str='(wd + pref) * gain - pd')

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
        self.LAG_lim = AntiWindupLimiter(u=self.LAG_x,
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


class TGOV1(TGOV1Data, TGOV1ModelAlt):
    """
    TGOV1 model.
    """
    def __init__(self, system, config):
        TGOV1Data.__init__(self)
        TGOV1ModelAlt.__init__(self, system, config)

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
