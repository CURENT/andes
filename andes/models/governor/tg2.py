"""
Turbine governor type 2 from Milano's book.
"""

from andes.core import (Algeb, ConstService, DeadBandRT, HardLimiter, LeadLag,
                        NumParam,)
from andes.models.governor.tgbase import TGBase, TGBaseData


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
                         e_str='ue*(wref-omega) - w_d',
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
                         e_str='pref0 + ll_y - pnl',
                         )
        self.plim = HardLimiter(u=self.pnl,
                                lower=self.pmin,
                                upper=self.pmax,
                                enable=self.config.hardlimit,
                                )

        self.pout.e_str = 'pnl * plim_zi + pmax * plim_zu + pmin * plim_zl - pout'
