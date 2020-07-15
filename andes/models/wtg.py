from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.block import Piecewise, Lag, GainLimiter, LagAntiWindupRate
from andes.core.var import ExtAlgeb, Algeb
from andes.core.service import ConstService, FlagValue, ExtService


class REGCAU1Data(ModelData):
    """
    REGC_A model data.
    """
    def __init__(self):
        ModelData.__init__(self)

        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            mandatory=True,
                            )
        self.Tg = NumParam(default=0.1, tex_name='T_g',
                           info='converter time const.', unit='s',
                           )
        self.Rrpwr = NumParam(default=999, tex_name='R_{rpwr}',
                              info='Low voltage power logic (LVPL) ramp limit',
                              unit='p.u.',
                              )
        self.Brkpt = NumParam(default=1.0, tex_name='B_{rkpt}',
                              info='LVPL characteristic voltage 2',
                              unit='p.u.',
                              )
        self.Zerox = NumParam(default=0.5, tex_name='Z_{erox}',
                              info='LVPL characteristic voltage 1',
                              unit='p.u',
                              )
        # TODO: ensure Brkpt > Zerox
        self.Lvpl1 = NumParam(default=1.0, tex_name='L_{vpl1}',
                              info='LVPL gain',
                              unit='p.u',
                              )
        self.Volim = NumParam(default=1.2, tex_name='V_{olim}',
                              info='Voltage lim for high volt. reactive current mgnt.',
                              unit='p.u.',
                              )
        self.Lvpnt1 = NumParam(default=1.0, tex_name='L_{vpnt1}',
                               info='High volt. point for low volt. active current mgnt.',
                               unit='p.u.',
                               )
        self.Lvpnt0 = NumParam(default=0.4, tex_name='L_{vpnt0}',
                               info='Low volt. point for low volt. active current mgnt.',
                               unit='p.u.',
                               )
        # TODO: ensure Lvpnt1 > Lvpnt0
        self.Iolim = NumParam(default=999.0, tex_name='I_{olim}',
                              info='current limit for high volt. reactive current mgnt.',
                              unit='p.u.',
                              )
        self.Tfltr = NumParam(default=0.1, tex_name='T_{fltr}',
                              info='Voltage filter T const for low volt. active current mgnt.',
                              unit='s',
                              )
        self.Khv = NumParam(default=0.7, tex_name='K_{hv}',
                            info='Overvolt. compensation gain in high volt. reactive current mgnt.',
                            )
        self.Iqrmax = NumParam(default=999, tex_name='I_{qrmax}',
                               info='Upper limit on the ROC for reactive current',
                               unit='p.u.',
                               )
        self.Iqrmin = NumParam(default=-999, tex_name='I_{qrmin}',
                               info='Lower limit on the ROC for reactive current',
                               unit='p.u.',
                               )
        self.Accel = NumParam(default=0.0, tex_name='A_{ccel}',
                              info='Acceleration factor',
                              vrange=(0, 1.0),
                              )
        self.Iqmax = NumParam(default=999, tex_name='I_{qmax}',
                              info='Upper limit for reactive current',
                              unit='p.u.',
                              )
        self.Iqmin = NumParam(default=-999, tex_name='I_{qmin}',
                              info='Lower limit for reactive current',
                              unit='p.u.',
                              )


class REGCAU1Model(Model):
    """
    REGCAU1 implementation.
    """
    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle',
                          e_str='Ipout * v',
                          )

        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          e_str='Iqout_y * v',
                          )

        self.p0 = ExtService(model='StaticGen',
                             src='p',
                             indexer=self.gen,
                             tex_name='P_0',
                             )
        self.q0 = ExtService(model='StaticGen',
                             src='q',
                             indexer=self.gen,
                             tex_name='Q_0',
                             )
        self.ra = ExtParam(model='StaticGen',
                           src='ra',
                           indexer=self.gen,
                           tex_name='r_a',
                           )
        self.xs = ExtParam(model='StaticGen',
                           src='xs',
                           indexer=self.gen,
                           tex_name='x_s',
                           )

        # --- INITIALIZATION ---
        self.Ipcmd0 = ConstService('p0 / v', info='initial Ipcmd')

        self.Iqcmd0 = ConstService('-q0 / v', info='initial Iqcmd')

        self.Ipcmd = Algeb(tex_name='I_{pcmd}',
                           info='current component for active power',
                           e_str='Ipcmd0 - Ipcmd', v_str='Ipcmd0')

        self.Iqcmd = Algeb(tex_name='I_{qcmd}',
                           info='current component for reactive power',
                           e_str='Iqcmd0 - Iqcmd', v_str='Iqcmd0')

        # reactive power management

        # TODO: create conditions for rate limiting.
        #   In a fault recovery, activate upper limit when Qg0 > 0
        #                        activate lower limit when Qg0 < 0

        self.S1 = LagAntiWindupRate(u=self.Iqcmd, T=self.Tg, K=-1,
                                    lower=self.Iqmin, upper=self.Iqmax,
                                    rate_lower=self.Iqrmin, rate_upper=self.Iqrmax,
                                    # rate_lower_cond, rate_upper_cond,
                                    tex_name='S_1',
                                    info='Iqcmd delay',
                                    )  # output `S1_y` == `Iq`

        # piece-wise gain for low voltage reactive current mgnt.
        self.kLVG = ConstService(v_str='1 / (Lvpnt1 - Lvpnt0)',
                                 tex_name='k_{LVG}',
                                 )

        self.LVG = Piecewise(u=self.v, points=('Lvpnt0', 'Lvpnt1'),
                             funs=('0', '(v - Lvpnt0) * kLVG', '1'),
                             info='Low voltage current gain',
                             )

        self.Lvplsw = FlagValue(u=self.Lvpl1, value=0, flag=0, tex_name='z_{Lvplsw}',
                                info='LVPL enable switch',
                                )
        # piece-wise gain for LVPL
        self.kLVPL = ConstService(v_str='Lvplsw * Lvpl1 / (Brkpt - Zerox)',
                                  tex_name='k_{LVPL}',
                                  )

        self.S2 = Lag(u=self.v, T=self.Tfltr, K=1.0,
                      info='Voltage filter with no anti-windup',
                      )
        self.LVPL = Piecewise(u=self.S2_y,
                              points=('Zerox', 'Brkpt'),
                              funs=('0 + 9999*(1-Lvplsw)',
                                    '(S2_y - Zerox) * kLVPL + 9999 * (1-Lvplsw)',
                                    '9999'),
                              info='Low voltage Ipcmd upper limit',
                              )

        self.S0 = LagAntiWindupRate(u=self.Ipcmd, T=self.Tg, K=1,
                                    upper=self.LVPL_y, rate_upper=self.Rrpwr,
                                    lower=-999, rate_lower=-999,
                                    no_lower=True, rate_no_lower=True,
                                    )  # `S0_y` is the output `Ip` in the block diagram

        self.Ipout = Algeb(e_str='S0_y * LVG_y -Ipout',
                           v_str='Ipcmd * LVG_y',
                           info='Output Ip current',
                           )

        # high voltage part
        self.HVG = GainLimiter(u='v - Volim', K=self.Khv, info='High voltage gain block',
                               lower=0, upper=999, no_upper=True)

        self.Iqout = GainLimiter(u='S1_y- HVG_y', K=1, lower=self.Iolim, upper=9999,
                                 no_upper=True, info='Iq output block')  # `Iqout_y` is the final Iq output

    def v_numeric(self, **kwargs):
        """
        Disable the corresponding `StaticGen`s.
        """
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class REGCAU1(REGCAU1Data, REGCAU1Model):
    def __init__(self, system, config):
        REGCAU1Data.__init__(self)
        REGCAU1Model.__init__(self, system, config)
