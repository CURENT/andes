"""
Module for renewable energy generator (converter) model A.
"""

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        IdxParam, Lag, Model, ModelData, NumParam, Piecewise,)
from andes.core.block import GainLimiter, LagAntiWindupRate
from andes.core.service import PostInitService
from andes.core.var import AliasAlgeb


class REGCA1Data(ModelData):
    """
    REGC_A model data.

    12/09/2020: Made `Lvplsw` a parameter.
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
        self.Sn = NumParam(default=100.0, tex_name='S_n',
                           info='Model MVA base',
                           unit='MVA',
                           )

        self.Tg = NumParam(default=0.1, tex_name='T_g',
                           info='converter time const.', unit='s',
                           )
        self.Rrpwr = NumParam(default=10.0, tex_name='R_{rpwr}',
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
        self.Lvplsw = NumParam(default=1.0, tex_name='z_{Lvplsw}',
                               info='Low volt. P logic: 1-enable, 0-disable',
                               unit='bool',
                               )

        self.Lvpl1 = NumParam(default=1.0, tex_name='L_{vpl1}',
                              info='LVPL gain at Brkpt',
                              unit='p.u',
                              )
        self.Volim = NumParam(default=1.2, tex_name='V_{olim}',
                              info='Voltage lim for high volt. reactive current mgnt.',
                              unit='p.u.',
                              )
        self.Lvpnt1 = NumParam(default=0.8, tex_name='L_{vpnt1}',
                               info='High volt. point for low volt. active current mgnt.',
                               unit='p.u.',
                               )
        self.Lvpnt0 = NumParam(default=0.4, tex_name='L_{vpnt0}',
                               info='Low volt. point for low volt. active current mgnt.',
                               unit='p.u.',
                               )
        # TODO: ensure Lvpnt1 > Lvpnt0
        self.Iolim = NumParam(default=-1.5, tex_name='I_{olim}',
                              info='lower current limit for high volt. reactive current mgnt.',
                              unit='p.u. (mach base)',
                              current=True,
                              )
        self.Tfltr = NumParam(default=0.1, tex_name='T_{fltr}',
                              info='Voltage filter T const for low volt. active current mgnt.',
                              unit='s',
                              )
        self.Khv = NumParam(default=0.7, tex_name='K_{hv}',
                            info='Overvolt. compensation gain in high volt. reactive current mgnt.',
                            )
        self.Iqrmax = NumParam(default=1, tex_name='I_{qrmax}',
                               info='Upper limit on the ROC for reactive current',
                               unit='p.u.',
                               current=True,
                               )
        self.Iqrmin = NumParam(default=-1, tex_name='I_{qrmin}',
                               info='Lower limit on the ROC for reactive current',
                               unit='p.u.',
                               current=True,
                               )
        self.Accel = NumParam(default=0.0, tex_name='A_{ccel}',
                              info='Acceleration factor',
                              vrange=(0, 1.0),
                              )
        self.gammap = NumParam(default=1.0,
                               info="P ratio of linked static gen",
                               tex_name=r'\gamma_P'
                               )
        self.gammaq = NumParam(default=1.0,
                               info="Q ratio of linked static gen",
                               tex_name=r'\gamma_Q'
                               )


class REGCA1Model(Model):
    """
    REGCA1 implementation.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'RenGen'

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle',
                          e_str='-Pe',
                          ename='P',
                          tex_ename='P',
                          )

        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          e_str='-Qe',
                          ename='Q',
                          tex_ename='Q',
                          )

        self.vd = AliasAlgeb(self.v)

        self.p0s = ExtService(model='StaticGen',
                              src='p',
                              indexer=self.gen,
                              tex_name='P_{0s}',
                              info='initial P of the static gen',
                              )
        self.q0s = ExtService(model='StaticGen',
                              src='q',
                              indexer=self.gen,
                              tex_name='Q_{0s}',
                              info='initial Q of the static gen',
                              )
        self.p0 = ConstService(v_str='p0s * gammap',
                               tex_name='P_0',
                               info='initial P of this gen',
                               )
        self.q0 = ConstService(v_str='q0s * gammaq',
                               tex_name='Q_0',
                               info='initial Q of this gen',
                               )
        self.ra = ExtParam(model='StaticGen',
                           src='ra',
                           indexer=self.gen,
                           tex_name='r_a',
                           export=False,
                           )
        self.xs = ExtParam(model='StaticGen',
                           src='xs',
                           indexer=self.gen,
                           tex_name='x_s',
                           export=False,
                           )

        # --- INITIALIZATION ---
        self.q0gt0 = ConstService('Indicator(q0> 0)', tex_name='z_{q0>0}',
                                  info='flags for q0 below zero',
                                  )
        self.q0lt0 = ConstService('Indicator(q0< 0)', tex_name='z_{q0<0}',
                                  info='flags for q0 below zero',
                                  )

        self.Ipcmd0 = ConstService('p0 / v', info='initial Ipcmd',
                                   tex_name='I_{pcmd0}',
                                   )

        self.Iqcmd0 = ConstService('-q0 / v', info='initial Iqcmd',
                                   tex_name='I_{qcmd0}',
                                   )

        self.LVG = Piecewise(u=self.v, points=('Lvpnt0', 'Lvpnt1'),
                             funs=('0', '(v - Lvpnt0) * kLVG', '1'),
                             info='Ip gain during low voltage',
                             tex_name='L_{VG}',
                             )

        # `Ipcmd` is not defined when the initial `LVG_y` is zero
        self.Ipcmd = Algeb(tex_name='I_{pcmd}',
                           info='current component for active power',
                           e_str='Ipcmd0_LVG - Ipcmd',
                           v_str='Indicator(LVG_y>0) * Ipcmd0 / LVG_y + Indicator(LVG_y<=0) * 1',
                           diag_eps=True,
                           )

        self.Ipcmd0_LVG = PostInitService(v_str='Ipcmd',
                                          info='initial Ipcmd considering initial LVG output')

        self.Iqcmd = Algeb(tex_name='I_{qcmd}',
                           info='current component for reactive power',
                           e_str='Iqcmd0 - Iqcmd', v_str='Iqcmd0')

        # reactive power management

        # rate limiting logic (for fault recovery, although it does not detect any recovery)
        #   - activate upper limit when q0 > 0 (self.q0gt0)
        #   - activate lower limit when q0 < 0 (self.q0lt0)

        self.S1 = LagAntiWindupRate(u=self.Iqcmd,
                                    T=self.Tg, K=-1,
                                    lower=-9999, upper=9999, no_lower=True, no_upper=True,
                                    rate_lower=self.Iqrmin, rate_upper=self.Iqrmax,
                                    rate_lower_cond=self.q0lt0, rate_upper_cond=self.q0gt0,
                                    tex_name='S_1',
                                    info='Iqcmd delay',
                                    )  # output `S1_y` == `Iq`

        # piece-wise gain for low voltage active current mgnt.
        self.kLVG = ConstService(v_str='1 / (Lvpnt1 - Lvpnt0)',
                                 tex_name='k_{LVG}',
                                 )

        # piece-wise gain for LVPL
        self.kLVPL = ConstService(v_str='Lvplsw * Lvpl1 / (Brkpt - Zerox)',
                                  tex_name='k_{LVPL}',
                                  )

        self.S2 = Lag(u=self.v, T=self.Tfltr, K=1.0,
                      info='Voltage filter with no anti-windup',
                      tex_name='S_2',
                      )
        self.LVPL = Piecewise(u=self.S2_y,
                              points=('Zerox', 'Brkpt'),
                              funs=('0 + 9999*(1-Lvplsw)',
                                    '(S2_y - Zerox) * kLVPL + 9999 * (1-Lvplsw)',
                                    '9999'),
                              info='Low voltage Ipcmd upper limit',
                              tex_name='L_{VPL}',
                              )

        self.S0 = LagAntiWindupRate(u=self.Ipcmd, T=self.Tg, K=1,
                                    upper=self.LVPL_y, rate_upper=self.Rrpwr,
                                    lower=-9999, rate_lower=-9999,
                                    no_lower=True, rate_no_lower=True,
                                    tex_name='S_0',
                                    )  # `S0_y` is the output `Ip` in the block diagram

        self.Ipout = Algeb(e_str='S0_y * LVG_y -Ipout',
                           v_str='Ipcmd * LVG_y',
                           info='Output Ip current',
                           tex_name='I_{pout}',
                           )

        # high voltage part
        self.HVG = GainLimiter(u='v - Volim',
                               K=self.Khv, R=1,
                               info='High voltage gain block',
                               lower=0, upper=999, no_upper=True,
                               tex_name='H_{VG}'
                               )
        self.HVG.lim.no_warn = True
        self.HVG.lim.allow_adjust = False

        self.Iqout = GainLimiter(u='S1_y - HVG_y',
                                 K=1, R=1,
                                 lower=self.Iolim, upper=9999,
                                 no_upper=True, info='Iq output block',
                                 tex_name='I^{qout}',
                                 )  # `Iqout_y` is the final Iq output

        self.Pe = Algeb(tex_name='P_e', info='Active power output',
                        v_str='p0', e_str='Ipout * v - Pe')
        self.Qe = Algeb(tex_name='Q_e', info='Reactive power output',
                        v_str='q0', e_str='Iqout_y * v - Qe')

    def v_numeric(self, **kwargs):
        """
        Disable the corresponding `StaticGen`s.
        """
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class REGCA1(REGCA1Data, REGCA1Model):
    """
    Renewable energy generator model type A.

    Implements ``REGCA1`` in PSS/E, or ``REGC_A`` in PSLF and Powerworld.

    Volim is the voltage limit for high voltage reactive current management,
    which should be large than static bus voltage (Volim > v),
    or initialization error will occur.
    """

    def __init__(self, system, config):
        REGCA1Data.__init__(self)
        REGCA1Model.__init__(self, system, config)
