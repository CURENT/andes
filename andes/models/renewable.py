from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.block import Piecewise, Lag, GainLimiter, LagAntiWindupRate, LagAWFreeze
from andes.core.block import PITrackAWFreeze, LagFreeze, DeadBand1, LagRate
from andes.core.var import ExtAlgeb, Algeb

from andes.core.service import ConstService, FlagValue, ExtService, DataSelect
from andes.core.service import VarService, ExtendedEvent, Replace, ApplyFunc, VarHold
from andes.core.discrete import Switcher, Limiter
from collections import OrderedDict

import numpy as np  # NOQA


class REGCA1Data(ModelData):
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
        self.Iolim = NumParam(default=0.0, tex_name='I_{olim}',
                              info='lower current limit for high volt. reactive current mgnt.',
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
                          )

        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          e_str='-Qe',
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
                               lower=0, upper=999, no_upper=True,
                               )
        self.HVG.lim.no_warn = True

        self.Iqout = GainLimiter(u='S1_y- HVG_y', K=1, lower=self.Iolim, upper=9999,
                                 no_upper=True, info='Iq output block')  # `Iqout_y` is the final Iq output

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
    def __init__(self, system, config):
        REGCA1Data.__init__(self)
        REGCA1Model.__init__(self, system, config)


class REECA1Data(ModelData):
    """
    Renewable energy electrical control model REECA1 (reec_a) data.

    TODO: Flag the parameters in the machine base.
    """
    def __init__(self):
        ModelData.__init__(self)

        self.reg = IdxParam(model='RenGen',
                            info='Renewable generator idx',
                            mandatory=True,
                            )

        self.busr = IdxParam(info='Optional remote bus for voltage control',
                             model='Bus',
                             default=None,
                             )
        self.PFFLAG = NumParam(info='Power factor control flag; 1-PF control, 0-Q control',
                               mandatory=True,
                               unit='bool',
                               )
        self.VFLAG = NumParam(info='Voltage control flag; 1-Q control, 0-V control',
                              mandatory=True,
                              unit='bool',
                              )
        self.QFLAG = NumParam(info='Q control flag; 1-V or Q control, 0-const. PF or Q',
                              mandatory=True,
                              unit='bool',
                              )
        self.PFLAG = NumParam(info='P speed-dependency flag; 1-has speed dep., 0-no dep.',
                              mandatory=True,
                              unit='bool',
                              )
        self.PQFLAG = NumParam(info='P/Q priority flag for I limit; 0-Q priority, 1-P priority',
                               mandatory=True,
                               unit='bool',
                               )

        self.Vdip = NumParam(default=0.8,
                             tex_name='V_{dip}',
                             info='Low V threshold to activate Iqinj logic',
                             unit='p.u.',
                             )
        self.Vup = NumParam(default=1.2,
                            tex_name='V_{up}',
                            info='V threshold above which to activate Iqinj logic',
                            unit='p.u.',
                            )
        self.Trv = NumParam(default=0.02,
                            tex_name='T_{rv}',
                            info='Voltage filter time constant',
                            )
        self.dbd1 = NumParam(default=-0.1,
                             tex_name='d_{bd1}',
                             info='Lower bound of the voltage deadband (<=0)',
                             )
        self.dbd2 = NumParam(default=0.1,
                             tex_name='d_{bd2}',
                             info='Upper bound of the voltage deadband (>=0)',
                             )
        self.Kqv = NumParam(default=1.0,
                            vrange=(0, 10),
                            tex_name='K_{qv}',
                            info='Gain to compute Iqinj from V error',
                            )
        self.Iqh1 = NumParam(default=999.0,
                             tex_name='I_{qh1}',
                             info='Upper limit on Iqinj',
                             )
        self.Iql1 = NumParam(default=-999.0,
                             tex_name='I_{ql1}',
                             info='Lower limit on Iqinj',
                             )
        self.Vref0 = NumParam(default=1.0,
                              tex_name='V_{ref0}',
                              info='User defined Vref (if 0, use initial bus V)',
                              )
        self.Iqfrz = NumParam(default=0.0,
                              tex_name='I_{qfrz}',
                              info='Value at which Iqinj is held for Thld (if >0) seconds following a Vdip',
                              )
        self.Thld = NumParam(default=0.0,
                             tex_name='T_{hld}',
                             unit='s',
                             info='Time for which Iqinj is held. If >0, hold at Iqinj; if <0, hold at State 1 '
                                  'for |Thld|.',
                             )
        self.Thld2 = NumParam(default=0.0,
                              tex_name='T_{hld2}',
                              unit='s',
                              info='Time for which IPMAX is held after voltage dip ends',
                              )
        self.Tp = NumParam(default=0.02,
                           tex_name='T_p',
                           unit='s',
                           info='Filter time constant for Pe',
                           )
        self.QMax = NumParam(default=999.0,
                             tex_name='Q_{max}',
                             info='Upper limit for reactive power regulator',
                             )
        self.QMin = NumParam(default=-999.0,
                             tex_name='Q_{min}',
                             info='Lower limit for reactive power regulator',
                             )
        self.VMAX = NumParam(default=999.0,
                             tex_name='V_{max}',
                             info='Upper limit for voltage control',
                             )
        self.VMIN = NumParam(default=-999.0,
                             tex_name='V_{min}',
                             info='Lower limit for voltage control',
                             )
        self.Kqp = NumParam(default=1.0,
                            tex_name='K_{qp}',
                            info='Proportional gain for reactive power error',
                            )
        self.Kqi = NumParam(default=0.1,
                            tex_name='K_{qi}',
                            info='Integral gain for reactive power error',
                            )
        self.Kvp = NumParam(default=1.0,
                            tex_name='K_{vp}',
                            info='Proportional gain for voltage error',
                            )
        self.Kvi = NumParam(default=0.1,
                            tex_name='K_{vi}',
                            info='Integral gain for voltage error',
                            )
        self.Vref1 = NumParam(default=1.0,
                              non_zero=True,
                              tex_name='V_{ref1}',
                              info='Voltage ref. if VFLAG=0',
                              )
        self.Tiq = NumParam(default=0.02,
                            tex_name='T_{iq}',
                            info='Filter time constant for Iq',
                            )
        self.dPmax = NumParam(default=999.0,
                              tex_name='d_{Pmax}',
                              info='Power reference max. ramp rate (>0)',
                              )
        self.dPmin = NumParam(default=-999.0,
                              tex_name='d_{Pin}',
                              info='Power reference min. ramp rate (<0)',
                              )
        self.PMAX = NumParam(default=999.0,
                             tex_name='P_{max}',
                             info='Max. active power limit > 0',
                             )
        self.PMIN = NumParam(default=0.0,
                             tex_name='P_{min}',
                             info='Min. active power limit',
                             )
        self.Imax = NumParam(default=999.0,
                             tex_name='I_{max}',
                             info='Max. apparent current limit',
                             )
        self.Tpord = NumParam(default=0.02,
                              tex_name='T_{pord}',
                              info='Filter time constant for power setpoint',
                              )
        self.Vq1 = NumParam(default=0.2,
                            tex_name='V_{q1}',
                            info='Reactive power V-I pair (point 1), voltage',
                            )
        self.Iq1 = NumParam(default=2.0,
                            tex_name='I_{q1}',
                            info='Reactive power V-I pair (point 1), current',
                            )
        self.Vq2 = NumParam(default=0.4,
                            tex_name='V_{q2}',
                            info='Reactive power V-I pair (point 2), voltage',
                            )
        self.Iq2 = NumParam(default=4.0,
                            tex_name='I_{q2}',
                            info='Reactive power V-I pair (point 2), current',
                            )
        self.Vq3 = NumParam(default=0.8,
                            tex_name='V_{q3}',
                            info='Reactive power V-I pair (point 3), voltage',
                            )
        self.Iq3 = NumParam(default=8.0,
                            tex_name='I_{q3}',
                            info='Reactive power V-I pair (point 3), current',
                            )
        self.Vq4 = NumParam(default=1.0,
                            tex_name='V_{q4}',
                            info='Reactive power V-I pair (point 4), voltage',
                            )
        self.Iq4 = NumParam(default=10,
                            tex_name='I_{q4}',
                            info='Reactive power V-I pair (point 4), current',
                            )
        self.Vp1 = NumParam(default=0.2,
                            tex_name='V_{p1}',
                            info='Active power V-I pair (point 1), voltage',
                            )
        self.Ip1 = NumParam(default=2.0,
                            tex_name='I_{p1}',
                            info='Active power V-I pair (point 1), current',
                            )
        self.Vp2 = NumParam(default=0.4,
                            tex_name='V_{p2}',
                            info='Active power V-I pair (point 2), voltage',
                            )
        self.Ip2 = NumParam(default=4.0,
                            tex_name='I_{p2}',
                            info='Active power V-I pair (point 2), current',
                            )
        self.Vp3 = NumParam(default=0.8,
                            tex_name='V_{p3}',
                            info='Active power V-I pair (point 3), voltage',
                            )
        self.Ip3 = NumParam(default=8.0,
                            tex_name='I_{p3}',
                            info='Active power V-I pair (point 3), current',
                            )
        self.Vp4 = NumParam(default=1.0,
                            tex_name='V_{p4}',
                            info='Active power V-I pair (point 4), voltage',
                            )
        self.Ip4 = NumParam(default=12.0,
                            tex_name='I_{p4}',
                            info='Active power V-I pair (point 4), current',
                            )


class REECA1Model(Model):
    """
    REEC_A model implementation.
    """
    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.config.add(OrderedDict((('kqs', 2),
                                     ('kvs', 2),
                                     ('tpfilt', 0.02),
                                     )))
        self.config.add_extra('_help',
                              kqs='Q PI controller tracking gain',
                              kvs='Voltage PI controller tracking gain',
                              tpfilt='Time const. for Pref filter',
                              )

        # --- Sanitize inputs ---
        self.Imaxr = Replace(self.Imax, flt=lambda x: np.less_equal(x, 0), new_val=1e8)

        # --- Flag switchers ---
        self.SWPF = Switcher(u=self.PFFLAG, options=(0, 1), tex_name='SW_{PF}', cache=True)

        self.SWV = Switcher(u=self.VFLAG, options=(0, 1), tex_name='SW_{V}', cache=True)

        self.SWQ = Switcher(u=self.QFLAG, options=(0, 1), tex_name='SW_{V}', cache=True)

        self.SWP = Switcher(u=self.PFLAG, options=(0, 1), tex_name='SW_{P}', cache=True)

        self.SWPQ = Switcher(u=self.PQFLAG, options=(0, 1), tex_name='SW_{PQ}', cache=True)

        # --- External parameters ---
        self.bus = ExtParam(model='RenGen', src='bus', indexer=self.reg, export=False,
                            info='Retrieved bus idx', dtype=str, default=None,
                            )

        self.buss = DataSelect(self.busr, self.bus, info='selected bus (bus or busr)')

        self.gen = ExtParam(model='RenGen', src='gen', indexer=self.reg, export=False,
                            info='Retrieved StaticGen idx', dtype=str, default=None,
                            )

        # --- External variables ---
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle',
                          )

        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          )  # check whether to use `bus` or `buss`

        self.Pe = ExtAlgeb(model='RenGen', src='Pe', indexer=self.reg, export=False,
                           info='Retrieved Pe of RenGen')

        self.Qe = ExtAlgeb(model='RenGen', src='Qe', indexer=self.reg, export=False,
                           info='Retrieved Qe of RenGen')

        self.Ipcmd = ExtAlgeb(model='RenGen', src='Ipcmd', indexer=self.reg, export=False,
                              info='Retrieved Ipcmd of RenGen',
                              e_str='-Ipcmd0 + IpHL_y',
                              )

        self.Iqcmd = ExtAlgeb(model='RenGen', src='Iqcmd', indexer=self.reg, export=False,
                              info='Retrieved Iqcmd of RenGen',
                              e_str='-Iqcmd0 - IqHL_y',
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

        # Initial current commands
        self.Ipcmd0 = ConstService('p0 / v', info='initial Ipcmd')

        self.Iqcmd0 = ConstService('-q0 / v', info='initial Iqcmd')

        # --- Initial power factor ---
        self.pfaref0 = ConstService(v_str='atan(q0/p0)', tex_name=r'\Phi_{ref0}',
                                    info='Initial power factor angle',
                                    )

        # --- Discrete components ---
        self.Vcmp = Limiter(u=self.v, lower=self.Vdip, upper=self.Vup, tex_name='V_{cmp}',
                            info='Voltage dip comparator', equal=False,
                            )
        self.Volt_dip = VarService(v_str='1 - Vcmp_zi',
                                   info='Voltage dip flag; 1-dip, 0-normal',
                                   tex_name='z_{Vdip}',
                                   )

        # --- Equations begin ---
        self.s0 = Lag(u=self.v, T=self.Trv, K=1,
                      info='Voltage filter',
                      )
        self.VLower = Limiter(u=self.v, lower=0.01, upper=999, no_upper=True,
                              info='Limiter for lower voltage cap',
                              )
        self.vp = Algeb(tex_name='V_p',
                        info='Sensed lower-capped voltage',
                        v_str='v * VLower_zi + 0.01 * VLower_zl',
                        e_str='v * VLower_zi + 0.01 * VLower_zl - vp',
                        )

        self.pfaref = Algeb(tex_name=r'\Phi_{ref}',
                            info='power factor angle ref',
                            unit='rad',
                            v_str='pfaref0',
                            e_str='pfaref0 - pfaref',
                            )

        self.S1 = Lag(u='Pe', T=self.Tp, K=1, tex_name='S_1', info='Pe filter',
                      )

        self.Qcpf = Algeb(tex_name='Q_{cpf}',
                          info='Q calculated from P and power factor',
                          v_str='q0',
                          e_str='S1_y * tan(pfaref) - Qcpf',
                          unit='p.u.',
                          )

        self.Pref = Algeb(tex_name='P_{ref}',
                          info='external P ref',
                          v_str='p0',
                          e_str='p0 - Pref',
                          unit='p.u.',
                          )

        self.Qref = Algeb(tex_name='Q_{ref}',
                          info='external Q ref',
                          v_str='q0',
                          e_str='q0 - Qref',
                          unit='p.u.',
                          )

        self.PFsel = Algeb(v_str='SWPF_s0*Qref + SWPF_s1*Qcpf',
                           e_str='SWPF_s0*Qref + SWPF_s1*Qcpf - PFsel',
                           info='Output of PFFLAG selector',
                           )

        self.PFlim = Limiter(u=self.PFsel, lower=self.QMin, upper=self.QMax)

        self.Qerr = Algeb(tex_name='Q_{err}',
                          info='Reactive power error',
                          v_str='(PFsel*PFlim_zi + QMin*PFlim_zl + QMax*PFlim_zu) - Qe',
                          e_str='(PFsel*PFlim_zi + QMin*PFlim_zl + QMax*PFlim_zu) - Qe - Qerr',
                          )

        self.PIQ = PITrackAWFreeze(u=self.Qerr,
                                   kp=self.Kqp, ki=self.Kqi, ks=self.config.kqs,
                                   lower=self.VMIN, upper=self.VMAX,
                                   freeze=self.Volt_dip,
                                   )

        # If `VFLAG=0`, set the input as `Vref1` (see the NREL report)
        self.Vsel = GainLimiter(u='SWV_s0 * Vref1 + SWV_s1 * PIQ_y', K=1,
                                lower=self.VMIN, upper=self.VMAX,
                                info='Selection output of VFLAG',
                                )

        # --- Placeholders for `Iqmin` and `Iqmax` ---

        self.s4 = LagFreeze(u='PFsel / vp', T=self.Tiq, K=1,
                            freeze=self.Volt_dip,
                            tex_name='s_4',
                            info='Filter for calculated voltage with freeze',
                            )

        self.Qsel = Algeb(info='Selection output of QFLAG',
                          v_str='SWQ_s1 * PIV_y + SWQ_s0 * s4_y',
                          e_str='SWQ_s1 * PIV_y + SWQ_s0 * s4_y - Qsel',
                          tex_name='Q_{sel}',
                          )

        # --- Upper portion - Iqinj calculation ---

        self.Verr = Algeb(info='Voltage error (Vref0)',
                          v_str='Vref0 - s0_y',
                          e_str='Vref0 - s0_y - Verr',
                          tex_name='V_{err}',
                          )
        self.dbV = DeadBand1(u=self.Verr, lower=self.dbd1, upper=self.dbd2,
                             center=0.0,
                             enable='DB_{V}',
                             info='Deadband for voltage error (ref0)'
                             )

        self.pThld = ConstService(v_str='Thld > 0', tex_name='p_{Thld}')

        self.nThld = ConstService(v_str='Thld < 0', tex_name='n_{Thld}')

        self.Thld_abs = ConstService(v_str='abs(Thld)', tex_name='|Thld|')

        self.fThld = ExtendedEvent(self.Volt_dip,
                                   t_ext=self.Thld_abs,
                                   )

        # Gain after dbB
        Iqv = "(dbV_y * Kqv)"
        Iqinj = f'{Iqv} * Volt_dip + ' \
                f'(1 - Volt_dip) * fThld * ({Iqv} * nThld + Iqfrz * pThld)'

        # state transition, output of Iqinj
        self.Iqinj = Algeb(v_str=Iqinj,
                           e_str=Iqinj + ' - Iqinj',
                           tex_name='I_{qinj}',
                           info='Additional Iq signal during under- or over-voltage',
                           )

        # TODO: calculate Iqcmd

        # --- Lower portion - active power ---
        self.wg = Algeb(tex_name=r'\omega_g',
                        info='Drive train generator speed',
                        v_str='1.0',
                        e_str='1.0 - wg',
                        )
        self.pfilt = LagRate(u=self.Pref, T=self.config.tpfilt, K=1,
                             rate_lower=self.dPmin, rate_upper=self.dPmax,
                             info='Active power filter with rate limits',
                             tex_name='P_{filt}',
                             )
        self.Psel = Algeb(tex_name='P_{sel}',
                          info='Output selection of PFLAG',
                          v_str='SWP_s1*wg*pfilt_y + SWP_s0*pfilt_y',
                          e_str='SWP_s1*wg*pfilt_y + SWP_s0*pfilt_y - Psel',
                          )
        self.s5 = LagAWFreeze(u=self.Psel, T=self.Tpord, K=1,
                              lower=self.PMIN, upper=self.PMAX,
                              freeze=self.Volt_dip,
                              tex_name='s5',
                              )
        self.Ipulim = Algeb(info='Unlimited Ipcmd',
                            tex_name='I_{pulim}',
                            v_str='s5_y / vp',
                            e_str='s5_y / vp - Ipulim',
                            )

        # --- Current limit logic ---

        self.kVq12 = ConstService(v_str='(Iq2 - Iq1) / (Vq2 - Vq1)',
                                  tex_name='k_{Vq12}',
                                  )
        self.kVq23 = ConstService(v_str='(Iq3 - Iq2) / (Vq3 - Vq2)',
                                  tex_name='k_{Vq23}',
                                  )
        self.kVq34 = ConstService(v_str='(Iq4 - Iq3) / (Vq4 - Vq3)',
                                  tex_name='k_{Vq34}',
                                  )

        self.zVDL1 = ConstService(v_str='(Vq1 <= Vq2) & (Vq2 <= Vq3) & (Vq3 <= Vq4) & '
                                        '(Iq1 <= Iq2) & (Iq2 <= Iq3) & (Iq3 <= Iq4)',
                                  tex_name='z_{VDL1}',
                                  info='True if VDL1 is in service',
                                  )

        self.VDL1 = Piecewise(u=self.s0_y,
                              points=('Vq1', 'Vq2', 'Vq3', 'Vq4'),
                              funs=('Iq1',
                                    f'({self.s0_y.name} - Vq1) * kVq12 + Iq1',
                                    f'({self.s0_y.name} - Vq2) * kVq23 + Iq2',
                                    f'({self.s0_y.name} - Vq3) * kVq34 + Iq3',
                                    'Iq4'),
                              tex_name='V_{DL1}',
                              info='Piecewise linear characteristics of Vq-Iq',
                              )

        self.kVp12 = ConstService(v_str='(Ip2 - Ip1) / (Vp2 - Vp1)',
                                  tex_name='k_{Vp12}',
                                  )
        self.kVp23 = ConstService(v_str='(Ip3 - Ip2) / (Vp3 - Vp2)',
                                  tex_name='k_{Vp23}',
                                  )
        self.kVp34 = ConstService(v_str='(Ip4 - Ip3) / (Vp4 - Vp3)',
                                  tex_name='k_{Vp34}',
                                  )

        self.zVDL2 = ConstService(v_str='(Vp1 <= Vp2) & (Vp2 <= Vp3) & (Vp3 <= Vp4) & '
                                        '(Ip1 <= Ip2) & (Ip2 <= Ip3) & (Ip3 <= Ip4)',
                                  tex_name='z_{VDL2}',
                                  info='True if VDL2 is in service',
                                  )

        self.VDL2 = Piecewise(u=self.s0_y,
                              points=('Vp1', 'Vp2', 'Vp3', 'Vp4'),
                              funs=('Ip1',
                                    f'({self.s0_y.name} - Vp1) * kVp12 + Ip1',
                                    f'({self.s0_y.name} - Vp2) * kVp23 + Ip2',
                                    f'({self.s0_y.name} - Vp3) * kVp34 + Ip3',
                                    'Ip4'),
                              tex_name='V_{DL2}',
                              info='Piecewise linear characteristics of Vp-Ip',
                              )

        self.fThld2 = ExtendedEvent(self.Volt_dip,
                                    t_ext=self.Thld2,
                                    extend_only=True,
                                    )

        # --- For debugging, TODO: remove after testing ---
        # self.fThld_out = Algeb(v_str='fThld',
        #                        e_str='fThld - fThld_out',
        #                        )
        #
        # self.fThld2_out = Algeb(v_str='fThld2',
        #                         e_str='fThld2 - fThld2_out',
        #                         )
        # --- Debugging ends

        self.VDL1c = VarService(v_str='VDL1_y < Imaxr')

        self.VDL2c = VarService(v_str='VDL2_y < Imaxr')

        # `Iqmax` not considering mode or `Thld2`
        Iqmax1 = '(zVDL1*(VDL1c*VDL1_y + (1-VDL1c)*Imaxr) + 1e8*(1-zVDL1))'

        # `Ipmax` not considering mode or `Thld2`
        Ipmax1 = '(zVDL2*(VDL2c*VDL2_y + (1-VDL2c)*Imaxr) + 1e8*(1-zVDL2))'

        Ipmax2sq = '(Imax**2 - IqHL_y**2)'

        Ipmax2sq0 = '(Imax**2 - Iqcmd0**2)'

        Ipmax2 = f'Piecewise((0, {Ipmax2sq} <= 0.0), (sqrt({Ipmax2sq}), True))'

        Ipmax20 = f'Piecewise((0, {Ipmax2sq0} <= 0.0), (sqrt({Ipmax2sq0}), True))'

        # --- For debugging `Ipmax` ---

        # self.Ipmax1 = Algeb(v_str=Ipmax1, e_str=f'{Ipmax1} - Ipmax1')
        #
        # self.Ipmax2sq = Algeb(v_str=Ipmax2sq0, e_str=f'{Ipmax2sq} - Ipmax2sq')
        #
        # self.Ipmax2 = Algeb(v_str=Ipmax20, e_str=f'{Ipmax2} - Ipmax2')
        #
        # self.Ipmaxhv = Algeb(v_str='Ipmax', e_str=f'Ipmaxh - Ipmaxhv', tex_name='I_{pmaxhv}')

        # --- Debugging ends ---

        Ipmax = f'((1-fThld2) * (SWPQ_s0*{Ipmax2} + SWPQ_s1*{Ipmax1}))'  # TODO: +fThld2 * Ipmaxh

        Ipmax0 = f'((1-fThld2) * (SWPQ_s0*{Ipmax20} + SWPQ_s1*{Ipmax1}))'

        self.Ipmax = Algeb(v_str=f'{Ipmax0}',
                           e_str=f'{Ipmax} + (fThld2 * Ipmaxh) - Ipmax',
                           tex_name='I_{pmax}',
                           diag_eps=1e-6,
                           info='Upper limit on Ipcmd',
                           )

        self.Ipmaxh = VarHold(self.Ipmax, hold=self.fThld2)

        Iqmax2sq = '(Imax**2 - IpHL_y**2)'

        Iqmax2sq0 = '(Imax**2 - Ipcmd0**2)'   # initialization equation by using `Ipcmd0`

        Iqmax2 = f'Piecewise((0, {Iqmax2sq} <= 0.0), (sqrt({Iqmax2sq}), True))'

        Iqmax20 = f'Piecewise((0, {Iqmax2sq0} <= 0.0), (sqrt({Iqmax2sq0}), True))'

        Iqmax = f'(SWPQ_s0*{Iqmax1} + SWPQ_s1*{Iqmax2})'

        Iqmax0 = f'(SWPQ_s0*{Iqmax1} + SWPQ_s1*{Iqmax20})'

        self.Iqmax = Algeb(v_str=f'{Iqmax0}',
                           e_str=f'{Iqmax} - Iqmax',
                           tex_name='I_{qmax}',
                           info='Upper limit on Iqcmd',
                           )

        self.Iqmin = ApplyFunc(self.Iqmax, lambda x: -x, cache=False,
                               tex_name='I_{qmin}',
                               info='Lower limit on Iqcmd',
                               )

        self.Ipmin = ConstService(v_str='0.0', tex_name='I_{pmin}',
                                  info='Lower limit on Ipcmd',
                                  )

        self.PIV = PITrackAWFreeze(u='Vsel_y - s0_y * SWV_s0',
                                   kp=self.Kvp, ki=self.Kvi, ks=self.config.kvs,
                                   lower=self.Iqmin, upper=self.Iqmax,
                                   freeze=self.Volt_dip,
                                   )

        self.IpHL = GainLimiter(u='s5_y / vp', K=1, lower=self.Ipmin, upper=self.Ipmax,
                                )

        self.IqHL = GainLimiter(u='Qsel + Iqinj', K=1, lower=self.Iqmin, upper=self.Iqmax)

        # --- Duplicate output - consider removing later ---

        # self.Ipout = Algeb(info='Ipcmd limited output',
        #                    v_str='IpHL_y',
        #                    e_str='IpHL_y - Ipout',
        #                    )
        #
        # self.Iqout = Algeb(info='Iqcmd limited output',
        #                    v_str='IqHL_y',
        #                    e_str='IqHL_y - Iqout',
        #                    )

        # ---


class REECA1(REECA1Data, REECA1Model):
    """
    Renewable energy electrical control.
    """
    def __init__(self, system, config):
        REECA1Data.__init__(self)
        REECA1Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'RenElectrical'
