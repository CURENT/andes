from collections import OrderedDict

import numpy as np

from andes.core import ModelData, IdxParam, NumParam, Model, Switcher, ExtParam, ExtAlgeb, ExtService, \
    ConstService, \
    Limiter, Lag, Algeb, Piecewise
from andes.core.block import PITrackAWFreeze, GainLimiter, LagFreeze, DeadBand1, LagRate, LagAWFreeze
from andes.core.service import Replace, DataSelect, VarService, ExtendedEvent, VarHold, ApplyFunc
from andes.core.var import AliasState


class REECA1Data(ModelData):
    """
    Renewable energy electrical control model REECA1 (reec_a) data.
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
        self.dbd1 = NumParam(default=-0.02,
                             tex_name='d_{bd1}',
                             info='Lower bound of the voltage deadband (<=0)',
                             )
        self.dbd2 = NumParam(default=0.02,
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
                              info='Hold Iqinj at the value for Thld (>0) seconds following a Vdip',
                              )
        self.Thld = NumParam(default=0.0,
                             tex_name='T_{hld}',
                             unit='s',
                             info='Time for which Iqinj is held. Hold at Iqinj if>0; hold at State 1 if<0',
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
                             current=True,
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
                            current=True,
                            )
        self.Vq2 = NumParam(default=0.4,
                            tex_name='V_{q2}',
                            info='Reactive power V-I pair (point 2), voltage',
                            )
        self.Iq2 = NumParam(default=4.0,
                            tex_name='I_{q2}',
                            info='Reactive power V-I pair (point 2), current',
                            current=True,
                            )
        self.Vq3 = NumParam(default=0.8,
                            tex_name='V_{q3}',
                            info='Reactive power V-I pair (point 3), voltage',
                            )
        self.Iq3 = NumParam(default=8.0,
                            tex_name='I_{q3}',
                            info='Reactive power V-I pair (point 3), current',
                            current=True,
                            )
        self.Vq4 = NumParam(default=1.0,
                            tex_name='V_{q4}',
                            info='Reactive power V-I pair (point 4), voltage',
                            )
        self.Iq4 = NumParam(default=10,
                            tex_name='I_{q4}',
                            info='Reactive power V-I pair (point 4), current',
                            current=True,
                            )
        self.Vp1 = NumParam(default=0.2,
                            tex_name='V_{p1}',
                            info='Active power V-I pair (point 1), voltage',
                            )
        self.Ip1 = NumParam(default=2.0,
                            tex_name='I_{p1}',
                            info='Active power V-I pair (point 1), current',
                            current=True,
                            )
        self.Vp2 = NumParam(default=0.4,
                            tex_name='V_{p2}',
                            info='Active power V-I pair (point 2), voltage',
                            )
        self.Ip2 = NumParam(default=4.0,
                            tex_name='I_{p2}',
                            info='Active power V-I pair (point 2), current',
                            current=True,
                            )
        self.Vp3 = NumParam(default=0.8,
                            tex_name='V_{p3}',
                            info='Active power V-I pair (point 3), voltage',
                            )
        self.Ip3 = NumParam(default=8.0,
                            tex_name='I_{p3}',
                            info='Active power V-I pair (point 3), current',
                            current=True,
                            )
        self.Vp4 = NumParam(default=1.0,
                            tex_name='V_{p4}',
                            info='Active power V-I pair (point 4), voltage',
                            )
        self.Ip4 = NumParam(default=12.0,
                            tex_name='I_{p4}',
                            info='Active power V-I pair (point 4), current',
                            current=True,
                            )


class REECA1Model(Model):
    """
    REEC_A model implementation.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'RenExciter'

        self.config.add(OrderedDict((('kqs', 2),
                                     ('kvs', 2),
                                     ('tpfilt', 0.02),
                                     )))
        self.config.add_extra('_help',
                              kqs='Q PI controller tracking gain',
                              kvs='Voltage PI controller tracking gain',
                              tpfilt='Time const. for Pref filter',
                              )
        self.config.add_extra('_tex',
                              kqs='K_{qs}',
                              kvs='K_{vs}',
                              tpfilt='T_{pfilt}',
                              )

        # --- Sanitize inputs ---
        self.Imaxr = Replace(self.Imax, flt=lambda x: np.less_equal(x, 0), new_val=1e8,
                             tex_name='I_{maxr}')

        # --- Flag switchers ---
        self.SWPF = Switcher(u=self.PFFLAG, options=(0, 1), tex_name='SW_{PF}', cache=True)

        self.SWV = Switcher(u=self.VFLAG, options=(0, 1), tex_name='SW_{V}', cache=True)

        self.SWQ = Switcher(u=self.QFLAG, options=(0, 1), tex_name='SW_{V}', cache=True)

        self.SWP = Switcher(u=self.PFLAG, options=(0, 1), tex_name='SW_{P}', cache=True)

        self.SWPQ = Switcher(u=self.PQFLAG, options=(0, 1), tex_name='SW_{PQ}', cache=True)

        # --- External parameters ---
        self.bus = ExtParam(model='RenGen', src='bus', indexer=self.reg, export=False,
                            info='Retrieved bus idx', vtype=str, default=None,
                            )

        self.buss = DataSelect(self.busr, self.bus, info='selected bus (bus or busr)')

        self.gen = ExtParam(model='RenGen', src='gen', indexer=self.reg, export=False,
                            info='Retrieved StaticGen idx', vtype=str, default=None,
                            )

        self.Sn = ExtParam(model='RenGen', src='Sn', indexer=self.reg,
                           tex_name='S_n', export=False,
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

        self.p0 = ExtService(model='RenGen',
                             src='p0',
                             indexer=self.reg,
                             tex_name='P_0',
                             )
        self.q0 = ExtService(model='RenGen',
                             src='q0',
                             indexer=self.reg,
                             tex_name='Q_0',
                             )

        # Initial current commands
        self.Ipcmd0 = ConstService('p0 / v', info='initial Ipcmd')

        self.Iqcmd0 = ConstService('-q0 / v', info='initial Iqcmd')

        # --- Initial power factor angle ---
        # NOTE: if `p0` = 0, `pfaref0` = pi/2, `tan(pfaref0)` = inf
        self.pfaref0 = ConstService(v_str='atan2(q0, p0)', tex_name=r'\Phi_{ref0}',
                                    info='Initial power factor angle',
                                    )
        # flag devices with `p0`=0, which causes `tan(PF) = +inf`
        self.zp0 = ConstService(v_str='Eq(p0, 0)',
                                vtype=float,
                                tex_name='z_{p0}',
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

        # ignore `Qcpf` if `pfaref` is pi/2 by multiplying (1-zp0)
        self.Qcpf = Algeb(tex_name='Q_{cpf}',
                          info='Q calculated from P and power factor',
                          v_str='q0',
                          e_str='(1-zp0) * (S1_y * tan(pfaref) - Qcpf)',
                          diag_eps=True,
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
        self.Vsel = GainLimiter(u='SWV_s0 * Vref1 + SWV_s1 * PIQ_y',
                                K=1, R=1,
                                lower=self.VMIN, upper=self.VMAX,
                                info='Selection output of VFLAG',
                                )

        # --- Placeholders for `Iqmin` and `Iqmax` ---

        self.s4 = LagFreeze(u='PFsel / vp', T=self.Tiq, K=1,
                            freeze=self.Volt_dip,
                            tex_name='s_4',
                            info='Filter for calculated voltage with freeze',
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

        self.pThld = ConstService(v_str='Indicator(Thld > 0)', tex_name='p_{Thld}')

        self.nThld = ConstService(v_str='Indicator(Thld < 0)', tex_name='n_{Thld}')

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

        # --- Lower portion - active power ---
        self.wg = Algeb(tex_name=r'\omega_g',
                        info='Drive train generator speed',
                        v_str='1.0',
                        e_str='1.0 - wg',
                        )

        self.Pref = Algeb(tex_name='P_{ref}',
                          info='external P ref',
                          v_str='p0 / wg',
                          e_str='p0 / wg - Pref',
                          unit='p.u.',
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

        # `s5_y` is `Pord`
        self.s5 = LagAWFreeze(u=self.Psel, T=self.Tpord, K=1,
                              lower=self.PMIN, upper=self.PMAX,
                              freeze=self.Volt_dip,
                              tex_name='s5',
                              )

        self.Pord = AliasState(self.s5_y)

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

        self.VDL1c = VarService(v_str='Lt(VDL1_y, Imaxr)')

        self.VDL2c = VarService(v_str='Lt(VDL2_y, Imaxr)')

        # `Iqmax` not considering mode or `Thld2`
        Iqmax1 = '(zVDL1*(VDL1c*VDL1_y + (1-VDL1c)*Imaxr) + 1e8*(1-zVDL1))'

        # `Ipmax` not considering mode or `Thld2`
        Ipmax1 = '(zVDL2*(VDL2c*VDL2_y + (1-VDL2c)*Imaxr) + 1e8*(1-zVDL2))'

        Ipmax2sq0 = '(Imax**2 - Iqcmd0**2)'

        Ipmax2sq = '(Imax**2 - IqHL_y**2)'

        # `Ipmax20`-squared (non-negative)
        self.Ipmax2sq0 = ConstService(v_str=f'Piecewise((0.0, Le({Ipmax2sq0}, 0.0)), ({Ipmax2sq0}, True))',
                                      tex_name='I_{pmax20,nn}^2',
                                      )

        self.Ipmax2sq = VarService(v_str=f'Piecewise((0.0, Le({Ipmax2sq}, 0.0)), ({Ipmax2sq}, True))',
                                   tex_name='I_{pmax2}^2',
                                   )

        Ipmax = f'((1-fThld2) * (SWPQ_s0*sqrt(Ipmax2sq) + SWPQ_s1*{Ipmax1}))'

        Ipmax0 = f'((1-fThld2) * (SWPQ_s0*sqrt(Ipmax2sq0) + SWPQ_s1*{Ipmax1}))'

        self.Ipmax = Algeb(v_str=f'{Ipmax0}',
                           e_str=f'{Ipmax} + (fThld2 * Ipmaxh) - Ipmax',
                           tex_name='I_{pmax}',
                           diag_eps=True,
                           info='Upper limit on Ipcmd',
                           )

        self.Ipmaxh = VarHold(self.Ipmax, hold=self.fThld2)

        Iqmax2sq = '(Imax**2 - IpHL_y**2)'

        Iqmax2sq0 = '(Imax**2 - Ipcmd0**2)'  # initialization equation by using `Ipcmd0`

        self.Iqmax2sq0 = ConstService(v_str=f'Piecewise((0.0, Le({Iqmax2sq0}, 0.0)), ({Iqmax2sq0}, True))',
                                      tex_name='I_{qmax,nn}^2',
                                      )

        self.Iqmax2sq = VarService(v_str=f'Piecewise((0.0, Le({Iqmax2sq}, 0.0)), ({Iqmax2sq}, True))',
                                   tex_name='I_{qmax2}^2')

        self.Iqmax = Algeb(v_str=f'(SWPQ_s0*{Iqmax1} + SWPQ_s1*sqrt(Iqmax2sq0))',
                           e_str=f'(SWPQ_s0*{Iqmax1} + SWPQ_s1*sqrt(Iqmax2sq)) - Iqmax',
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
                                   x0='-SWQ_s1 * Iqcmd0',
                                   kp=self.Kvp, ki=self.Kvi, ks=self.config.kvs,
                                   lower=self.Iqmin, upper=self.Iqmax,
                                   freeze=self.Volt_dip,
                                   )

        self.Qsel = Algeb(info='Selection output of QFLAG',
                          v_str='SWQ_s1 * PIV_y + SWQ_s0 * s4_y',
                          e_str='SWQ_s1 * PIV_y + SWQ_s0 * s4_y - Qsel',
                          tex_name='Q_{sel}',
                          )

        # `IpHL_y` is `Ipcmd`
        self.IpHL = GainLimiter(u='s5_y / vp',
                                K=1, R=1,
                                lower=self.Ipmin, upper=self.Ipmax,
                                )

        # `IqHL_y` is `Iqcmd`
        self.IqHL = GainLimiter(u='Qsel + Iqinj',
                                K=1, R=1,
                                lower=self.Iqmin, upper=self.Iqmax)


class REECA1(REECA1Data, REECA1Model):
    """
    Renewable energy electrical control.

    There are two user-defined voltages: `Vref0` and `Vref1`.

    - The difference between the initial bus voltage and `Vref0`
      should be within the voltage deadbands `dbd1` and `dbd2`.
    - If `VFLAG=0`, the input to the second PI controller will
      be `Vref1`.

    """

    def __init__(self, system, config):
        REECA1Data.__init__(self)
        REECA1Model.__init__(self, system, config)
