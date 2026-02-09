from collections import OrderedDict

import numpy as np

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        IdxParam, Lag, Limiter, Model, ModelData, NumParam,
                        Switcher, IsEqual)
from andes.core.block import (DeadBand1, GainLimiter, LagAWFreeze,
                              LagFreeze, LagRate, PITrackAWFreeze,)
from andes.core.service import (ApplyFunc, DataSelect, Replace,
                                VarService,)
from andes.core.var import AliasState


class REECB1Data(ModelData):
    """
    Renewable energy electrical control model REECB1 (reec_b) data.
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
                            info='Gain to compute Iqinj from V error (caution!!)',
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
        self.Tp = NumParam(default=0.02,
                           tex_name='T_p',
                           unit='s',
                           info='Filter time constant for Pe',
                           )
        self.QMax = NumParam(default=999.0,
                             tex_name='Q_{max}',
                             info='Upper limit for reactive power regulator',
                             power=True,
                             )
        self.QMin = NumParam(default=-999.0,
                             tex_name='Q_{min}',
                             info='Lower limit for reactive power regulator',
                             power=True,
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
        self.Tiq = NumParam(default=0.02,
                            tex_name='T_{iq}',
                            info='Filter time constant for Iq (used when QFLAG=0)',
                            )
        self.dPmax = NumParam(default=999.0,
                              tex_name='d_{Pmax}',
                              info='Power reference max. ramp rate (>0)',
                              power=True,
                              )
        self.dPmin = NumParam(default=-999.0,
                              tex_name='d_{Pin}',
                              info='Power reference min. ramp rate (<0)',
                              power=True,
                              )
        self.PMAX = NumParam(default=999.0,
                             tex_name='P_{max}',
                             info='Max. active power limit > 0',
                             power=True,
                             )
        self.PMIN = NumParam(default=0.0,
                             tex_name='P_{min}',
                             info='Min. active power limit',
                             power=True,
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


class REECB1Model(Model):
    """
    REEC_B model implementation.
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

        self.SWQ = Switcher(u=self.QFLAG, options=(0, 1), tex_name='SW_{Q}', cache=True)

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

        self.v = ExtAlgeb(model='RenGen',
                          src='vd',
                          indexer=self.reg,
                          tex_name=r'V_d',
                          info='d-axis bus voltage magnitude',
                          )

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
        self.pfaref0 = ConstService(v_str='atan2(q0, p0)', tex_name=r'\Phi_{ref0}',
                                    info='Initial power factor angle',
                                    )
        self.zp = IsEqual(self.p0, 0, cache=True)

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

        self.qref0 = ConstService(tex_name='q_{ref0}',
                                  v_str='-q0',
                                  )
        self.Qref = Algeb(tex_name='Q_{ref}',
                          info='external Q ref',
                          v_str='-qref0',
                          e_str='-qref0 - Qref',
                          unit='p.u.',
                          )

        # ignore `Qcpf` if `pfaref` is pi/2 by multiplying (1-zp_z1)
        self.Qcpf = Algeb(tex_name='Q_{cpf}',
                          info='Q calculated from P and power factor',
                          v_str='q0',
                          e_str='(1-zp_z1) * (S1_y * tan(pfaref) - Qcpf)',
                          diag_eps=True,
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

        self.PIQ = PITrackAWFreeze(u='SWV_s1 * Qerr + SWV_s0 * 0',
                                   kp=self.Kqp, ki=self.Kqi, ks=self.config.kqs,
                                   lower=self.VMIN, upper=self.VMAX,
                                   freeze=self.Volt_dip,
                                   )

        # REECB1: no Vref1; VFLAG=0 outputs 0, VFLAG=1 outputs PIQ_y
        self.Vsel = GainLimiter(u='SWV_s1 * PIQ_y',
                                K=1, R=1,
                                lower=self.VMIN, upper=self.VMAX,
                                info='Selection output of VFLAG',
                                )

        # --- Upper portion - Iqinj calculation (simplified, no state machine) ---

        # PSS/E convention: Vref0=0 means use initial bus voltage
        self.Vref0r = ConstService(v_str='Vref0',
                                   v_numeric=self._replace_vref0,
                                   tex_name='V_{ref0,r}',
                                   info='Replaced Vref0 (0 -> bus V)')

        self.Verr = Algeb(info='Voltage error (Vref0)',
                          v_str='Vref0r - s0_y',
                          e_str='Vref0r - s0_y - Verr',
                          tex_name='V_{err}',
                          )
        self.dbV = DeadBand1(u=self.Verr, lower=self.dbd1, upper=self.dbd2,
                             center=0.0,
                             enable='DB_{V}',
                             info='Deadband for voltage error (ref0)'
                             )

        self.Iqinj = GainLimiter(u='dbV_y', K=self.Kqv, R=1,
                                 lower=self.Iql1, upper=self.Iqh1,
                                 info='Gain-limited Iqinj from voltage error',
                                 )

        # --- Current limit logic (simplified, no VDL, no Thld2) ---

        Ipmax2sq0 = '(Imaxr**2 - Iqcmd0**2)'
        Ipmax2sq = '(Imaxr**2 - IqHL_y**2)'

        self.Ipmax2sq0 = ConstService(v_str=f'Piecewise((0, Le({Ipmax2sq0}, 0.0)), ({Ipmax2sq0}, True), \
                                              evaluate=False)',
                                      tex_name='I_{pmax20,nn}^2',
                                      )

        self.Ipmax2sq = VarService(v_str=f'Piecewise((0, Le({Ipmax2sq}, 0.0)), ({Ipmax2sq}, True), \
                                           evaluate=False)',
                                   tex_name='I_{pmax2}^2',
                                   )

        # Q priority (SWPQ_s0): Ipmax = sqrt(Imax^2 - Iqcmd^2)
        # P priority (SWPQ_s1): Ipmax = Imax
        self.Ipmax = Algeb(v_str='SWPQ_s0*sqrt(Ipmax2sq0) + SWPQ_s1*Imaxr',
                           e_str='SWPQ_s0*sqrt(Ipmax2sq) + SWPQ_s1*Imaxr - Ipmax',
                           tex_name='I_{pmax}',
                           diag_eps=True,
                           info='Upper limit on Ipcmd',
                           )

        Iqmax2sq0 = '(Imaxr**2 - Ipcmd0**2)'
        Iqmax2sq = '(Imaxr**2 - IpHL_y**2)'

        self.Iqmax2sq0 = ConstService(v_str=f'Piecewise((0, Le({Iqmax2sq0}, 0.0)), ({Iqmax2sq0}, True), \
                                              evaluate=False)',
                                      tex_name='I_{qmax,nn}^2',
                                      )

        self.Iqmax2sq = VarService(v_str=f'Piecewise((0, Le({Iqmax2sq}, 0.0)), ({Iqmax2sq}, True), \
                                           evaluate=False)',
                                   tex_name='I_{qmax2}^2')

        # Q priority (SWPQ_s0): Iqmax = Imax
        # P priority (SWPQ_s1): Iqmax = sqrt(Imax^2 - Ipcmd^2)
        self.Iqmax = Algeb(v_str='SWPQ_s0*Imaxr + SWPQ_s1*sqrt(Iqmax2sq0)',
                           e_str='SWPQ_s0*Imaxr + SWPQ_s1*sqrt(Iqmax2sq) - Iqmax',
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

        self.PIV = PITrackAWFreeze(u='SWQ_s1 * Vsel_y',
                                   x0='-SWQ_s1 * Iqcmd0',
                                   kp=self.Kvp, ki=self.Kvi, ks=self.config.kvs,
                                   lower=self.Iqmin, upper=self.Iqmax,
                                   freeze=self.Volt_dip,
                                   )

        self.s4 = LagFreeze(u='PFsel / vp',
                            T=self.Tiq, K=1,
                            freeze=self.Volt_dip,
                            tex_name='s_4',
                            info='Filter for reactive current (QFLAG=0 path)',
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
        self.IqHL = GainLimiter(u='Qsel + Iqinj_y',
                                K=1, R=1,
                                lower=self.Iqmin, upper=self.Iqmax)

        # --- Active power path (no PFLAG, no wg multiplication) ---
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

        # `s5_y` is `Pord` — pfilt feeds directly (no Psel/PFLAG)
        self.s5 = LagAWFreeze(u='pfilt_y', T=self.Tpord, K=1,
                              lower=self.PMIN, upper=self.PMAX,
                              freeze=self.Volt_dip,
                              tex_name='s5',
                              )

        self.Pord = AliasState(self.s5_y)

    def _replace_vref0(self, **kwargs):
        """PSS/E convention: Vref0=0 means use initial bus voltage."""
        out = np.array(self.Vref0.v, dtype=float)
        mask = np.equal(out, 0.0)
        out[mask] = self.v.v[mask]
        return out


class REECB1(REECB1Data, REECB1Model):
    """
    Renewable energy electrical control model B (REEC_B).

    REECB1 is a simplified variant of REECA1 with the following differences:

    - No voltage-dependent current limit (VDL) piecewise characteristics.
      Current limits use simple algebraic expressions based on ``Imax``
      and ``PQFLAG`` priority.
    - No state transition logic for reactive current injection hold.
      ``Iqinj`` is a direct proportional gain with limits.
    - No ``PFLAG`` speed dependency. ``Pref`` feeds directly through
      the rate limiter to ``Pord``.
    - No ``Vref1``. Only ``Vref0`` is used.
    - ``QFLAG=0`` uses direct reactive current (``PFsel/vp``) via
      a lag filter, bypassing the voltage PI controller.

    Regarding the reactive current injection during voltage dip:

    - Exercise caution when coordinating ``dbd1``, ``dbd2``,
      ``Vdip``, and ``Vup`` to avoid unintended responses.
    - ``Kqv`` controls the intensity of reactive power injection and
      needs to be tuned properly to avoid voltage overshoot.

    """

    def __init__(self, system, config):
        REECB1Data.__init__(self)
        REECB1Model.__init__(self, system, config)
