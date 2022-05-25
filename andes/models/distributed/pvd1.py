"""
Distributed PV models.
"""
from collections import OrderedDict

from andes.core.block import DeadBand1, GainLimiter, Lag
from andes.core.discrete import Limiter, Switcher
from andes.core.model import Model, ModelData
from andes.core.param import IdxParam, NumParam
from andes.core.service import (ConstService, DataSelect, DeviceFinder,
                                ExtService, VarService,)
from andes.core.var import Algeb, ExtAlgeb


class PVD1Data(ModelData):
    """
    Data for distributed PV.
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
                           info='device MVA rating',
                           unit='MVA',
                           )

        self.fn = NumParam(default=60.0, tex_name='f_n',
                           info='nominal frequency',
                           unit='Hz',
                           )

        self.busf = IdxParam(info='Optional BusFreq measurement device idx',
                             model='BusFreq',
                             default=None,
                             )

        self.xc = NumParam(default=0.0, tex_name='x_c',
                           info='coupling reactance',
                           unit='p.u.',
                           z=True,
                           )

        self.pqflag = NumParam(info='P/Q priority for I limit; 0-Q priority, 1-P priority',
                               mandatory=True,
                               unit='bool',
                               )

        # --- parameters found from ESIG.energy ---
        self.igreg = IdxParam(model='Bus',
                              info='Remote bus idx for droop response, None for local',
                              )

        self.qmx = NumParam(default=0.33, tex_name='q_{mx}',
                            info='Max. reactive power command',
                            power=True,
                            unit='pu',
                            )

        self.qmn = NumParam(default=-0.33, tex_name='q_{mn}',
                            info='Min. reactive power command',
                            power=True,
                            unit='pu',
                            )
        self.pmx = NumParam(default=9999.0, info='maximum power limit',
                            tex_name='p_{mx}',
                            power=True,
                            unit='pu',
                            )

        self.v0 = NumParam(default=0.8, tex_name='v_0',
                           info='Lower limit of deadband for Vdroop response',
                           unit='pu', non_zero=True,
                           )
        self.v1 = NumParam(default=1.1, tex_name='v_1',
                           info='Upper limit of deadband for Vdroop response',
                           unit='pu', non_zero=True,
                           )

        self.dqdv = NumParam(default=-1.0, tex_name='dq/dv',
                             info='Q-V droop characteristics (negative)',
                             power=True, non_zero=True
                             )

        self.fdbd = NumParam(default=-0.017, tex_name='f_{dbd}',
                             info='frequency deviation deadband',
                             unit='Hz',
                             non_positive=True,
                             )

        # added on 11/14/2020: convert to system base pu
        self.ddn = NumParam(default=0.0, tex_name='D_{dn}',
                            info='Gain after f deadband',
                            unit='pu (MW)/Hz',
                            power=True,
                            non_negative=True,
                            )

        self.ialim = NumParam(default=1.3, tex_name='I_{alim}',
                              info='Apparent power limit',
                              current=True,
                              non_negative=True,
                              non_zero=True,
                              )

        self.vt0 = NumParam(default=0.88, tex_name='V_{t0}',
                            info='Voltage tripping response curve point 0',
                            unit='p.u.',
                            non_negative=True,
                            non_zero=True,
                            )

        self.vt1 = NumParam(default=0.90, tex_name='V_{t1}',
                            info='Voltage tripping response curve point 1',
                            unit='p.u.',
                            non_negative=True,
                            non_zero=True,
                            )

        self.vt2 = NumParam(default=1.1, tex_name='V_{t2}',
                            info='Voltage tripping response curve point 2',
                            unit='p.u.',
                            non_negative=True,
                            non_zero=True,
                            )

        self.vt3 = NumParam(default=1.2, tex_name='V_{t3}',
                            info='Voltage tripping response curve point 3',
                            unit='p.u.',
                            non_negative=True,
                            non_zero=True,
                            )

        self.vrflag = NumParam(default=0.0, tex_name='z_{VR}',
                               info='V-trip is latching (0) or self-resetting (0-1)',
                               )

        self.ft0 = NumParam(default=59.5, tex_name='f_{t0}',
                            info='Frequency tripping response curve point 0',
                            unit='Hz',
                            non_negative=True,
                            non_zero=True,
                            )

        self.ft1 = NumParam(default=59.7, tex_name='f_{t1}',
                            info='Frequency tripping response curve point 1',
                            unit='Hz',
                            non_negative=True,
                            non_zero=True,
                            )

        self.ft2 = NumParam(default=60.3, tex_name='f_{t2}',
                            info='Frequency tripping response curve point 2',
                            unit='Hz',
                            non_negative=True,
                            non_zero=True,
                            )

        self.ft3 = NumParam(default=60.5, tex_name='f_{t3}',
                            info='Frequency tripping response curve point 3',
                            unit='Hz',
                            non_negative=True,
                            non_zero=True,
                            )

        self.frflag = NumParam(default=0.0, tex_name='z_{FR}',
                               info='f-trip is latching (0) or self-resetting (0-1)',
                               )

        self.tip = NumParam(default=0.02, tex_name='T_{ip}',
                            info='Inverter active current lag time constant',
                            unit='s',
                            non_negative=True,
                            )

        self.tiq = NumParam(default=0.02, tex_name='T_{iq}',
                            info='Inverter reactive current lag time constant',
                            unit='s',
                            non_negative=True,
                            )

        self.gammap = NumParam(default=1.0, tex_name=r'\gamma_p',
                               info='Ratio of PVD1.pref0 w.r.t to that of static PV',
                               vrange='(0, 1]',
                               )

        self.gammaq = NumParam(default=1.0, tex_name=r'\gamma_q',
                               info='Ratio of PVD1.qref0 w.r.t to that of static PV',
                               vrange='(0, 1]',
                               )

        self.recflag = NumParam(default=1, tex_name=r'z_{rec}',
                                info='Enable flag for voltage and frequency recovery limiters',
                                vrange='(0, 1)',
                                )


class PVD1Model(Model):
    """
    Model implementation of PVD1.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'DG'

        self.config.add(OrderedDict((('plim', 1),
                                     )))

        self.config.add_extra('_help',
                              plim='enable input power limit check bound by [0, pmx]',
                              )
        self.config.add_extra('_tex',
                              plim='P_{lim}',
                              )
        self.config.add_extra('_alt',
                              plim=(0, 1),
                              )

        self.SWPQ = Switcher(u=self.pqflag, options=(0, 1), tex_name='SW_{PQ}', cache=True)

        self.buss = DataSelect(self.igreg, self.bus,
                               info='selected bus (bus or igreg)',
                               )

        self.busfreq = DeviceFinder(self.busf, link=self.buss, idx_name='bus',
                                    default_model='BusFreq')

        # --- initial values from power flow ---
        # a : bus voltage angle
        # v : bus voltage magnitude
        # p0s : active power from connected static PV generator
        # q0s : reactive power from connected static PV generator
        # pref0 : initial active power set point for the PVD1 device
        # qref0 : initial reactive power set point for the PVD1 device

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.buss, tex_name=r'\theta',
                          info='bus (or igreg) phase angle',
                          unit='rad.',
                          e_str='-Ipout_y * v * u',
                          ename='P',
                          tex_ename='P',
                          )

        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.buss, tex_name='V',
                          info='bus (or igreg) terminal voltage',
                          unit='p.u.',
                          e_str='-Iqout_y * v * u',
                          ename='Q',
                          tex_ename='Q',
                          )

        self.p0s = ExtService(model='StaticGen',
                              src='p',
                              indexer=self.gen,
                              tex_name='P_{0s}',
                              info='Initial P from static gen',
                              )
        self.q0s = ExtService(model='StaticGen',
                              src='q',
                              indexer=self.gen,
                              tex_name='Q_{0s}',
                              info='Initial Q from static gen',
                              )
        # --- calculate the initial P and Q for this distributed device ---
        self.pref0 = ConstService(v_str='gammap * p0s', tex_name='P_{ref0}',
                                  info='Initial P for the PVD1 device',
                                  )
        self.qref0 = ConstService(v_str='gammaq * q0s', tex_name='Q_{ref0}',
                                  info='Initial Q for the PVD1 device',
                                  )

        # frequency measurement variable `f`
        self.f = ExtAlgeb(model='FreqMeasurement', src='f', indexer=self.busfreq, export=False,
                          info='Bus frequency', unit='p.u.',
                          )

        self.fHz = Algeb(info='frequency in Hz',
                         v_str='fn * f', e_str='fn * f - fHz',
                         unit='Hz',
                         tex_name='f_{Hz}',
                         )

        # --- frequency branch ---
        self.FL1 = Limiter(u=self.fHz, lower=self.ft0, upper=self.ft1,
                           info='Under frequency comparer', no_warn=True,
                           allow_adjust=False,
                           )
        self.FL2 = Limiter(u=self.fHz, lower=self.ft2, upper=self.ft3,
                           info='Over frequency comparer', no_warn=True,
                           allow_adjust=False,
                           )

        self.Kft01 = ConstService(v_str='1/(ft1 - ft0)', tex_name='K_{ft01}')

        self.Ffl = Algeb(info='Coeff. for under frequency',
                         v_str='FL1_zi * Kft01 * (fHz - ft0) + FL1_zu',
                         e_str='FL1_zi * Kft01 * (fHz - ft0) + FL1_zu - Ffl',
                         tex_name='F_{fl}',
                         discrete=self.FL1,
                         )

        self.Kft23 = ConstService(v_str='1/(ft3 - ft2)', tex_name='K_{ft23}')

        self.Ffh = Algeb(info='Coeff. for over frequency',
                         v_str='FL2_zl + FL2_zi * (1 + Kft23 * (ft2 - fHz))',
                         e_str='FL2_zl + FL2_zi * (1 + Kft23 * (ft2 - fHz)) - Ffh',
                         tex_name='F_{fh}',
                         discrete=self.FL2,
                         )

        self.Fdev = Algeb(info='Frequency deviation',
                          v_str='fn - fHz', e_str='fn - fHz - Fdev',
                          unit='Hz', tex_name='f_{dev}',
                          )

        self.DB = DeadBand1(u=self.Fdev, center=0.0, lower=self.fdbd, upper=0.0, gain=self.ddn,
                            info='frequency deviation deadband with gain',
                            )  # outputs   `Pdrp`
        self.DB.db.no_warn = True

        # --- Voltage flags ---
        self.VL1 = Limiter(u=self.v, lower=self.vt0, upper=self.vt1,
                           info='Under voltage comparer', no_warn=True,
                           allow_adjust=False,
                           )
        self.VL2 = Limiter(u=self.v, lower=self.vt2, upper=self.vt3,
                           info='Over voltage comparer', no_warn=True,
                           allow_adjust=False,
                           )

        self.Kvt01 = ConstService(v_str='1/(vt1 - vt0)', tex_name='K_{vt01}')

        self.Fvl = Algeb(info='Coeff. for under voltage',
                         v_str='VL1_zi * Kvt01 * (v - vt0) + VL1_zu',
                         e_str='VL1_zi * Kvt01 * (v - vt0) + VL1_zu - Fvl',
                         tex_name='F_{vl}',
                         discrete=self.VL1,
                         )

        self.Kvt23 = ConstService(v_str='1/(vt3 - vt2)', tex_name='K_{vt23}')

        self.Fvh = Algeb(info='Coeff. for over voltage',
                         v_str='VL2_zl + VL2_zi * (1 + Kvt23 * (vt2 - v))',
                         e_str='VL2_zl + VL2_zi * (1 + Kvt23 * (vt2 - v)) - Fvh',
                         tex_name='F_{vh}',
                         discrete=self.VL2,
                         )
        # --- sensed voltage with lower limit of 0.01 ---
        self.VLo = Limiter(u=self.v, lower=0.01, upper=999, no_upper=True,
                           info='Voltage lower limit (0.01) flag',
                           )

        self.vp = Algeb(tex_name='V_p',
                        info='Sensed positive voltage',
                        v_str='v * VLo_zi + 0.01 * VLo_zl',
                        e_str='v * VLo_zi + 0.01 * VLo_zl - vp',
                        )

        self.Pext0 = ConstService(info='External additional signal added to Pext',
                                  tex_name='P_{ext0}',
                                  v_str='0',
                                  )

        self.Pext = Algeb(tex_name='P_{ext}',
                          info='External power signal (for AGC)',
                          v_str='u * Pext0',
                          e_str='u * Pext0 - Pext'
                          )

        self.Pref = Algeb(tex_name='P_{ref}',
                          info='Reference power signal (for scheduling setpoint)',
                          v_str='u * pref0',
                          e_str='u * pref0 - Pref'
                          )

        self.Psum = Algeb(tex_name='P_{tot}',
                          info='Sum of P signals',
                          v_str='u * (Pext + Pref + DB_y)',
                          e_str='u * (Pext + Pref + DB_y) - Psum',
                          )  # `DB_y` is `Pdrp` (f droop)

        self.PHL = Limiter(u=self.Psum, lower=0.0, upper=self.pmx,
                           enable=self.config.plim,
                           info='limiter for Psum in [0, pmx]',
                           allow_adjust=False,
                           )

        self.Vcomp = VarService(v_str='abs(v*exp(1j*a) + (1j * xc) * (Ipout_y + 1j * Iqout_y))',
                                info='Voltage before Xc compensation',
                                tex_name='V_{comp}'
                                )

        self.Vqu = ConstService(v_str='v1 - (qref0 - qmn) / dqdv',
                                info='Upper voltage bound => qmx',
                                tex_name='V_{qu}',
                                )

        self.Vql = ConstService(v_str='v0 + (qmx - qref0) / dqdv',
                                info='Lower voltage bound => qmn',
                                tex_name='V_{ql}',
                                )

        self.VQ1 = Limiter(u=self.Vcomp, lower=self.Vql, upper=self.v0,
                           info='Under voltage comparer for Q droop',
                           no_warn=True,
                           allow_adjust=False,
                           )

        self.VQ2 = Limiter(u=self.Vcomp, lower=self.v1, upper=self.Vqu,
                           info='Over voltage comparer for Q droop',
                           no_warn=True,
                           allow_adjust=False,
                           )

        Qdrp = 'u * VQ1_zl * qmx + VQ2_zu * qmn + ' \
               'u * VQ1_zi * (qmx + dqdv *(Vqu - Vcomp)) + ' \
               'u * VQ2_zi * (dqdv * (v1 - Vcomp)) '

        self.Qdrp = Algeb(tex_name='Q_{drp}',
                          info='External power signal (for AGC)',
                          v_str=Qdrp,
                          e_str=f'{Qdrp} - Qdrp',
                          discrete=(self.VQ1, self.VQ2),
                          )

        self.Qref = Algeb(tex_name=r'Q_{ref}',
                          info='Reference power signal (for scheduling setpoint)',
                          v_str='u * qref0',
                          e_str='u * qref0 - Qref'
                          )

        self.Qsum = Algeb(tex_name=r'Q_{tot}',
                          info='Sum of Q signals',
                          v_str=f'u * (qref0 + {Qdrp})',
                          e_str='u * (Qref + Qdrp) - Qsum',
                          discrete=(self.VQ1, self.VQ2),
                          )

        self.Ipul = Algeb(info='Ipcmd before Ip hard limit',
                          v_str='(Psum * PHL_zi + pmx * PHL_zu) / vp',
                          e_str='(Psum * PHL_zi + pmx * PHL_zu) / vp - Ipul',
                          tex_name='I_{p,ul}',
                          )

        self.Iqul = Algeb(info='Iqcmd before Iq hard limit',
                          v_str='Qsum / vp',
                          e_str='Qsum / vp - Iqul',
                          tex_name='I_{q,ul}',
                          )

        # --- Ipmax, Iqmax and Iqmin ---
        Ipmaxsq = "(Piecewise((0, Le(ialim**2 - Iqcmd_y**2, 0)), ((ialim**2 - Iqcmd_y ** 2), True)))"
        Ipmaxsq0 = "(Piecewise((0, Le(ialim**2 - (u*qref0/v)**2, 0)), ((ialim**2 - (u*qref0/v) ** 2), True)))"
        self.Ipmaxsq = VarService(v_str=Ipmaxsq, tex_name='I_{pmax}^2')
        self.Ipmaxsq0 = ConstService(v_str=Ipmaxsq0, tex_name='I_{pmax0}^2')

        self.Ipmax = Algeb(v_str='(SWPQ_s1 * ialim + SWPQ_s0 * sqrt(Ipmaxsq0))',
                           e_str='(SWPQ_s1 * ialim + SWPQ_s0 * sqrt(Ipmaxsq)) - Ipmax',
                           tex_name='I_{pmax}',
                           info='Upper limit of Ip',
                           )

        Iqmaxsq = "(Piecewise((0, Le(ialim**2 - Ipcmd_y**2, 0)), ((ialim**2 - Ipcmd_y ** 2), True)))"
        Iqmaxsq0 = "(Piecewise((0, Le(ialim**2 - (u*pref0/v)**2, 0)), ((ialim**2 - (u*pref0/v) ** 2), True)))"
        self.Iqmaxsq = VarService(v_str=Iqmaxsq, tex_name='I_{qmax}^2')
        self.Iqmaxsq0 = ConstService(v_str=Iqmaxsq0, tex_name='I_{qmax0}^2')

        self.Iqmax = Algeb(v_str='SWPQ_s0 * ialim + SWPQ_s1 * sqrt(Iqmaxsq0)',
                           e_str='SWPQ_s0 * ialim + SWPQ_s1 * sqrt(Iqmaxsq) - Iqmax',
                           tex_name='I_{qmax}',
                           info='Upper limit of Iq',
                           )

        # TODO: set option whether to use degrading gain
        # --- `Ipcmd` and `Iqcmd` ---
        self.Ipcmd = GainLimiter(u=self.Ipul,
                                 K=1, R='Fvl * Fvh * Ffl * Ffh * recflag + 1 * (1 - recflag)',
                                 lower=0, upper=self.Ipmax,
                                 info='Ip with limiter and coeff.',
                                 tex_name='I^{pcmd}',
                                 )

        # disable auto limit adjustment because it is not supported for limits that are variables
        self.Ipcmd.lim.allow_adjust = False

        self.Iqcmd = GainLimiter(u=self.Iqul,
                                 K=1, R='Fvl * Fvh * Ffl * Ffh * recflag + 1 * (1 - recflag)',
                                 lower=self.Iqmax, sign_lower=-1,
                                 upper=self.Iqmax,
                                 info='Iq with limiter and coeff.',
                                 tex_name='I^{qcmd}',
                                 )
        self.Iqcmd.lim.allow_adjust = False

        self.Ipout = Lag(u=self.Ipcmd_y, T=self.tip, K=1.0,
                         info='Output Ip filter',
                         )

        self.Iqout = Lag(u=self.Iqcmd_y, T=self.tiq, K=1.0,
                         info='Output Iq filter',
                         )

    def v_numeric(self, **kwargs):
        """
        Disable the corresponding `StaticGen`s.
        """
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class PVD1(PVD1Data, PVD1Model):
    """
    WECC Distributed PV model.

    Device power rating is specified in `Sn`.
    Output currents are named `Ipout_y` and `Iqout_y`.
    Output power can be computed as ``Pe = Ipout_y * v`` and
    ``Qe = Iqout_y * v``.

    Frequency tripping response points `ft0`, `ft1`, `ft2`, and `ft3`
    must be monotinically increasing.
    Same rule applies to the voltage tripping response points
    `vt0`, `vt1`, `vt2`, and `vt3`.
    The program does not check these values, and the user
    is responsible for the parameter validity.

    Frequency and voltage recovery latching is yet to be implemented.

    Modifications to the active and reactive power references,
    typically by an external scheduling program, should
    write to `pref0.v` and `qref0.v` in place.
    AGC signals should write to `pext0.v` in place.

    Maximum power limit `pmx` can be disabled by editing the configuration
    file by setting `plim=0`. It cannot be modified in runtime.

    Reference:
    [1] ESIG, WECC Distributed and Small PV Plants Generic Model (PVD1), [Online],
    Available:

    https://www.esig.energy/wiki-main-page/wecc-distributed-and-small-pv-plants-generic-model-pvd1/
    """

    def __init__(self, system, config):
        PVD1Data.__init__(self)
        PVD1Model.__init__(self, system, config)
