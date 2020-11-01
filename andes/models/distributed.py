"""
Distributed energy resource models.
"""

from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam
from andes.core.block import Lag, DeadBand1, LimiterGain
from andes.core.var import ExtAlgeb, Algeb

from andes.core.service import ConstService, ExtService, VarService
from andes.core.service import DataSelect, DeviceFinder
from andes.core.discrete import Switcher, Limiter


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
                           info='Model MVA base',
                           unit='MVA',
                           )

        self.fn = NumParam(default=60.0, tex_name='f_n',
                           info='nominal frequency',
                           unit='Hz',
                           )

        self.busf = IdxParam(info='Optional BusFreq idx',
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
                            )

        self.qmn = NumParam(default=-0.33, tex_name='q_{mn}',
                            info='Min. reactive power command',
                            power=True,
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

        self.fdbd = NumParam(default=-0.01, tex_name='f_{dbd}',
                             info='frequency deviation deadband',
                             )

        self.ddn = NumParam(default=0.0, tex_name='D_{dn}',
                            info='Gain after f deadband',
                            )

        self.ialim = NumParam(default=1.3, tex_name='I_{alim}',
                              info='Apparent power limit',
                              current=True,
                              )

        self.vt0 = NumParam(default=0.88, tex_name='V_{t0}',
                            info='Voltage tripping response curve point 0',
                            )

        self.vt1 = NumParam(default=0.90, tex_name='V_{t1}',
                            info='Voltage tripping response curve point 1',
                            )

        self.vt2 = NumParam(default=1.1, tex_name='V_{t2}',
                            info='Voltage tripping response curve point 2',
                            )

        self.vt3 = NumParam(default=1.2, tex_name='V_{t3}',
                            info='Voltage tripping response curve point 3',
                            )

        self.vrflag = NumParam(default=0.0, tex_name='z_{VR}',
                               info='Voltage tripping is latching (0) or partially self-resetting (0-1)',
                               )

        self.ft0 = NumParam(default=59.5, tex_name='f_{t0}',
                            info='Frequency tripping response curve point 0',
                            )

        self.ft1 = NumParam(default=59.7, tex_name='f_{t1}',
                            info='Frequency tripping response curve point 1',
                            )

        self.ft2 = NumParam(default=60.3, tex_name='f_{t2}',
                            info='Frequency tripping response curve point 2',
                            )

        self.ft3 = NumParam(default=60.5, tex_name='f_{t3}',
                            info='Frequency tripping response curve point 3',
                            )

        self.frflag = NumParam(default=0.0, tex_name='z_{FR}',
                               info='Frequency tripping is latching (0) or partially self-resetting (0-1)',
                               )

        self.tip = NumParam(default=0.02, tex_name='T_{ip}',
                            info='Inverter active current lag time constant',
                            unit='s',
                            )

        self.tiq = NumParam(default=0.02, tex_name='T_{iq}',
                            info='Inverter reactive current lag time constant',
                            unit='s',
                            )

        self.gammap = NumParam(default=1.0, tex_name=r'\gamma_p',
                               info='Ratio of P from PVD1 w.r.t to that from PV generator',
                               vrange='(0, 1]',
                               )

        self.gammaq = NumParam(default=1.0, tex_name=r'\gamma_q',
                               info='Ratio of Q from PVD1 w.r.t to that from PV generator',
                               vrange='(0, 1]',
                               )


class PVD1Model(Model):
    """
    Model implementation of PVD1.
    """
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'DG'

        self.SWPQ = Switcher(u=self.pqflag, options=(0, 1), tex_name='SW_{PQ}', cache=True)

        self.buss = DataSelect(self.igreg, self.bus,
                               info='selected bus (bus or igreg)',
                               )

        self.busfreq = DeviceFinder(self.busf, link=self.buss, idx_name='bus')

        # --- initial values from power flow ---
        # v : bus voltage magnitude
        # a : bus voltage angle
        # p0s : active power from connected static PV generator
        # q0s : reactive power from connected static PV generator

        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.buss, tex_name='V',
                          info='bus (or igreg) terminal voltage',
                          unit='p.u.',
                          e_str='-Iqout_y * v * u',
                          )

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.buss, tex_name=r'\theta',
                          info='bus (or igreg) phase angle',
                          unit='rad.',
                          e_str='-Ipout_y * v * u',
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
        self.p0 = ConstService(v_str='gammap * p0s', tex_name='P_0',
                               info='Initial P for the PVD1 device',
                               )
        self.q0 = ConstService(v_str='gammaq * q0s', tex_name='Q_0',
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
                           )
        self.FL2 = Limiter(u=self.fHz, lower=self.ft2, upper=self.ft3,
                           info='Over frequency comparer', no_warn=True,
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

        self.Dfdbd = ConstService(v_str='fdbd * ddn', info='Deadband lower limit after gain',
                                  tex_name='D_{fdbd}',
                                  )
        self.DB = DeadBand1(u=self.Fdev, center=0.0, lower=self.Dfdbd, upper=0.0,
                            info='frequency deviation deadband with gain',
                            )  # outputs   `Pdrp`
        self.DB.db.no_warn = True

        # --- Voltage flags ---
        self.VL1 = Limiter(u=self.v, lower=self.vt0, upper=self.vt1,
                           info='Under voltage comparer', no_warn=True,
                           )
        self.VL2 = Limiter(u=self.v, lower=self.vt2, upper=self.vt3,
                           info='Over voltage comparer', no_warn=True,
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

        self.Pext = Algeb(tex_name='P_{ext}',
                          info='External power signal',
                          v_str='0',
                          e_str='0 - Pext'
                          )

        self.Psum = Algeb(tex_name='P_{tot}',
                          info='Sum of P signals',
                          v_str='Pext + p0 + DB_y',
                          e_str='Pext + p0 + DB_y - Psum',
                          )  # `p0` is the initial `Pref`, and `DB_y` is `Pdrp` (f droop)

        self.Vcomp = VarService(v_str='abs(v*exp(1j*a) + (1j * xc) * (Ipout_y + 1j * Iqout_y))',
                                info='Voltage before Xc compensation',
                                tex_name='V_{comp}'
                                )

        self.Vqu = ConstService(v_str='v1 - (q0 - qmn) / dqdv',
                                info='Upper voltage bound => qmx',
                                tex_name='V_{qu}',
                                )

        self.Vql = ConstService(v_str='v0 + (qmx - q0) / dqdv',
                                info='Lower voltage bound => qmn',
                                tex_name='V_{ql}',
                                )

        self.VQ1 = Limiter(u=self.Vcomp, lower=self.Vql, upper=self.v0,
                           info='Under voltage comparer for Q droop',
                           no_warn=True,
                           )

        self.VQ2 = Limiter(u=self.Vcomp, lower=self.v1, upper=self.Vqu,
                           info='Over voltage comparer for Q droop',
                           no_warn=True,
                           )

        Qsum = 'VQ1_zl * qmx + VQ2_zu * qmn + ' \
               'VQ1_zi * (qmx + dqdv *(Vqu - Vcomp)) + ' \
               'VQ2_zi * (q0 + dqdv * (v1 - Vcomp)) + ' \
               'q0'

        self.Qsum = Algeb(info='Total Q (droop + initial)',
                          v_str=Qsum,
                          e_str=f'{Qsum} - Qsum',
                          tex_name='Q_{sum}',
                          discrete=(self.VQ1, self.VQ2),
                          )

        self.Ipul = Algeb(info='Ipcmd before hard limit',
                          v_str='Psum / vp',
                          e_str='Psum / vp - Ipul',
                          tex_name='I_{p,ul}',
                          )

        self.Iqul = Algeb(info='Iqcmd before hard limit',
                          v_str='Qsum / vp',
                          e_str='Qsum / vp - Iqul',
                          tex_name='I_{q,ul}',
                          )

        # --- Ipmax, Iqmax and Iqmin ---
        Ipmaxsq = "(Piecewise((0, Le(ialim**2 - Iqcmd_y**2, 0)), ((ialim**2 - Iqcmd_y ** 2), True)))"
        Ipmaxsq0 = "(Piecewise((0, Le(ialim**2 - (q0 / v)**2, 0)), ((ialim**2 - (q0 / v) ** 2), True)))"
        self.Ipmaxsq = VarService(v_str=Ipmaxsq, tex_name='I_{pmax}^2')
        self.Ipmaxsq0 = ConstService(v_str=Ipmaxsq0, tex_name='I_{pmax0}^2')

        self.Ipmax = Algeb(v_str='SWPQ_s1 * ialim + SWPQ_s0 * sqrt(Ipmaxsq0)',
                           e_str='SWPQ_s1 * ialim + SWPQ_s0 * sqrt(Ipmaxsq) - Ipmax',
                           tex_name='I_{pmax}',
                           )

        Iqmaxsq = "(Piecewise((0, Le(ialim**2 - Ipcmd_y**2, 0)), ((ialim**2 - Ipcmd_y ** 2), True)))"
        Iqmaxsq0 = "(Piecewise((0, Le(ialim**2 - (p0 / v)**2, 0)), ((ialim**2 - (p0 / v) ** 2), True)))"
        self.Iqmaxsq = VarService(v_str=Iqmaxsq, tex_name='I_{qmax}^2')
        self.Iqmaxsq0 = ConstService(v_str=Iqmaxsq0, tex_name='I_{qmax0}^2')

        self.Iqmax = Algeb(v_str='SWPQ_s0 * ialim + SWPQ_s1 * sqrt(Iqmaxsq0)',
                           e_str='SWPQ_s0 * ialim + SWPQ_s1 * sqrt(Iqmaxsq) - Iqmax',
                           tex_name='I_{qmax}',
                           )

        self.Iqmin = VarService(v_str='-Iqmax', tex_name='I_{qmin}')

        # --- Ipcmd, Iqcmd ---

        self.Ipcmd = LimiterGain(u=self.Ipul, K='Fvl * Fvh * Ffl * Ffh',
                                 lower=0.0, upper=self.Ipmax,
                                 info='Ip with limiter and coeff.',
                                 tex_name='I^{pcmd}',
                                 )

        self.Iqcmd = LimiterGain(u=self.Iqul, K='Fvl * Fvh * Ffl * Ffh',
                                 lower=self.Iqmin, upper=self.Iqmax,
                                 info='Iq with limiter and coeff.',
                                 tex_name='I^{qcmd}',
                                 )

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

    Power rating specified in `Sn`.
    Frequency and voltage recovery latching has not been implemented.

    Reference:
    [1] ESIG, WECC Distributed and Small PV Plants Generic Model (PVD1), [Online],
    Available: https://www.esig.energy/wiki-main-page/wecc-distributed-and-small-pv-plants-generic-model-pvd1/
    """
    def __init__(self, system, config):
        PVD1Data.__init__(self)
        PVD1Model.__init__(self, system, config)
