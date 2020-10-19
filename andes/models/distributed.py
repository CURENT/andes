"""
Distributed energy resource models.
"""

from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam
from andes.core.block import Lag, GainLimiter, DeadBand1  # NOQA
from andes.core.var import ExtAlgeb, ExtState, Algeb, State  # NOQA

from andes.core.service import ConstService, ExtService, FlagValue, DataSelect, DeviceFinder  # NOQA
from andes.core.service import VarService, Replace  # NOQA
from andes.core.service import NumSelect  # NOQA
from andes.core.discrete import Switcher, Limiter  # NOQA


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

        self.v0 = NumParam(default=0.0, tex_name='v_0',
                           info='Lower limit of deadband for Vdroop response',
                           unit='pu',
                           )
        self.v1 = NumParam(default=0.0, tex_name='v_1',
                           info='Upper limit of deadband for Vdroop response',
                           unit='pu',
                           )

        self.dqdv = NumParam(default=0.0, tex_name='dq/dv',
                             info='Q-V droop characteristics',
                             power=True,
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


class PVD1Model(Model):
    """
    Model implementation of PVD1.
    """
    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'DG'

        self.buss = DataSelect(self.igreg, self.bus,
                               info='selected bus (bus or igreg)',
                               )

        self.busfreq = DeviceFinder(self.busf, link=self.buss, idx_name='bus')

        # initial voltages from selected bus
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.buss, tex_name='V',
                          info='bus (or igreg) terminal voltage',
                          unit='p.u.',
                          )

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.buss, tex_name=r'\theta',
                          info='bus (or igreg) phase angle',
                          unit='rad.',
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
        # frequency measurement variable `f`
        self.f = ExtAlgeb(model='FreqMeasurement', src='f', indexer=self.busfreq, export=False,
                          info='Bus frequency', unit='p.u.',
                          )

        self.fHz = Algeb(info='frequency in Hz',
                         v_str='fn * f', e_str='fn * f - fHz',
                         unit='Hz',
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

class PVD1(PVD1Data, PVD1Model):
    """
    Distributed PV model. (TODO: work in progress)

    Power rating specified in `Sn`.

    Reference: ESIG, WECC Distributed and Small PV Plants Generic Model (PVD1), [Online],
    Available: https://www.esig.energy/wiki-main-page/wecc-distributed-and-small-pv-plants-generic-model-pvd1/
    """
    def __init__(self, system, config):
        PVD1Data.__init__(self)
        PVD1Model.__init__(self, system, config)
