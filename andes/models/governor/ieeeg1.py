import numpy as np

from andes.core import IdxParam, NumParam, ConstService, ExtParam, ExtService, ExtAlgeb, Algeb, LeadLag, \
    HardLimiter, Lag
from andes.core.block import IntegratorAntiWindup
from andes.core.service import InitChecker, PostInitService, ParamCalc, NumSelect, FlagValue
from andes.models.governor.tgbase import TGBaseData, TGBase


class IEEEG1Data(TGBaseData):

    def __init__(self):
        TGBaseData.__init__(self)

        self.syn2 = IdxParam(model='SynGen',
                             info='Optional SynGen idx',
                             )
        self.K = NumParam(default=20, tex_name='K',
                          info='Gain (1/R) in mach. base',
                          unit='p.u. (power)',
                          power=True,
                          vrange=(5, 30),
                          )
        self.T1 = NumParam(default=1, tex_name='T_1',
                           info='Gov. lag time const.',
                           vrange=(0, 5),
                           )
        self.T2 = NumParam(default=1, tex_name='T_2',
                           info='Gov. lead time const.',
                           vrange=(0, 10),
                           )
        self.T3 = NumParam(default=0.1, tex_name='T_3',
                           info='Valve controller time const.',
                           vrange=(0.04, 1),
                           )
        # "UO" is "U" and capitalized "o" character
        self.UO = NumParam(default=0.1, tex_name='U_o',
                           info='Max. valve opening rate',
                           unit='p.u./sec', vrange=(0.01, 0.3),
                           )
        self.UC = NumParam(default=-0.1, tex_name='U_c',
                           info='Max. valve closing rate',
                           unit='p.u./sec', vrange=(-0.3, 0),
                           )
        self.PMAX = NumParam(default=5, tex_name='P_{MAX}',
                             info='Max. turbine power',
                             vrange=(0.5, 2), power=True,
                             )
        self.PMIN = NumParam(default=0., tex_name='P_{MIN}',
                             info='Min. turbine power',
                             vrange=(0.0, 0.5), power=True,
                             )

        self.T4 = NumParam(default=0.4, tex_name='T_4',
                           info='Inlet piping/steam bowl time constant',
                           vrange=(0, 1.0),
                           )
        self.K1 = NumParam(default=0.5, tex_name='K_1',
                           info='Fraction of power from HP',
                           vrange=(0, 1.0),
                           )
        self.K2 = NumParam(default=0, tex_name='K_2',
                           info='Fraction of power from LP',
                           vrange=(0,),
                           )
        self.T5 = NumParam(default=8, tex_name='T_5',
                           info='Time constant of 2nd boiler pass',
                           vrange=(0, 10),
                           )
        self.K3 = NumParam(default=0.5, tex_name='K_3',
                           info='Fraction of HP shaft power after 2nd boiler pass',
                           vrange=(0, 0.5),
                           )
        self.K4 = NumParam(default=0.0, tex_name='K_4',
                           info='Fraction of LP shaft power after 2nd boiler pass',
                           vrange=(0,),
                           )

        self.T6 = NumParam(default=0.5, tex_name='T_6',
                           info='Time constant of 3rd boiler pass',
                           vrange=(0, 10),
                           )
        self.K5 = NumParam(default=0.0, tex_name='K_5',
                           info='Fraction of HP shaft power after 3rd boiler pass',
                           vrange=(0, 0.35),
                           )
        self.K6 = NumParam(default=0, tex_name='K_6',
                           info='Fraction of LP shaft power after 3rd boiler pass',
                           vrange=(0, 0.55),
                           )

        self.T7 = NumParam(default=0.05, tex_name='T_7',
                           info='Time constant of 4th boiler pass',
                           vrange=(0, 10),
                           )
        self.K7 = NumParam(default=0, tex_name='K_7',
                           info='Fraction of HP shaft power after 4th boiler pass',
                           vrange=(0, 0.3),
                           )
        self.K8 = NumParam(default=0, tex_name='K_8',
                           info='Fraction of LP shaft power after 4th boiler pass',
                           vrange=(0, 0.3),
                           )


class IEEEG1Model(TGBase):
    def __init__(self, system, config):
        TGBase.__init__(self, system, config, add_sn=False)

        # check if K1-K8 sums up to 1
        self._sumK18 = ConstService(v_str='K1+K2+K3+K4+K5+K6+K7+K8',
                                    info='summation of K1-K8',
                                    tex_name=r"\sum_{i=1}^8 K_i"
                                    )
        self._K18c1 = InitChecker(u=self._sumK18,
                                  info='summation of K1-K8 and 1.0',
                                  equal=1,
                                  )

        # check if  `tm0 * (K2 + k4 + K6 + K8) = tm02 *(K1 + K3 + K5 + K7)
        self._tm0K2 = PostInitService(info='mul of tm0 and (K2+K4+K6+K8)',
                                      v_str='zsyn2*tm0*(K2+K4+K6+K8)',
                                      )
        self._tm02K1 = PostInitService(info='mul of tm02 and (K1+K3+K5+K6)',
                                       v_str='tm02*(K1+K3+K5+K7)',
                                       )
        self._Pc = InitChecker(u=self._tm0K2,
                               info='proportionality of tm0 and tm02',
                               equal=self._tm02K1,
                               )

        self.Sg2 = ExtParam(src='Sn',
                            model='SynGen',
                            indexer=self.syn2,
                            allow_none=True,
                            default=0.0,
                            tex_name='S_{n2}',
                            info='Rated power of Syn2',
                            unit='MVA',
                            export=False,
                            )
        self.Sg12 = ParamCalc(self.Sg, self.Sg2, func=np.add,
                              tex_name="S_{g12}",
                              info='Sum of generator power ratings',
                              )
        self.Sn = NumSelect(self.Tn,
                            fallback=self.Sg12,
                            tex_name='S_n',
                            info='Turbine or Gen rating',
                            )

        self.zsyn2 = FlagValue(self.syn2,
                               value=None,
                               tex_name='z_{syn2}',
                               info='Exist flags for syn2',
                               )

        self.tm02 = ExtService(src='tm',
                               model='SynGen',
                               indexer=self.syn2,
                               tex_name=r'\tau_{m02}',
                               info='Initial mechanical input of syn2',
                               allow_none=True,
                               default=0.0,
                               )
        self.tm012 = ConstService(info='total turbine power',
                                  v_str='tm0 + tm02',
                                  )

        self.tm2 = ExtAlgeb(src='tm',
                            model='SynGen',
                            indexer=self.syn2,
                            allow_none=True,
                            tex_name=r'\tau_{m2}',
                            e_str='zsyn2 * ue * (PLP - tm02)',
                            info='Mechanical power to syn2',
                            )

        self.wd = Algeb(info='Generator under speed',
                        unit='p.u.',
                        tex_name=r'\omega_{dev}',
                        v_str='0',
                        e_str='ue * (wref - omega) - wd',
                        )

        self.LL = LeadLag(u=self.wd,
                          T1=self.T2,
                          T2=self.T1,
                          K=self.K,
                          info='Signal conditioning for wd',
                          )

        # `P0` == `tm0`
        self.vs = Algeb(info='Valve speed',
                        tex_name='V_s',
                        v_str='0',
                        e_str='ue * (LL_y + tm012 + paux - IAW_y) / T3 - vs',
                        )

        self.HL = HardLimiter(u=self.vs,
                              lower=self.UC,
                              upper=self.UO,
                              info='Limiter on valve acceleration',
                              )

        self.vsl = Algeb(info='Valve move speed after limiter',
                         tex_name='V_{sl}',
                         v_str='vs * HL_zi + UC * HL_zl + UO * HL_zu',
                         e_str='vs * HL_zi + UC * HL_zl + UO * HL_zu - vsl',
                         )

        self.IAW = IntegratorAntiWindup(u=self.vsl,
                                        T=1,
                                        K=1,
                                        y0=self.tm012,
                                        lower=self.PMIN,
                                        upper=self.PMAX,
                                        info='Valve position integrator',
                                        )

        self.L4 = Lag(u=self.IAW_y, T=self.T4, K=1,
                      info='first process',)

        self.L5 = Lag(u=self.L4_y, T=self.T5, K=1,
                      info='second (reheat) process',
                      )

        self.L6 = Lag(u=self.L5_y, T=self.T6, K=1,
                      info='third process',
                      )

        self.L7 = Lag(u=self.L6_y, T=self.T7, K=1,
                      info='fourth (second reheat) process',
                      )

        self.PHP = Algeb(info='HP output',
                         tex_name='P_{HP}',
                         v_str='ue * (K1*L4_y + K3*L5_y + K5*L6_y + K7*L7_y)',
                         e_str='ue * (K1*L4_y + K3*L5_y + K5*L6_y + K7*L7_y) - PHP',
                         )

        self.PLP = Algeb(info='LP output',
                         tex_name='P_{LP}',
                         v_str='ue * (K2*L4_y + K4*L5_y + K6*L6_y + K8*L7_y)',
                         e_str='ue * (K2*L4_y + K4*L5_y + K6*L6_y + K8*L7_y) - PLP',
                         )

        self.pout.e_str = 'ue * PHP - pout'


class IEEEG1(IEEEG1Data, IEEEG1Model):
    """
    IEEE Type 1 Speed-Governing Model.

    If only one generator is connected, its `idx` must
    be given to `syn`, and `syn2` must be left blank.
    Each generator must provide data in its `Sn` base.

    `syn` is connected to the high-pressure output (PHP)
    and the optional `syn2` is connected to the low-
    pressure output (PLP).

    The speed deviation of generator 1 (syn) is measured.
    If the turbine rating `Tn` is not specified, the sum
    of `Sn` of all connected generators will be used.

    Normally, K1 + K2 + ... + K8 = 1.0.
    If the second generator is not connected,
    K1 + K3 + K5 + K7 = 1, and K2 + K4 + K6 + K8 = 0.

    IEEEG1 does not yet support the change of reference
    (scheduling).
    """

    def __init__(self, system, config):
        IEEEG1Data.__init__(self)
        IEEEG1Model.__init__(self, system, config)
