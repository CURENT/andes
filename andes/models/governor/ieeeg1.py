import numpy as np

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        HardLimiter, IdxParam, Lag, LeadLag, NumParam,)
from andes.core.block import IntegratorAntiWindup
from andes.core.service import (FlagValue, InitChecker, NumSelect, ParamCalc,
                                PostInitService,)
from andes.models.governor.tgbase import TGBase, TGBaseData


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

        self._Kcoeff = ConstService(v_str='1/_sumK18',
                                    info='normalization factor to be multiplied to K1-K8',
                                    tex_name='K_{coeff}',
                                    )
        self.K1n = ConstService(v_str='K1 * _Kcoeff',
                                info='normalized K1',
                                tex_name='K_{1n}',
                                )
        self.K2n = ConstService(v_str='K2 * _Kcoeff',
                                info='normalized K2',
                                tex_name='K_{2n}',
                                )
        self.K3n = ConstService(v_str='K3 * _Kcoeff',
                                info='normalized K3',
                                tex_name='K_{3n}',
                                )
        self.K4n = ConstService(v_str='K4 * _Kcoeff',
                                info='normalized K4',
                                tex_name='K_{4n}',
                                )
        self.K5n = ConstService(v_str='K5 * _Kcoeff',
                                info='normalized K5',
                                tex_name='K_{5n}',
                                )
        self.K6n = ConstService(v_str='K6 * _Kcoeff',
                                info='normalized K6',
                                tex_name='K_{6n}',
                                )
        self.K7n = ConstService(v_str='K7 * _Kcoeff',
                                info='normalized K7',
                                tex_name='K_{7n}',
                                )
        self.K8n = ConstService(v_str='K8 * _Kcoeff',
                                info='normalized K8',
                                tex_name='K_{8n}',
                                )

        # check if  `tm0 * (K2 + k4 + K6 + K8) = tm02 *(K1 + K3 + K5 + K7)
        self._tm0K2 = PostInitService(info='mul of tm0 and (K2n+K4n+K6n+K8n)',
                                      v_str='zsyn2*tm0*(K2n + K4n + K6n + K8n)',
                                      )
        self._tm02K1 = PostInitService(info='mul of tm02 and (K1n+K3n+K5n+K7n)',
                                       v_str='tm02*(K1n + K3n + K5n + K7n)',
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

        # Note: the following applies `zsyn2` to disable the syn2
        self.tm2 = ExtAlgeb(src='tm',
                            model='SynGen',
                            indexer=self.syn2,
                            allow_none=True,
                            tex_name=r'\tau_{m2}',
                            e_str='zsyn2 * ue * (PLP - tm02)',
                            info='Mechanical power to syn2',
                            ename='tm2',
                            tex_ename=r'\tau_{m2}',
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
                              info='Limiter on valve speed',
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
                         v_str='ue * (K1n*L4_y + K3n*L5_y + K5n*L6_y + K7n*L7_y)',
                         e_str='ue * (K1n*L4_y + K3n*L5_y + K5n*L6_y + K7n*L7_y) - PHP',
                         )

        self.PLP = Algeb(info='LP output',
                         tex_name='P_{LP}',
                         v_str='ue * (K2n*L4_y + K4n*L5_y + K6n*L6_y + K8n*L7_y)',
                         e_str='ue * (K2n*L4_y + K4n*L5_y + K6n*L6_y + K8n*L7_y) - PLP',
                         )

        self.pout.e_str = 'ue * PHP - pout'


class IEEEG1(IEEEG1Data, IEEEG1Model):
    """
    IEEE Type 1 Speed-Governing Model.

    If only one generator is connected, its `idx` must be given to `syn`, and
    `syn2` must be left blank. Each generator must provide data in its `Sn`
    base.

    `syn` is connected to the high-pressure output (PHP) and the optional `syn2`
    is connected to the low- pressure output (PLP).

    The speed deviation of generator 1 (syn) is measured. If the turbine rating
    `Tn` is not specified, the sum of `Sn` of all connected generators will be
    used.

    Normally, K1 + K2 + ... + K8 = 1.0. If the second generator is not
    connected, K1 + K3 + K5 + K7 = 1, and K2 + K4 + K6 + K8 = 0. If K1 to K8 do
    not sum up to 1.0, they will be normalized. The normalized parameters are
    called ``K1n`` through ``K8n``.

    If initialization error occurs for variable ``vs``, it is due to the limits
    ``PMIN`` and ``PMAX``.

    IEEEG1 does not yet support the change of reference (scheduling).
    """

    def __init__(self, system, config):
        IEEEG1Data.__init__(self)
        IEEEG1Model.__init__(self, system, config)
