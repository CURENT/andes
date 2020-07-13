from andes.core.model import ModelData, Model
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.common import dummify
from andes.core.var import Algeb, ExtState, ExtAlgeb, State
from andes.core.service import ConstService, ExtService, VarService, PostInitService, FlagValue
from andes.core.service import InitChecker, Replace  # NOQA
from andes.core.block import Block, LagAntiWindup, LeadLag, Washout, Lag, HVGate
from andes.core.block import Piecewise, GainLimiter, LessThan  # NOQA
from andes.core.block import Integrator
from andes.core.discrete import HardLimiter
from _collections import OrderedDict  # NOQA
import numpy as np  # NOQA


class ExcBaseData(ModelData):
    """
    Common parameters for exciters.
    """

    def __init__(self):
        super().__init__()
        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            )


class ExcBase(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'Exciter'
        self.flags.tds = True

        # from synchronous generators, get Sn, Vn, bus; tm0; omega
        self.Sn = ExtParam(src='Sn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='S_m',
                           info='Rated power from generator',
                           unit='MVA',
                           export=False,
                           )
        self.Vn = ExtParam(src='Vn',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name='V_m',
                           info='Rated voltage from generator',
                           unit='kV',
                           export=False,
                           )
        self.vf0 = ExtService(src='vf',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name='v_{f0}',
                              info='Steady state excitation voltage')
        self.bus = ExtParam(src='bus',
                            model='SynGen',
                            indexer=self.syn,
                            tex_name='bus',
                            info='Bus idx of the generators',
                            export=False,
                            dtype=str,
                            )
        self.omega = ExtState(src='omega',
                              model='SynGen',
                              indexer=self.syn,
                              tex_name=r'\omega',
                              info='Generator speed',
                              )
        self.vf = ExtAlgeb(src='vf',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'v_f',
                           e_str='u * (vout - vf0)',
                           info='Excitation field voltage to generator',
                           )
        self.XadIfd = ExtAlgeb(src='XadIfd',
                               model='SynGen',
                               indexer=self.syn,
                               tex_name=r'X_{ad}I_{fd}',
                               info='Armature excitation current',
                               )
        # from bus, get a and v
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          )

        # output excitation voltage
        self.vout = Algeb(info='Exciter final output voltage',
                          tex_name='v_{out}',
                          v_str='vf0',
                          )

        # Note:
        # Subclasses need to define `self.vref0` in the appropriate place.
        # Subclasses also need to define `self.vref`.


class EXDC2Data(ExcBaseData):
    def __init__(self):
        super().__init__()
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u.',
                           )
        self.TA = NumParam(info='Lag time constant in anti-windup lag',
                           tex_name='T_A',
                           default=0.04,
                           unit='p.u.',
                           )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=1,
                           unit='p.u.',
                           )
        self.TB = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_B',
                           default=1,
                           unit='p.u.',
                           )
        self.TE = NumParam(info='Exciter integrator time constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u.',
                           )
        self.TF1 = NumParam(info='Feedback washout time constant',
                            tex_name='T_{F1}',
                            default=1,
                            unit='p.u.',
                            non_zero=True
                            )
        self.KF1 = NumParam(info='Feedback washout gain',
                            tex_name='K_{F1}',
                            default=0.03,
                            unit='p.u.',
                            )
        self.KA = NumParam(info='Gain in anti-windup lag TF',
                           tex_name='K_A',
                           default=40,
                           unit='p.u.',
                           )
        self.KE = NumParam(info='Gain added to saturation',
                           tex_name='K_E',
                           default=1,
                           unit='p.u.',
                           )
        self.VRMAX = NumParam(info='Maximum excitation limit',
                              tex_name='V_{RMAX}',
                              default=7.3,
                              unit='p.u.')
        self.VRMIN = NumParam(info='Minimum excitation limit',
                              tex_name='V_{RMIN}',
                              default=-7.3,
                              unit='p.u.')
        self.E1 = NumParam(info='First saturation point',
                           tex_name='E_1',
                           default=0.0,
                           unit='p.u.',
                           )
        self.SE1 = NumParam(info='Value at first saturation point',
                            tex_name='S_{E1}',
                            default=0.0,
                            unit='p.u.',
                            )
        # the defaults for `E2` and `SE2` has been changed to 1
        # so that when E1=SE1=0, E2=SE2=1, saturation is disabled.
        # This will be patched later to allow all to be 0

        self.E2 = NumParam(info='Second saturation point',
                           tex_name='E_2',
                           default=1.0,
                           unit='p.u.',
                           )
        self.SE2 = NumParam(info='Value at second saturation point',
                            tex_name='S_{E2}',
                            default=1.0,
                            unit='p.u.',
                            )


class ExcExpSat(Block):
    r"""
    Exponential exciter saturation block to calculate
    A and B from E1, SE1, E2 and SE2.
    Input parameters will be corrected and the user will be warned.
    To disable saturation, set either E1 or E2 to 0.

    Parameters
    ----------
    E1 : BaseParam
        First point of excitation field voltage
    SE1: BaseParam
        Coefficient corresponding to E1
    E2 : BaseParam
        Second point of excitation field voltage
    SE2: BaseParam
        Coefficient corresponding to E2
    """

    def __init__(self, E1, SE1, E2, SE2, name=None, tex_name=None, info=None):
        Block.__init__(self, name=name, tex_name=tex_name, info=info)

        self._E1 = E1
        self._E2 = E2
        self._SE1 = SE1
        self._SE2 = SE2

        self.zE1 = FlagValue(self._E1, value=0.,
                             info='Flag non-zeros in E1',
                             tex_name='z^{E1}',
                             )
        self.zE2 = FlagValue(self._E2, value=0.,
                             info='Flag non-zeros in E2',
                             tex_name='z^{E2}',
                             )
        self.zSE1 = FlagValue(self._SE1, value=0.,
                              info='Flag non-zeros in SE1',
                              tex_name='z^{SE1}',
                              )
        self.zSE2 = FlagValue(self._SE2, value=0.,
                              info='Flag non-zeros in SE2',
                              tex_name='z^{SE2}')

        # disallow E1 = E2 != 0 since the curve fitting will fail
        self.E12c = InitChecker(
            self._E1, not_equal=self._E2,
            info='E1 and E2 after correction',
            error_out=True,
        )

        # data correction for E1, E2, SE1
        self.E1 = ConstService(
            tex_name='E^{1c}',
            info='Corrected E1 data',
        )
        self.E2 = ConstService(
            tex_name='E^{2c}',
            info='Corrected E2 data',
        )
        self.SE1 = ConstService(
            tex_name='SE^{1c}',
            info='Corrected SE1 data',
        )
        self.SE2 = ConstService(
            tex_name='SE^{2c}',
            info='Corrected SE2 data',
        )
        self.A = ConstService(info='Saturation gain',
                              tex_name='A^e',
                              )
        self.B = ConstService(info='Exponential coef. in saturation',
                              tex_name='B^e',
                              )
        self.vars = {
            'E1': self.E1,
            'E2': self.E2,
            'SE1': self.SE1,
            'SE2': self.SE2,
            'zE1': self.zE1,
            'zE2': self.zE2,
            'zSE1': self.zSE1,
            'zSE2': self.zSE2,
            'A': self.A,
            'B': self.B,
        }

    def define(self):
        r"""
        Notes
        -----
        The implementation solves for coefficients `A` and `B`
        which satisfy

        .. math ::
            E_1  S_{E1} = A e^{E1\times B}
            E_2  S_{E2} = A e^{E2\times B}

        The solutions are given by

        .. math ::
            E_{1} S_{E1} e^{ \frac{E_1 \log{ \left( \frac{E_2 S_{E2}} {E_1 S_{E1}} \right)} } {E_1 - E_2}}
            - \frac{\log{\left(\frac{E_2 S_{E2}}{E_1 S_{E1}} \right)}}{E_1 - E_2}
        """
        self.E1.v_str = f'{self._E1.name} + (1 - {self.name}_zE1)'
        self.E2.v_str = f'{self._E2.name} + 2*(1 - {self.name}_zE2)'

        self.SE1.v_str = f'{self._SE1.name} + (1 - {self.name}_zSE1)'
        self.SE2.v_str = f'{self._SE2.name} + 2*(1 - {self.name}_zSE2)'

        self.A.v_str = f'{self.name}_zE1*{self.name}_zE2 * ' \
                       f'{self.name}_E1*{self.name}_SE1*' \
                       f'exp({self.name}_E1*log({self.name}_E2*{self.name}_SE2/' \
                       f'({self.name}_E1*{self.name}_SE1))/({self.name}_E1-{self.name}_E2))'

        self.B.v_str = f'-log({self.name}_E2*{self.name}_SE2/({self.name}_E1*{self.name}_SE1))/' \
                       f'({self.name}_E1 - {self.name}_E2)'


class ExcQuadSat(Block):
    r"""
    Exponential exciter saturation block to calculate
    A and B from E1, SE1, E2 and SE2.
    Input parameters will be corrected and the user will be warned.
    To disable saturation, set either E1 or E2 to 0.

    Parameters
    ----------
    E1 : BaseParam
        First point of excitation field voltage
    SE1: BaseParam
        Coefficient corresponding to E1
    E2 : BaseParam
        Second point of excitation field voltage
    SE2: BaseParam
        Coefficient corresponding to E2
    """

    def __init__(self, E1, SE1, E2, SE2, name=None, tex_name=None, info=None):
        Block.__init__(self, name=name, tex_name=tex_name, info=info)

        self._E1 = dummify(E1)
        self._E2 = dummify(E2)
        self._SE1 = SE1
        self._SE2 = SE2

        self.zSE2 = FlagValue(self._SE2, value=0.,
                              info='Flag non-zeros in SE2',
                              tex_name='z^{SE2}')

        # data correction for E1, E2, SE1 (TODO)
        self.E1 = ConstService(
            tex_name='E^{1c}',
            info='Corrected E1 data',
        )
        self.E2 = ConstService(
            tex_name='E^{2c}',
            info='Corrected E2 data',
        )
        self.SE1 = ConstService(
            tex_name='SE^{1c}',
            info='Corrected SE1 data',
        )
        self.SE2 = ConstService(
            tex_name='SE^{2c}',
            info='Corrected SE2 data',
        )
        self.a = ConstService(info='Intermediate Sat coeff',
                              tex_name='a',
                              )
        self.A = ConstService(info='Saturation start',
                              tex_name='A^q',
                              )
        self.B = ConstService(info='Saturation gain',
                              tex_name='B^q',
                              )
        self.vars = {
            'E1': self.E1,
            'E2': self.E2,
            'SE1': self.SE1,
            'SE2': self.SE2,
            'zSE2': self.zSE2,
            'a': self.a,
            'A': self.A,
            'B': self.B,
        }

    def define(self):
        r"""
        Notes
        -----
        TODO.
        """
        # self.E1.v_str = f'{self._E1.name} + (1 - {self.name}_zE1)'
        # self.E2.v_str = f'{self._E2.name} + 2*(1 - {self.name}_zE2)'
        #
        # self.SE1.v_str = f'{self._SE1.name} + (1 - {self.name}_zSE1)'
        # self.SE2.v_str = f'{self._SE2.name} + 2*(1 - {self.name}_zSE2)'

        self.E1.v_str = f'{self._E1.name}'
        self.E2.v_str = f'{self._E2.name}'
        self.SE1.v_str = f'{self._SE1.name}'
        self.SE2.v_str = f'{self._SE2.name} + 2 * (1 - {self.name}_zSE2)'

        self.a.v_str = f'(({self.name}_SE2>0)+({self.name}_SE2<0)) * ' \
                       f'sqrt({self.name}_SE1 * {self.name}_E1 /({self.name}_SE2 * {self.name}_E2))'

        self.A.v_str = f'{self.name}_E2 - ({self.name}_E1 - {self.name}_E2) / ({self.name}_a - 1)'

        self.B.v_str = f'(({self.name}_a>0)+({self.name}_a<0)) *' \
                       f'{self.name}_SE2 * {self.name}_E2 * ({self.name}_a - 1)**2 / ' \
                       f'({self.name}_E1 - {self.name}_E2)** 2'


class EXDC2Model(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        self.SAT = ExcQuadSat(self.E1, self.SE1, self.E2, self.SE2,
                              info='Field voltage saturation',
                              )

        # calculate `Se0` ahead of time in order to calculate `vr0`
        self.Se0 = ConstService(info='Initial saturation output',
                                tex_name='S_{e0}',
                                v_str='(vf0>SAT_A) * SAT_B * (SAT_A - vf0) ** 2 / vf0',
                                # v_str='vf0',
                                )
        self.vr0 = ConstService(info='Initial vr',
                                tex_name='V_{r0}',
                                v_str='(KE + Se0) * vf0')
        self.vb0 = ConstService(info='Initial vb',
                                tex_name='V_{b0}',
                                v_str='vr0 / KA')

        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name='V_{ref0}',
                                  v_str='vb0 + v',
                                  )  # derived classes to-do: provide `v_str`

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='vref0',
                          e_str='vref0 - vref'
                          )

        self.SL = LessThan(u=self.vout,
                           bound=self.SAT_A,
                           equal=False,
                           enable=True,
                           cache=False,
                           )

        self.Se = Algeb(tex_name=r"S_e(|V_{out}|)", info='saturation output',
                        v_str='Se0',
                        e_str='SL_z0 * (vp - SAT_A) ** 2 * SAT_B / vp - Se',
                        )

        self.vp = State(info='Voltage after saturation feedback, before speed term',
                        tex_name='V_p',
                        unit='p.u.',
                        v_str='vf0',
                        e_str='LA_y - KE*vp - Se*vp',
                        t_const=self.TE,
                        )

        self.LS = Lag(u=self.v, T=self.TR, K=1.0, info='Sensing lag TF')

        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        )
        self.vi.v_str = 'vb0'
        self.vi.e_str = '(vref - LS_y - W_y) - vi'

        self.LL = LeadLag(u=self.vi,
                          T1=self.TC,
                          T2=self.TB,
                          info='Lead-lag for internal delays',
                          zero_out=True,
                          )
        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VRMAX,
                                lower=self.VRMIN,
                                info='Anti-windup lag',
                                )
        self.W = Washout(u=self.vp,
                         T=self.TF1,
                         K=self.KF1,
                         info='Signal conditioner'
                         )
        self.vout.e_str = 'omega * vp - vout'


class EXDC2(EXDC2Data, EXDC2Model):
    """
    EXDC2 model.
    """

    def __init__(self, system, config):
        EXDC2Data.__init__(self)
        EXDC2Model.__init__(self, system, config)


class SEXSData(ExcBaseData):
    """Data class for Simplified Excitation System model (SEXS)"""

    def __init__(self):
        ExcBaseData.__init__(self)
        self.TATB = NumParam(default=0.4,
                             tex_name='T_A/T_B',
                             info='Time constant TA/TB',
                             vrange=(0.05, 1),
                             )
        self.TB = NumParam(default=5,
                           tex_name='T_B',
                           info='Time constant TB in LL',
                           vrange=(5, 20),
                           )
        self.K = NumParam(default=20,
                          tex_name='K',
                          info='Gain',
                          non_zero=True,
                          vrange=(20, 100),
                          )
        # 5 <= K * TA / TB <= 15
        self.TE = NumParam(default='1',
                           tex_name='T_E',
                           info='AW Lag time constant',
                           vrange=(0, 0.5),
                           )
        self.EMIN = NumParam(default=-99,
                             tex_name='E_{MIN}',
                             info='lower limit',
                             )
        self.EMAX = NumParam(default=99,
                             tex_name='E_{MAX}',
                             info='upper limit',
                             vrange=(3, 6),
                             )


class SEXSModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        self.TA = ConstService(v_str='TATB * TB')

        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name='V_{ref0}',
                                  v_str='vf0/K + v',
                                  )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='vref0',
                          e_str='vref0 - vref'
                          )
        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        )
        self.vi.e_str = '(vref - v) - vi'
        self.vi.v_str = 'vref0 - v'

        self.LL = LeadLag(u=self.vi, T1=self.TA, T2=self.TB, zero_out=True)

        self.LAW = LagAntiWindup(u=self.LL_y,
                                 T=self.TE,
                                 K=self.K,
                                 lower=self.EMIN,
                                 upper=self.EMAX,
                                 )

        self.vout.e_str = 'LAW_y - vout'


class SEXS(SEXSData, SEXSModel):
    """Simplified Excitation System"""

    def __init__(self, system, config):
        SEXSData.__init__(self)
        SEXSModel.__init__(self, system, config)


class EXST1Data(ExcBaseData):
    """Parameters for EXST1."""

    def __init__(self):
        ExcBaseData.__init__(self)

        self.TR = NumParam(default=0.01,
                           info='Measurement delay',
                           tex_name='T_R',
                           )
        self.VIMAX = NumParam(default=0.2,
                              info='Max. input voltage',
                              tex_name='V_{IMAX}',
                              )

        self.VIMIN = NumParam(default=0,
                              info='Min. input voltage',
                              tex_name='V_{IMIN}',
                              )
        self.TC = NumParam(default=1,
                           info='LL numerator',
                           tex_name='T_C',
                           )
        self.TB = NumParam(default=1,
                           info='LL denominator',
                           tex_name='T_B',
                           )
        self.KA = NumParam(default=80,
                           info='Regulator gain',
                           tex_name='K_A',
                           )
        self.TA = NumParam(default=0.05,
                           info='Regulator delay',
                           tex_name='T_A',
                           )
        self.VRMAX = NumParam(default=8,
                              info='Max. regulator output',
                              tex_name='V_{RMAX}',
                              )

        self.VRMIN = NumParam(default=-3,
                              info='Min. regulator output',
                              tex_name='V_{RMIN}',
                              )
        self.KC = NumParam(default=0.2,
                           info='Coef. for Ifd',
                           tex_name='K_C',
                           )
        self.KF = NumParam(default=0.1,
                           info='Feedback gain',
                           tex_name='K_F',
                           )
        self.TF = NumParam(default=1.0,
                           info='Feedback delay',
                           tex_name='T_F',
                           positive=True,
                           )


class EXST1Model(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name='V_{ref0}',
                                  v_str='v + vf0 / KA',
                                  )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='vref0',
                          e_str='vref0 - vref'
                          )

        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        )
        self.vi.v_str = 'vf0 / KA'
        self.vi.e_str = '(vref - LG_y - WF_y) - vi'

        self.LG = Lag(u=self.v, T=self.TR, K=1,
                      info='Sensing delay',
                      )

        self.HLI = HardLimiter(u=self.vi, lower=self.VIMIN, upper=self.VIMAX,
                               info='Hard limiter on input',
                               )

        self.vl = Algeb(info='Input after limiter',
                        tex_name='V_l',
                        v_str='HLI_zi*vi + HLI_zu*VIMAX + HLI_zl*VIMIN',
                        e_str='HLI_zi*vi + HLI_zu*VIMAX + HLI_zl*VIMIN - vl',
                        )

        self.LL = LeadLag(u=self.vl, T1=self.TC, T2=self.TB, info='Lead-lag compensator', zero_out=True)

        self.LR = Lag(u=self.LL_y, T=self.TA, K=self.KA, info='Regulator')

        self.WF = Washout(u=self.LR_y, T=self.TF, K=self.KF, info='Stablizing circuit feedback')

        # the following uses `XadIfd` for `IIFD` in the PSS/E manual
        self.vfmax = Algeb(info='Upper bound of output limiter',
                           tex_name='V_{fmax}',
                           v_str='VRMAX - KC * XadIfd',
                           e_str='VRMAX - KC * XadIfd - vfmax',
                           )
        self.vfmin = Algeb(info='Lower bound of output limiter',
                           tex_name='V_{fmin}',
                           v_str='VRMIN - KC * XadIfd',
                           e_str='VRMIN - KC * XadIfd - vfmin',
                           )

        self.HLR = HardLimiter(u=self.WF_y, lower=self.vfmin, upper=self.vfmax,
                               info='Hard limiter on regulator output')

        self.vout.e_str = 'LR_y*HLR_zi + vfmin*HLR_zl + vfmax*HLR_zu - vout'


class EXST1(EXST1Data, EXST1Model):
    """
    EXST1-type static excitation system.
    """

    def __init__(self, system, config):
        EXST1Data.__init__(self)
        EXST1Model.__init__(self, system, config)


class ESST3AData(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u.',
                           )
        self.VIMAX = NumParam(default=0.8,
                              info='Max. input voltage',
                              tex_name='V_{IMAX}',
                              vrange=(0, 1),
                              )
        self.VIMIN = NumParam(default=-0.1,
                              info='Min. input voltage',
                              tex_name='V_{IMIN}',
                              vrange=(-1, 0),
                              )

        self.KM = NumParam(default=500,
                           tex_name='K_M',
                           info='Forward gain constant',
                           vrange=(0, 1000),
                           )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=3,
                           vrange=(0, 20),
                           )
        self.TB = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_B',
                           default=15,
                           vrange=(0, 20),
                           )

        self.KA = NumParam(info='Gain in anti-windup lag TF',
                           tex_name='K_A',
                           default=50,
                           vrange=(0, 200),
                           )
        self.TA = NumParam(info='Lag time constant in anti-windup lag',
                           tex_name='T_A',
                           default=0.1,
                           vrange=(0, 1),
                           )
        self.VRMAX = NumParam(info='Maximum excitation limit',
                              tex_name='V_{RMAX}',
                              default=8,
                              unit='p.u.',
                              vrange=(0.5, 10),
                              )
        self.VRMIN = NumParam(info='Minimum excitation limit',
                              tex_name='V_{RMIN}',
                              default=0,
                              unit='p.u.',
                              vrange=(-10, 0.5),
                              )
        self.KG = NumParam(info='Feedback gain of inner field regulator',
                           tex_name='K_G',
                           default=1,
                           vrange=(0, 1.1),
                           )
        self.KP = NumParam(info='Potential circuit gain coeff.',
                           tex_name='K_P',
                           default=4,
                           vrange=(1, 10),
                           )
        self.KI = NumParam(info='Potential circuit gain coeff.',
                           tex_name='K_I',
                           default=0.1,
                           vrange=(0, 1.1),
                           )
        self.VBMAX = NumParam(info='VB upper limit',
                              tex_name='V_{BMAX}',
                              default=18,
                              unit='p.u.',
                              vrange=(0, 20),
                              )
        self.KC = NumParam(default=0.1,
                           info='Rectifier loading factor proportional to commutating reactance',
                           tex_name='K_C',
                           vrange=(0, 1),
                           )
        self.XL = NumParam(default=0.01,
                           info='Potential source reactance',
                           tex_name='X_L',
                           vrange=(0, 0.5),
                           )
        self.VGMAX = NumParam(info='VG upper limit',
                              tex_name='V_{GMAX}',
                              default=4,
                              unit='p.u.',
                              vrange=(0, 20),
                              )
        self.THETAP = NumParam(info='Rectifier firing angle',
                               tex_name=r'\theta_P',
                               default=0,
                               unit='degree',
                               vrange=(0, 90),
                               )
        self.TM = NumParam(default=0.1,
                           info='Inner field regulator forward time constant',
                           tex_name='K_C',
                           )

        self.VMMAX = NumParam(info='Maximum VM limit',
                              tex_name='V_{MMAX}',
                              default=1,
                              unit='p.u.',
                              vrange=(0.5, 1.5),
                              )
        self.VMMIN = NumParam(info='Minimum VM limit',
                              tex_name='V_{RMIN}',
                              default=0.1,
                              unit='p.u.',
                              vrange=(-1.5, 0.5),
                              )


class ESST3AModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        self.KPC = ConstService(v_str='KP * exp(1j * radians(THETAP))',
                                tex_name='K_{PC}',
                                info='KP polar THETAP',
                                vtype=np.complex
                                )

        # vd, vq, Id, Iq from SynGen
        self.vd = ExtAlgeb(src='vd',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'V_d',
                           info='d-axis machine voltage',
                           )
        self.vq = ExtAlgeb(src='vq',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'V_q',
                           info='q-axis machine voltage',
                           )
        self.Id = ExtAlgeb(src='Id',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'I_d',
                           info='d-axis machine current',
                           )

        self.Iq = ExtAlgeb(src='Iq',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name=r'I_q',
                           info='q-axis machine current',
                           )

        # control block begin
        self.LG = Lag(self.v, T=self.TR, K=1,
                      info='Voltage transducer',
                      )

        self.UEL = Algeb(info='Interface var for under exc. limiter',
                         tex_name='U_{EL}',
                         v_str='0',
                         e_str='0 - UEL'
                         )

        self.VE = VarService(tex_name='V_E',
                             info='VE',
                             v_str='Abs(KPC*(vd + 1j*vq) + 1j*(KI + KPC*XL)*(Id + 1j*Iq))',
                             )

        self.IN = Algeb(tex_name='I_N',
                        info='Input to FEX',
                        v_str='KC * XadIfd / VE',
                        e_str='KC * XadIfd / VE - IN',
                        )

        self.FEX = Piecewise(u=self.IN,
                             points=(0, 0.433, 0.75, 1),
                             funs=('1', '1 - 0.577*IN', 'sqrt(0.75 - IN ** 2)', '1.732*(1 - IN)', 0),
                             info='Piecewise function FEX',
                             )

        self.VBMIN = dummify(-9999)
        self.VGMIN = dummify(-9999)

        self.VB = GainLimiter(u='VE*FEX_y',
                              K=1,
                              upper=self.VBMAX,
                              lower=self.VBMIN,
                              no_lower=True,
                              info='VB with limiter',
                              )

        self.VG = GainLimiter(u=self.vout,
                              K=self.KG,
                              upper=self.VGMAX,
                              lower=self.VGMIN,
                              no_lower=True,
                              info='Feedback gain with HL',
                              )

        self.vrs = Algeb(tex_name='V_{RS}',
                         info='VR subtract feedback VG',
                         v_str='vf0 / VB_y / KM',
                         e_str='LAW1_y - VG_y - vrs',
                         )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='(vrs + VG_y) / KA + v',
                          e_str='vref0 - vref',
                          )

        self.vref0 = PostInitService(info='Initial reference voltage input',
                                     tex_name='V_{ref0}',
                                     v_str='vref',
                                     )

        # input excitation voltages; PSS outputs summed at vi
        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        e_str='-LG_y + vref - vi',
                        v_str='-v + vref',
                        )

        self.vil = Algeb(info='Input voltage after limit',
                         tex_name='V_{il}',
                         v_str='HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX',
                         e_str='HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX - vil'
                         )

        self.HG = HVGate(u1=self.UEL,
                         u2=self.vil,
                         info='HVGate for under excitation',
                         )

        self.LL = LeadLag(u=self.HG_y, T1=self.TC, T2=self.TB,
                          info='Regulator',
                          zero_out=True,
                          )  # LL_y == VA

        self.LAW1 = LagAntiWindup(u=self.LL_y,
                                  T=self.TA,
                                  K=self.KA,
                                  lower=self.VRMIN,
                                  upper=self.VRMAX,
                                  info='Lag AW on VR',
                                  )  # LAW1_y == VR

        self.HLI = HardLimiter(u=self.vi,
                               lower=self.VIMIN,
                               upper=self.VIMAX,
                               info='Input limiter',
                               )

        self.LAW2 = LagAntiWindup(u=self.vrs,
                                  T=self.TM,
                                  K=self.KM,
                                  lower=self.VMMIN,
                                  upper=self.VMMAX,
                                  info='Lag AW on VM',
                                  )  # LAW2_y == VM

        self.vout.e_str = 'VB_y * LAW2_y - vout'


class ESST3A(ESST3AData, ESST3AModel):
    """
    Static exciter type 3A model
    """

    def __init__(self, system, config):
        ESST3AData.__init__(self)
        ESST3AModel.__init__(self, system, config)


class ESDC2AData(ExcBaseData):
    def __init__(self):
        ExcBaseData.__init__(self)
        self.TR = NumParam(info='Sensing time constant',
                           tex_name='T_R',
                           default=0.01,
                           unit='p.u.',
                           )
        self.KA = NumParam(default=80,
                           info='Regulator gain',
                           tex_name='K_A',
                           )
        self.TA = NumParam(info='Lag time constant in regulator',
                           tex_name='T_A',
                           default=0.04,
                           unit='p.u.',
                           )
        self.TB = NumParam(info='Lag time constant in lead-lag',
                           tex_name='T_B',
                           default=1,
                           unit='p.u.',
                           )
        self.TC = NumParam(info='Lead time constant in lead-lag',
                           tex_name='T_C',
                           default=1,
                           unit='p.u.',
                           )
        self.VRMAX = NumParam(info='Max. exc. limit (0-unlimited)',
                              tex_name='V_{RMAX}',
                              default=7.3,
                              unit='p.u.')
        self.VRMIN = NumParam(info='Min. excitation limit',
                              tex_name='V_{RMIN}',
                              default=-7.3,
                              unit='p.u.')
        self.KE = NumParam(info='Saturation feedback gain',
                           tex_name='K_E',
                           default=1,
                           unit='p.u.',
                           )
        self.TE = NumParam(info='Integrator time constant',
                           tex_name='T_E',
                           default=0.8,
                           unit='p.u.',
                           )
        self.KF = NumParam(default=0.1,
                           info='Feedback gain',
                           tex_name='K_F',
                           )
        self.TF1 = NumParam(info='Feedback washout time constant',
                            tex_name='T_{F1}',
                            default=1,
                            unit='p.u.',
                            positive=True
                            )
        self.Switch = NumParam(info='Switch that PSS/E did not implement',
                               tex_name='S_w',
                               default=0,
                               unit='bool',
                               )

        self.E1 = NumParam(info='First saturation point',
                           tex_name='E_1',
                           default=0.,
                           unit='p.u.',
                           )
        self.SE1 = NumParam(info='Value at first saturation point',
                            tex_name='S_{E1}',
                            default=0.,
                            unit='p.u.',
                            )
        self.E2 = NumParam(info='Second saturation point',
                           tex_name='E_2',
                           default=0.,
                           unit='p.u.',
                           )
        self.SE2 = NumParam(info='Value at second saturation point',
                            tex_name='S_{E2}',
                            default=0.,
                            unit='p.u.',
                            )


class ESDC2AModel(ExcBase):
    def __init__(self, system, config):
        ExcBase.__init__(self, system, config)

        # Set VRMAX to 999 when VRMAX = 0
        self._zVRM = FlagValue(self.VRMAX, value=0,
                               tex_name='z_{VRMAX}',
                               )
        self.VRMAXc = ConstService(v_str='VRMAX + 999*(1-_zVRM)',
                                   info='Set VRMAX=999 when zero',
                                   )
        self.LG = Lag(u=self.v, T=self.TR, K=1,
                      info='Transducer delay',
                      )

        self.SAT = ExcQuadSat(self.E1, self.SE1, self.E2, self.SE2,
                              info='Field voltage saturation',
                              )

        self.Se0 = ConstService(
            tex_name='S_{e0}',
            v_str='(vf0>SAT_A) * SAT_B*(SAT_A-vf0) ** 2 / vf0',
        )

        self.vfe0 = ConstService(v_str='vf0 * (KE + Se0)',
                                 tex_name='V_{FE0}',
                                 )
        self.vref0 = ConstService(info='Initial reference voltage input',
                                  tex_name='V_{ref0}',
                                  v_str='v + vfe0 / KA',
                                  )

        self.vref = Algeb(info='Reference voltage input',
                          tex_name='V_{ref}',
                          unit='p.u.',
                          v_str='vref0',
                          e_str='vref0 - vref'
                          )

        self.vi = Algeb(info='Total input voltages',
                        tex_name='V_i',
                        unit='p.u.',
                        v_str='vref0 - v',
                        e_str='(vref - v - WF_y) - vi',
                        )

        self.LL = LeadLag(u=self.vi,
                          T1=self.TC,
                          T2=self.TB,
                          info='Lead-lag compensator',
                          zero_out=True,
                          )

        self.UEL = Algeb(info='Interface var for under exc. limiter',
                         tex_name='U_{EL}',
                         v_str='0',
                         e_str='0 - UEL'
                         )

        self.HG = HVGate(u1=self.UEL,
                         u2=self.LL_y,
                         info='HVGate for under excitation',
                         )

        self.VRU = VarService(v_str='VRMAXc * v',
                              tex_name='V_T V_{RMAX}',
                              )
        self.VRL = VarService(v_str='VRMIN * v',
                              tex_name='V_T V_{RMIN}',
                              )

        # TODO: WARNING: HVGate is temporarily skipped
        self.LA = LagAntiWindup(u=self.LL_y,
                                T=self.TA,
                                K=self.KA,
                                upper=self.VRU,
                                lower=self.VRL,
                                info='Anti-windup lag',
                                )  # LA_y == VR

        # `LessThan` may be causing memory issue in (SL_z0 * vout) - uncertain yet
        self.SL = LessThan(u=self.vout, bound=self.SAT_A, equal=False, enable=True, cache=False)

        self.Se = Algeb(tex_name=r"S_e(|V_{out}|)", info='saturation output',
                        v_str='Se0',
                        e_str='SL_z0 * (INT_y - SAT_A) ** 2 * SAT_B / INT_y - Se',
                        )

        self.VFE = Algeb(info='Combined saturation feedback',
                         tex_name='V_{FE}',
                         unit='p.u.',
                         v_str='vfe0',
                         e_str='INT_y * (KE + Se) - VFE'
                         )

        self.INT = Integrator(u='LA_y - VFE',
                              T=self.TE,
                              K=1,
                              y0=self.vf0,
                              info='Integrator',
                              )

        self.WF = Washout(u=self.INT_y,
                          T=self.TF1,
                          K=self.KF,
                          info='Feedback to input'
                          )

        self.vout.e_str = 'INT_y - vout'


class ESDC2A(ESDC2AData, ESDC2AModel):
    """
    ESDC2A model.

    This model is implemented as described in the PSS/E manual,
    except that the HVGate is not in use.
    Due to the HVGate and saturation function, the results
    are close to but different from TSAT.
    """

    def __init__(self, system, config):
        ESDC2AData.__init__(self)
        ESDC2AModel.__init__(self, system, config)


class IEEEX1Model(EXDC2Model):

    def __init__(self, system, config):
        EXDC2Model.__init__(self, system, config)
        self.VRTMAX = VarService('VRMAX * v',
                                 tex_name='V_{RMAX}V_T')
        self.VRTMIN = VarService('VRMIN * v',
                                 tex_name='V_{RMIN}V_T')

        self.LA.upper = self.VRTMAX
        self.LA.lower = self.VRTMIN


class IEEEX1(EXDC2Data, IEEEX1Model):
    """
    IEEEX1 Type 1 exciter (DC)

    Derived from EXDC2 by varying the limiter bounds.
    """
    def __init__(self, system, config):
        EXDC2Data.__init__(self)
        IEEEX1Model.__init__(self, system, config)

        self.vout.e_str = 'vp - vout'
