from collections import OrderedDict

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        IdxParam, Lag, LeadLag, LessThan, Limiter, Model,
                        ModelData, NumParam, Switcher,)
from andes.core.block import DeadBand1, PITrackAW
from andes.core.service import (CurrentSign, DataSelect, DeviceFinder,
                                NumSelect, VarHold, VarService,)


class REPCA1Data(ModelData):
    """
    Parameters for the Renewable Energy Plant Control model.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.ree = IdxParam(info='RenExciter idx',
                            model='RenExciter',
                            mandatory=True,
                            )

        self.line = IdxParam(info='Idx of line that connect to measured bus',
                             model='ACLine',
                             mandatory=True,
                             )

        self.busr = IdxParam(info='Optional remote bus for voltage and freq. measurement',
                             model='Bus',
                             default=None,
                             )

        self.busf = IdxParam(info='BusFreq idx for mode 2',
                             model='BusFreq',
                             default=None,
                             )

        # --- flags ---
        self.VCFlag = NumParam(info='Droop flag; 0-with droop if power factor ctrl, 1-line drop comp.',
                               mandatory=True,
                               unit='bool',
                               )

        self.RefFlag = NumParam(info='Q/V select; 0-Q control, 1-V control',
                                mandatory=True,
                                unit='bool',
                                )

        self.Fflag = NumParam(info='Frequency control flag; 0-disable, 1-enable',
                              mandatory=True,
                              unit='bool',
                              )

        self.PLflag = NumParam(info='Pline ctrl. flag; 0-disable, 1-enable',
                               default=True,
                               unit='bool',
                               )

        self.Tfltr = NumParam(default=0.02,
                              tex_name='T_{fltr}',
                              info='V or Q filter time const.',
                              )

        self.Kp = NumParam(default=1.0,
                           tex_name='K_p',
                           info='Q proportional gain',
                           )

        self.Ki = NumParam(default=0.1,
                           tex_name='K_i',
                           info='Q integral gain',
                           )

        self.Tft = NumParam(default=1.0,
                            tex_name='T_{ft}',
                            info='Lead time constant',
                            )

        self.Tfv = NumParam(default=1.0,
                            tex_name='T_{fv}',
                            info='Lag time constant',
                            )

        self.Vfrz = NumParam(default=0.8,
                             tex_name='V_{frz}',
                             info='Voltage below which s2 is frozen',
                             )

        self.Rc = NumParam(default=None,
                           tex_name='R_c',
                           info='Line drop compensation R',
                           )

        self.Xc = NumParam(default=None,
                           tex_name='X_c',
                           info='Line drop compensation R',
                           )

        self.Kc = NumParam(default=0.0,
                           tex_name='K_c',
                           info='Reactive power compensation gain',
                           )

        self.emax = NumParam(default=999,
                             tex_name='e_{max}',
                             info='Upper limit on deadband output',
                             )

        self.emin = NumParam(default=-999,
                             tex_name='e_{min}',
                             info='Lower limit on deadband output',
                             )

        self.dbd1 = NumParam(default=-0.1,
                             tex_name='d_{bd1}',
                             info='Lower threshold for reactive power control deadband (<=0)',
                             )

        self.dbd2 = NumParam(default=0.1,
                             tex_name='d_{bd2}',
                             info='Upper threshold for reactive power control deadband (>=0)',
                             )

        self.Qmax = NumParam(default=999.0,
                             tex_name='Q_{max}',
                             info='Upper limit on output of V-Q control',
                             )

        self.Qmin = NumParam(default=-999.0,
                             tex_name='Q_{min}',
                             info='Lower limit on output of V-Q control',
                             )

        self.Kpg = NumParam(default=1.0,
                            tex_name='K_{pg}',
                            info='Proportional gain for power control',
                            )

        self.Kig = NumParam(default=0.1,
                            tex_name='K_{ig}',
                            info='Integral gain for power control',
                            )

        self.Tp = NumParam(default=0.02,
                           tex_name='T_p',
                           info='Time constant for P measurement',
                           )

        self.fdbd1 = NumParam(default=-0.0002833,
                              tex_name='f_{dbd1}',
                              info='Lower threshold for freq. error deadband',
                              unit='p.u. (Hz)',
                              )

        self.fdbd2 = NumParam(default=0.0002833,
                              tex_name='f_{dbd2}',
                              info='Upper threshold for freq. error deadband',
                              unit='p.u. (Hz)',
                              )

        self.femax = NumParam(default=0.05,
                              tex_name='f_{emax}',
                              info='Upper limit for freq. error',
                              )

        self.femin = NumParam(default=-0.05,
                              tex_name='f_{emin}',
                              info='Lower limit for freq. error',
                              )

        self.Pmax = NumParam(default=999,
                             tex_name='P_{max}',
                             info='Upper limit on power error (used by PI ctrl.)',
                             unit='p.u. (MW)',
                             power=True,
                             )

        self.Pmin = NumParam(default=-999,
                             tex_name='P_{min}',
                             info='Lower limit on power error (used by PI ctrl.)',
                             unit='p.u. (MW)',
                             power=True,
                             )

        self.Tg = NumParam(default=0.02,
                           tex_name='T_g',
                           info='Power controller lag time constant',
                           )

        self.Ddn = NumParam(default=10,
                            tex_name='D_{dn}',
                            info='Reciprocal of droop for over-freq. conditions',
                            )

        self.Dup = NumParam(default=10,
                            tex_name='D_{up}',
                            info='Reciprocal of droop for under-freq. conditions',
                            )


class REPCA1Model(Model):
    """
    REPCA1 model implementation
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.group = 'RenPlant'
        self.flags.tds = True

        self.config.add(OrderedDict((('kqs', 2),
                                     ('ksg', 2),
                                     ('freeze', 1),
                                     )))

        self.config.add_extra('_help',
                              kqs='Tracking gain for reactive power PI controller',
                              ksg='Tracking gain for active power PI controller',
                              freeze='Voltage dip freeze flag; 1-enable, 0-disable',
                              )
        self.config.add_extra('_tex',
                              kqs='K_{qs}',
                              ksg='K_{sg}',
                              freeze='f_{rz}')

        # --- from RenExciter ---
        self.reg = ExtParam(model='RenExciter', src='reg', indexer=self.ree, export=False,
                            info='Retrieved RenGen idx', vtype=str, default=None,
                            )
        self.Pext = ExtAlgeb(model='RenExciter', src='Pref', indexer=self.ree,
                             info='Pref from RenExciter renamed as Pext',
                             tex_name='P_{ext}',
                             )

        self.Qext = ExtAlgeb(model='RenExciter', src='Qref', indexer=self.ree,
                             info='Qref from RenExciter renamed as Qext',
                             tex_name='Q_{ext}',
                             )

        # --- from RenGen ---
        self.bus = ExtParam(model='RenGen', src='bus', indexer=self.reg, export=False,
                            info='Retrieved bus idx', vtype=str, default=None,
                            )

        self.buss = DataSelect(self.busr, self.bus, info='selected bus (bus or busr)')

        self.busfreq = DeviceFinder(self.busf, link=self.buss, idx_name='bus',
                                    default_model='BusFreq')

        # from Bus
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.buss, tex_name='V',
                          info='Bus (or busr, if given) terminal voltage',
                          )

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.buss, tex_name=r'\theta',
                          info='Bus (or busr, if given) phase angle',
                          )

        self.v0 = ExtService(model='Bus', src='v', indexer=self.buss, tex_name="V_0",
                             info='Initial bus voltage',
                             )

        # from BusFreq
        self.f = ExtAlgeb(model='FreqMeasurement', src='f', indexer=self.busfreq, export=False,
                          info='Bus frequency', unit='p.u.')

        # from Line
        self.bus1 = ExtParam(model='ACLine', src='bus1', indexer=self.line, export=False,
                             info='Retrieved Line.bus1 idx', vtype=str, default=None,
                             )

        self.bus2 = ExtParam(model='ACLine', src='bus2', indexer=self.line, export=False,
                             info='Retrieved Line.bus2 idx', vtype=str, default=None,
                             )
        self.r = ExtParam(model='ACLine', src='r', indexer=self.line, export=False,
                          info='Retrieved Line.r', vtype=str, default=None,
                          )

        self.x = ExtParam(model='ACLine', src='x', indexer=self.line, export=False,
                          info='Retrieved Line.x', vtype=str, default=None,
                          )

        self.v1 = ExtAlgeb(model='ACLine', src='v1', indexer=self.line, tex_name='V_1',
                           info='Voltage at Line.bus1',
                           )

        self.v2 = ExtAlgeb(model='ACLine', src='v2', indexer=self.line, tex_name='V_2',
                           info='Voltage at Line.bus2',
                           )

        self.a1 = ExtAlgeb(model='ACLine', src='a1', indexer=self.line, tex_name=r'\theta_1',
                           info='Angle at Line.bus1',
                           )

        self.a2 = ExtAlgeb(model='ACLine', src='a2', indexer=self.line, tex_name=r'\theta_2',
                           info='Angle at Line.bus2',
                           )

        # -- begin services ---

        self.Isign = CurrentSign(self.bus, self.bus1, self.bus2, tex_name='I_{sign}')

        Iline = '(Isign * (v1*exp(1j*a1) - v2*exp(1j*a2)) / (r + 1j*x))'

        self.Iline = VarService(v_str=Iline, vtype=complex,
                                info='Complex current from bus1 to bus2',
                                tex_name='I_{line}',
                                )

        self.Iline0 = ConstService(v_str='Iline', vtype=complex,
                                   info='Initial complex current from bus1 to bus2',
                                   tex_name='I_{line0}',
                                   )

        Pline = 're(Isign * v1*exp(1j*a1) * conj((v1*exp(1j*a1) - v2*exp(1j*a2)) / (r + 1j*x)))'

        self.Pline = VarService(v_str=Pline, vtype=float,
                                info='Complex power from bus1 to bus2',
                                tex_name='P_{line}',
                                )

        self.Pline0 = ConstService(v_str='Pline', vtype=float,
                                   info='Initial vomplex power from bus1 to bus2',
                                   tex_name='P_{line0}',
                                   )

        Qline = 'im(Isign * v1*exp(1j*a1) * conj((v1*exp(1j*a1) - v2*exp(1j*a2)) / (r + 1j*x)))'

        self.Qline = VarService(v_str=Qline, vtype=float,
                                info='Complex power from bus1 to bus2',
                                tex_name='Q_{line}',
                                )

        self.Qline0 = ConstService(v_str='Qline', vtype=float,
                                   info='Initial complex power from bus1 to bus2',
                                   tex_name='Q_{line0}',
                                   )

        self.Rcs = NumSelect(self.Rc, self.r, info='Line R (Rc if provided, otherwise line.r)',
                             tex_name='R_{cs}',
                             )

        self.Xcs = NumSelect(self.Xc, self.x, info='Line X (Xc if provided, otherwise line.x)',
                             tex_name='X_{cs}',
                             )

        self.Vcomp = VarService(v_str='abs(v*exp(1j*a) - (Rcs + 1j * Xcs) * Iline)',
                                info='Voltage after Rc/Xc compensation',
                                tex_name='V_{comp}'
                                )

        self.SWVC = Switcher(u=self.VCFlag, options=(0, 1), tex_name='SW_{VC}', cache=True)

        self.SWRef = Switcher(u=self.RefFlag, options=(0, 1), tex_name='SW_{Ref}', cache=True)

        self.SWF = Switcher(u=self.Fflag, options=(0, 1), tex_name='SW_{F}', cache=True)

        self.SWPL = Switcher(u=self.PLflag, options=(0, 1), tex_name='SW_{PL}', cache=True)

        VCsel = '(SWVC_s1 * Vcomp + SWVC_s0 * (Qline * Kc + v))'

        self.Vref0 = ConstService(v_str='(SWVC_s1 * Vcomp + SWVC_s0 * (Qline0 * Kc + v))',
                                  tex_name='V_{ref0}',
                                  )

        self.s0 = Lag(VCsel, T=self.Tfltr, K=1, tex_name='s_0',
                      info='V filter',
                      )  # s0_y is the filter output of voltage deviation

        self.s1 = Lag(self.Qline, T=self.Tfltr, K=1, tex_name='s_1')

        self.Vref = Algeb(v_str='Vref0', e_str='Vref0 - Vref', tex_name='Q_{ref}')

        self.Qlinef = Algeb(v_str='Qline0', e_str='Qline0 - Qlinef', tex_name='Q_{linef}')

        Refsel = '(SWRef_s0 * (Qlinef - s1_y) + SWRef_s1 * (Vref - s0_y))'

        self.Refsel = Algeb(v_str=Refsel, e_str=f'{Refsel} - Refsel', tex_name='R_{efsel}')

        self.dbd = DeadBand1(u=self.Refsel, lower=self.dbd1, upper=self.dbd2, center=0.0,
                             tex_name='d^{bd}',
                             )

        # --- e Hardlimit and hold logic ---
        self.eHL = Limiter(u=self.dbd_y, lower=self.emin, upper=self.emax,
                           tex_name='e_{HL}',
                           info='Hardlimit on deadband output',
                           )

        self.zf = VarService(v_str='Indicator(v < Vfrz) * freeze',
                             tex_name='z_f',
                             info='PI Q input freeze signal',
                             )

        self.enf = Algeb(tex_name='e_{nf}',
                         info='e Hardlimit output before freeze',
                         v_str='dbd_y*eHL_zi + emax*eHL_zu + emin*eHL_zl',
                         e_str='dbd_y*eHL_zi + emax*eHL_zu + emin*eHL_zl - enf',
                         )

        # --- hold of `enf` when v < vfrz

        self.eHld = VarHold(u=self.enf, hold=self.zf, tex_name='e_{hld}',
                            info='e Hardlimit output after conditional hold',
                            )

        self.s2 = PITrackAW(u='eHld',
                            kp=self.Kp, ki=self.Ki, ks=self.config.kqs,
                            lower=self.Qmin, upper=self.Qmax,
                            info='PI controller for eHL output',
                            tex_name='s_2',
                            )

        self.s3 = LeadLag(u=self.s2_y, T1=self.Tft, T2=self.Tfv, K=1,
                          tex_name='s_3',
                          )  # s3_y == Qext

        # Active power part

        self.s4 = Lag(self.Pline, T=self.Tp, K=1,
                      tex_name='s_4',
                      info='Pline filter',
                      )

        self.Freq_ref = ConstService(v_str='1.0',
                                     tex_name='f_{ref}',
                                     info='Initial Freq_ref')
        self.ferr = Algeb(tex_name='f_{err}',
                          info='Frequency deviation',
                          unit='p.u. (Hz)',
                          v_str='(Freq_ref - f)',
                          e_str='(Freq_ref - f) - ferr',
                          )

        self.fdbd = DeadBand1(u=self.ferr, center=0.0, lower=self.fdbd1,
                              upper=self.fdbd2,
                              tex_name='f^{dbd}',
                              info='frequency error deadband',
                              )

        self.fdlt0 = LessThan(self.fdbd_y, 0.0,
                              tex_name='f_{dlt0}',
                              info='frequency deadband output less than zero',
                              )

        fdroop = '(fdbd_y * Ddn * fdlt0_z1 + fdbd_y * Dup * fdlt0_z0)'

        self.Plant_pref = Algeb(tex_name='P_{ref}',
                                info='Plant P ref',
                                v_str='Pline0',
                                e_str='Pline0 - Plant_pref',
                                )

        self.Plerr = Algeb(tex_name='P_{lerr}',
                           info='Pline error',
                           v_str='- s4_y + Plant_pref',
                           e_str='- s4_y + Plant_pref - Plerr',
                           )

        self.Perr = Algeb(tex_name='P_{err}',
                          info='Power error before fe limits',
                          v_str=f'{fdroop} + Plerr * SWPL_s1',
                          e_str=f'{fdroop} + Plerr * SWPL_s1 - Perr',
                          )

        self.feHL = Limiter(self.Perr, lower=self.femin, upper=self.femax,
                            tex_name='f_{eHL}',
                            info='Limiter for power (frequency) error',
                            )

        feout = '(Perr * feHL_zi + femin * feHL_zl + femax * feHL_zu)'
        self.s5 = PITrackAW(u=feout, kp=self.Kpg, ki=self.Kig, ks=self.config.ksg,
                            lower=self.Pmin, upper=self.Pmax,
                            tex_name='s_5',
                            info='PI for fe limiter output',
                            )

        self.s6 = Lag(u=self.s5_y, T=self.Tg, K=1,
                      tex_name='s_6',
                      info='Output filter for Pext',
                      )

        Qext = '(s3_y)'

        Pext = '(SWF_s1 * s6_y)'

        self.Pext.e_str = Pext

        self.Qext.e_str = Qext


class REPCA1(REPCA1Data, REPCA1Model):
    """
    REPCA1: renewable energy power plat control model.

    The output of the model, ``Pext`` and ``Qext``,  are the increment signals
    of active and reactive power for the electrical control model.

    Notes for PSS/E DYR parser:

    1. If ICONs M+1 and M+2 are set to 0 when using generator power, an error
       will be thrown by the parser, saying "<REPCA1> cannot retrieve <bus1>
       from <ACLine> using <line>: KeyError('Group <ACLine> does not contain
       device with idx=False')". Manual effort is required to run the converted
       file. In the REPCA1 sheet, provide the idx of a line that connects to the
       RenGen bus.

    2. PSS/E enters ICONs M+3 as a string in single quotes. The pair of single
       quotes need to be removed, or the conversion will fail.
    """

    def __init__(self, system, config):
        REPCA1Data.__init__(self)
        REPCA1Model.__init__(self, system, config)
