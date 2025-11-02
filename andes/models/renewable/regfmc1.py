"""
REGFMC1 - Hybrid Grid-Forming Converter Model.

This model implements a parallel combination of:
- Grid-forming (GFM) voltage source with series impedance
- Grid-following (GFL) current source
"""

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtService, IdxParam,
                        Lag, Model, ModelData, NumParam, State, Switcher)
from andes.core.block import PIController, Washout


class REGFMC1Data(ModelData):
    """
    REGFMC1 model data.
    """

    def __init__(self):
        ModelData.__init__(self)

        # --- General Parameters ---
        self.bus = IdxParam(model='Bus',
                            info="Interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="Static generator index",
                            mandatory=True,
                            )
        self.Sn = NumParam(default=100.0, tex_name='S_n',
                           info='Model MVA base',
                           unit='MVA',
                           )
        self.gammap = NumParam(default=1.0,
                               info="P ratio of linked static gen",
                               tex_name=r'\gamma_P'
                               )
        self.gammaq = NumParam(default=1.0,
                               info="Q ratio of linked static gen",
                               tex_name=r'\gamma_Q'
                               )

        # --- Circuit Parameters ---
        self.Rs = NumParam(default=0.0,
                           info="Series resistance for GFM branch",
                           z=True,
                           tex_name='R_s'
                           )
        self.Xs = NumParam(default=0.1,
                           info="Series reactance for GFM branch",
                           z=True,
                           tex_name='X_s'
                           )

        # --- GFM Voltage Control Parameters ---
        self.Tvr = NumParam(default=0.02,
                            tex_name='T_{vr}',
                            info='Time constant for Vref filter',
                            unit='s',
                            )
        self.kq = NumParam(default=1.0,
                           tex_name='k_q',
                           info='Reactive power gain in voltage control',
                           )
        self.mq = NumParam(default=1.0,
                           tex_name='m_q',
                           info='Reactive power measurement gain',
                           )
        self.kpE = NumParam(default=0.1,
                            tex_name='k_{pE}',
                            info='Proportional gain for voltage magnitude error',
                            )
        self.kiE = NumParam(default=5.0,
                            tex_name='k_{iE}',
                            info='Integral gain for voltage magnitude error',
                            )
        self.Tvs = NumParam(default=0.02,
                            tex_name='T_{vs}',
                            info='Time constant for voltage controller output filter',
                            unit='s',
                            )
        self.dEmax = NumParam(default=0.2,
                              tex_name=r'\Delta E_{max}',
                              info='Maximum voltage magnitude deviation (PLACEHOLDER)',
                              )
        self.dEmin = NumParam(default=-0.2,
                              tex_name=r'\Delta E_{min}',
                              info='Minimum voltage magnitude deviation (PLACEHOLDER)',
                              )

        # --- GFM VSM Control Parameters ---
        self.fn = NumParam(default=60.0,
                           info="System frequency",
                           tex_name='f_n',
                           unit='Hz',
                           )
        self.Tomegar = NumParam(default=0.02,
                                tex_name=r'T_{\omega r}',
                                info='Time constant for omega_ref filter',
                                unit='s',
                                )
        self.mp = NumParam(default=0.05,
                           tex_name='m_p',
                           info='Active power droop gain',
                           )
        self.Tomegacmd = NumParam(default=0.02,
                                  tex_name=r'T_{\omega cmd}',
                                  info='Time constant for omega command filter',
                                  unit='s',
                                  )
        self.Tfrq = NumParam(default=0.02,
                             tex_name=r'T_{frq}',
                             info='Time constant for power measurement filter',
                             unit='s',
                             )
        self.Hs = NumParam(default=5.0,
                           tex_name='H_s',
                           info='Inertia constant (2H)',
                           unit='s',
                           )
        self.D1 = NumParam(default=0.0,
                           tex_name='D_1',
                           info='Primary damping coefficient',
                           )
        self.D2 = NumParam(default=0.0,
                           tex_name='D_2',
                           info='Secondary damping coefficient',
                           )
        self.omegaD = NumParam(default=1.0,
                               tex_name=r'\omega_D',
                               info='Damping filter frequency',
                               unit='rad/s',
                               )
        self.domegamax = NumParam(default=0.1,
                                  tex_name=r'\Delta\omega_{max}',
                                  info='Maximum frequency deviation (PLACEHOLDER)',
                                  )
        self.domegamin = NumParam(default=-0.1,
                                  tex_name=r'\Delta\omega_{min}',
                                  info='Minimum frequency deviation (PLACEHOLDER)',
                                  )
        self.dPGFMmax = NumParam(default=1.0,
                                 tex_name=r'\Delta P_{GFM,max}',
                                 info='Maximum active power deviation for GFM (PLACEHOLDER)',
                                 )
        self.dPGFMmin = NumParam(default=-1.0,
                                 tex_name=r'\Delta P_{GFM,min}',
                                 info='Minimum active power deviation for GFM (PLACEHOLDER)',
                                 )
        self.Tpf = NumParam(default=0.02,
                            tex_name='T_{pf}',
                            info='Time constant for frequency flag filter',
                            unit='s',
                            )
        self.FFFlag = NumParam(default=0.0,
                               tex_name='FF_{Flag}',
                               info='Frequency flag (0 or 1)',
                               unit='bool',
                               )

        # --- GFL Control Parameters ---
        self.Tvf = NumParam(default=0.02,
                            tex_name='T_{vf}',
                            info='Time constant for voltage filter in GFL',
                            unit='s',
                            )
        self.kgv = NumParam(default=1.0,
                            tex_name='k_{gv}',
                            info='Voltage error gain in GFL',
                            )
        self.dbVLI = NumParam(default=-0.02,
                              tex_name='db_{VLI}',
                              info='Voltage deadband lower limit (PLACEHOLDER)',
                              )
        self.dbVHI = NumParam(default=0.02,
                              tex_name='db_{VHI}',
                              info='Voltage deadband upper limit (PLACEHOLDER)',
                              )
        self.Pcmd_GFL_max = NumParam(default=999.0,
                                     tex_name='P_{cmd,GFL,max}',
                                     info='Maximum active power command for GFL (PLACEHOLDER)',
                                     )
        self.Pcmd_GFL_min = NumParam(default=0.0,
                                     tex_name='P_{cmd,GFL,min}',
                                     info='Minimum active power command for GFL (PLACEHOLDER)',
                                     )
        self.Qcmd_GFL_max = NumParam(default=999.0,
                                     tex_name='Q_{cmd,GFL,max}',
                                     info='Maximum reactive power command for GFL (PLACEHOLDER)',
                                     )
        self.Qcmd_GFL_min = NumParam(default=-999.0,
                                     tex_name='Q_{cmd,GFL,min}',
                                     info='Minimum reactive power command for GFL (PLACEHOLDER)',
                                     )
        self.Ipmax_GFL = NumParam(default=999.0,
                                  tex_name='I_{pmax,GFL}',
                                  info='Maximum active current for GFL (PLACEHOLDER)',
                                  )
        self.Ipmin_GFL = NumParam(default=0.0,
                                  tex_name='I_{pmin,GFL}',
                                  info='Minimum active current for GFL (PLACEHOLDER)',
                                  )
        self.Iqmax_GFL = NumParam(default=999.0,
                                  tex_name='I_{qmax,GFL}',
                                  info='Maximum reactive current for GFL (PLACEHOLDER)',
                                  )
        self.Iqmin_GFL = NumParam(default=-999.0,
                                  tex_name='I_{qmin,GFL}',
                                  info='Minimum reactive current for GFL (PLACEHOLDER)',
                                  )
        self.PQFlag = NumParam(default=1.0,
                               tex_name='PQ_{Flag}',
                               info='PQ priority flag (0 or 1)',
                               unit='bool',
                               )

        # --- Current Limiting Parameters ---
        self.Imax = NumParam(default=1.5,
                             tex_name='I_{max}',
                             info='Maximum total output current',
                             current=True,
                             )
        self.Vmin = NumParam(default=0.01,
                             tex_name='V_{min}',
                             info='Minimum voltage for current limiting (PLACEHOLDER)',
                             )


class REGFMC1Model(Model):
    """
    REGFMC1 model implementation.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'RenGen'

        # --- External References ---
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle',
                          e_str='-u * Pe',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name='V',
                          info='Bus voltage magnitude',
                          e_str='-u * Qe',
                          )

        self.p0s = ExtService(model='StaticGen',
                              src='p',
                              indexer=self.gen,
                              tex_name=r'P_{0s}',
                              info='Total P of the static gen',
                              )
        self.q0s = ExtService(model='StaticGen',
                              src='q',
                              indexer=self.gen,
                              tex_name=r'Q_{0s}',
                              info='Total Q of the static gen',
                              )

        # --- Initialization Services ---
        self.p0 = ConstService(v_str='gammap * p0s',
                               tex_name='P_0',
                               info='Initial P for this device',
                               )
        self.q0 = ConstService(v_str='gammaq * q0s',
                               tex_name='Q_0',
                               info='Initial Q for this device',
                               )

        # Initial current calculations (for both branches)
        self.Id0_GFL = ConstService(tex_name=r'I_{d0,GFL}',
                                    v_str='u * p0 / v',
                                    )
        self.Iq0_GFL = ConstService(tex_name=r'I_{q0,GFL}',
                                    v_str='u * q0 / v',
                                    )

        # Damping washout time constant (1 / omegaD)
        self.Tdamp = ConstService(tex_name=r'T_{damp}',
                                  v_str='1 / omegaD',
                                  info='Damping washout time constant',
                                  )

        # Voltage reference for GFL - initialized from bus voltage
        self.Vref0 = ConstService(v_str='v',
                                  tex_name='V_{ref0}',
                                  info='Reference voltage for GFL',
                                  )

        # GFM branch impedance squared (for current calculation)
        self.Zs2 = ConstService(v_str='Rs**2 + Xs**2',
                                tex_name='Z_s^2',
                                info='GFM series impedance magnitude squared',
                                )

        # --- External reference variables (to be controlled by plant controller) ---
        # GFM voltage reference (external input to voltage control)
        self.Vref_GFM = Algeb(tex_name='V_{ref,GFM}',
                              info='Voltage reference for GFM branch (from plant controller)',
                              v_str='v',
                              e_str='v - Vref_GFM',  # Default: maintain initial bus voltage
                              )

        # GFM frequency reference (external input to VSM control)
        self.fref_GFM = Algeb(tex_name='f_{ref,GFM}',
                              info='Frequency reference for GFM branch (from plant controller)',
                              v_str='1.0',
                              e_str='1.0 - fref_GFM',  # Default: maintain nominal frequency
                              )

        # --- GFM Branch: Voltage Control ---
        # Voltage reference filter
        self.VrefLag = Lag(u='Vref_GFM', T=self.Tvr, K=1,
                           info='Voltage reference filter',
                           name='VrefLag',
                           )

        # Inverter voltage filter
        self.VinvLag = Lag(u='v', T=self.Tvr, K=1,
                           info='Inverter voltage filter',
                           name='VinvLag',
                           )

        # Reactive power measurement path (PLACEHOLDER - simplified)
        self.Iqref_GFM = Algeb(tex_name='I_{qref,GFM}',
                               info='Reactive current reference for GFM',
                               v_str='0',
                               e_str='0 - Iqref_GFM',  # Initialize to zero
                               )

        # Voltage magnitude error
        self.Verr = Algeb(tex_name='V_{err}',
                          info='Voltage magnitude error',
                          v_str='0',
                          e_str='(VrefLag_y - VinvLag_y) * kq / mq + Iqref_GFM - Verr',
                          )

        # PI controller for voltage magnitude
        self.VmagPI = PIController(u=self.Verr,
                                    kp=self.kpE,
                                    ki=self.kiE,
                                    x0='v',
                                    info='Voltage magnitude PI controller',
                                    name='VmagPI',
                                    )

        # Output filter for EVSM (TODO: add limiters for dEmax, dEmin)
        self.EVSMLag = Lag(u='VmagPI_y',
                           T=self.Tvs,
                           K=1,
                           info='EVSM output filter',
                           name='EVSMLag',
                           )

        # EVSM is the output of EVSMLag
        self.EVSM = Algeb(tex_name='E_{VSM}',
                          info='GFM voltage magnitude',
                          v_str='v',
                          e_str='EVSMLag_y - EVSM',
                          )

        # --- GFM Branch: VSM Control ---
        # Omega reference filter
        self.OmegarefLag = Lag(u='fref_GFM',
                               T=self.Tomegar,
                               K=1,
                               info='Omega reference filter',
                               name='OmegarefLag',
                               )

        # Active power reference - can be controlled by plant controller
        self.Pref_GFM = Algeb(v_str='0',
                              e_str='0 - Pref_GFM',  # Defaults to 0, can be overridden externally
                              tex_name='P_{ref,GFM}',
                              info='Active power reference for GFM',
                              )

        # Plant controller changes omega_ref (PLACEHOLDER - use constant 1 for now)
        self.omega_ref = Algeb(tex_name=r'\omega_{ref}',
                               info='Omega reference',
                               v_str='1.0',
                               e_str='OmegarefLag_y - omega_ref',
                               )

        # GFM branch power measurement (for droop feedback)
        self.Pmv_GFM = Lag(u='PGFM',
                           T=self.Tfrq,
                           K=1,
                           info='Measured GFM power for droop control',
                           name='Pmv_GFM',
                           )

        # Frequency droop: converts frequency error to power command
        # dP_GFM_droop = (omega_ref - omegam) / mp
        self.dP_GFM_droop = Algeb(tex_name=r'\Delta P_{GFM,droop}',
                                  info='Power command from frequency droop',
                                  v_str='0',
                                  e_str='(OmegarefLag_y - omegam) / mp - dP_GFM_droop',
                                  )

        # Power command for GFM branch
        # Pcmd_GFM = Pref_GFM + dP_GFM_droop (reference + droop correction)
        self.Pcmd_GFM = Algeb(tex_name='P_{cmd,GFM}',
                              info='Power command for GFM branch',
                              v_str='Pref_GFM',
                              e_str='Pref_GFM + dP_GFM_droop - Pcmd_GFM',
                              )

        # Damping filter: sD2/(s+omegaD) using Washout
        # Washout implements sK/(1+sT), so we need K=D2, T=1/omegaD
        self.DampWash = Washout(u='omegam',
                                T=self.Tdamp,
                                K=self.D2,
                                info='Damping washout filter',
                                name='DampWash',
                                )

        # Inverter active power for GFM - measured at voltage source (for swing equation)
        self.Pinv_GFM = Algeb(tex_name='P_{inv,GFM}',
                              info='GFM inverter active power at voltage source',
                              v_str='0',
                              e_str='(EVSM * cos(dVSM - a) * Id_VSM + EVSM * sin(dVSM - a) * Iq_VSM) - Pinv_GFM',
                              )

        # Virtual machine angular frequency (swing equation)
        # Power balance: 2*Hs*d(Δω)/dt = P_cmd - P_inv - D1*Δω - D2*d(Δω)/dt
        # Since omegam is absolute frequency: Δω = omegam - 1.0
        # Equation: d(omegam)/dt = (Pcmd_GFM - Pinv_GFM - D1*(omegam-1) - DampWash_y) / (2*Hs)
        self.omegam = State(
            info='Virtual machine angular frequency (pu)',
            tex_name=r'\omega_m',
            v_str='1.0',
            e_str='(Pcmd_GFM - Pinv_GFM - D1 * (omegam - 1.0) - DampWash_y) / 2',
            t_const=self.Hs,
        )

        # Virtual synchronous machine angle (integration of omega deviation)
        self.dVSM = State(
            info='Virtual synchronous machine angle',
            tex_name=r'\delta_{VSM}',
            v_str='a',
            e_str='2 * pi * fn * (omegam - 1.0)',
        )

        # --- GFL Branch: Control ---
        # Voltage filter for GFL
        self.VinvGFLLag = Lag(u='v',
                              T=self.Tvf,
                              K=1,
                              info='GFL voltage filter',
                              name='VinvGFLLag',
                              )

        # Voltage error for GFL (TODO: add deadband)
        self.Verr_GFL = Algeb(tex_name='V_{err,GFL}',
                              info='Voltage error for GFL',
                              v_str='Vref0 - v',
                              e_str='Vref0 - VinvGFLLag_y - Verr_GFL',
                              )

        # Active and reactive power commands (controlled by plant controller)
        # Default equations lock to p0/q0, but can be overridden externally
        self.Pcmd_GFL = Algeb(tex_name='P_{cmd,GFL}',
                              info='Active power command for GFL',
                              v_str='p0',
                              e_str='p0 - Pcmd_GFL',  # Defaults to p0, can be overridden externally
                              )

        self.Qcmd_GFL = Algeb(tex_name='Q_{cmd,GFL}',
                              info='Reactive power command for GFL',
                              v_str='q0',
                              e_str='q0 - Qcmd_GFL',  # Defaults to q0, can be overridden externally
                              )

        # Current commands (PLACEHOLDER - TODO: add limiters and PQ priority)
        self.Ipcmd_GFL = Algeb(tex_name='I_{pcmd,GFL}',
                               info='Active current command for GFL',
                               v_str='Id0_GFL',
                               e_str='Pcmd_GFL / v - Ipcmd_GFL',
                               )

        self.Iqcmd_GFL = Algeb(tex_name='I_{qcmd,GFL}',
                               info='Reactive current command for GFL',
                               v_str='kgv * (Vref0 - v) + Iq0_GFL',
                               e_str='kgv * Verr_GFL + Qcmd_GFL / v - Iqcmd_GFL',
                               )

        # Current outputs (PLACEHOLDER - no limiting yet)
        self.Ip_GFL = Algeb(tex_name='I_{p,GFL}',
                            info='Active current output for GFL',
                            v_str='Id0_GFL',
                            e_str='Ipcmd_GFL - Ip_GFL',
                            )

        self.Iq_GFL = Algeb(tex_name='I_{q,GFL}',
                            info='Reactive current output for GFL',
                            v_str='kgv * (Vref0 - v) + Iq0_GFL',
                            e_str='Iqcmd_GFL - Iq_GFL',
                            )

        # --- Current Calculation (PLACEHOLDER - simplified) ---
        # GFM branch current magnitude (simplified)
        self.IVSM_mag = Algeb(tex_name='I_{VSM,mag}',
                              info='GFM branch current magnitude (PLACEHOLDER)',
                              v_str='0',
                              e_str='0 - IVSM_mag',  # To be calculated
                              )

        # GFM branch current angle (simplified)
        self.IVSM_ang = Algeb(tex_name=r'\phi_{VSM}',
                              info='GFM branch current angle (PLACEHOLDER)',
                              v_str='0',
                              e_str='0 - IVSM_ang',  # To be calculated
                              )

        # GFL branch current magnitude
        self.IGFL_mag = Algeb(tex_name='I_{GFL,mag}',
                              info='GFL branch current magnitude',
                              v_str='sqrt(Id0_GFL**2 + (kgv * (Vref0 - v) + Iq0_GFL)**2)',
                              e_str='sqrt(Ip_GFL**2 + Iq_GFL**2) - IGFL_mag',
                              )

        # Total current magnitude (PLACEHOLDER - vector sum needed)
        self.Itotal = Algeb(tex_name='I_{total}',
                            info='Total current magnitude (PLACEHOLDER)',
                            v_str='sqrt(Id0_GFL**2 + (kgv * (Vref0 - v) + Iq0_GFL)**2)',
                            e_str='IGFL_mag - Itotal',  # Simplified, should be vector sum
                            )

        # Scaling factor for current limiting (PLACEHOLDER)
        self.k_factor = Algeb(tex_name='k_{factor}',
                              info='Current scaling factor (PLACEHOLDER)',
                              v_str='1.0',
                              e_str='1.0 - k_factor',  # No limiting initially
                              )

        # --- Power Calculations ---
        # GFM branch current in dq-frame
        # Voltage source EVSM∠dVSM behind impedance Rs+jXs to bus V∠a
        # In dq-frame (d-axis aligned with bus voltage V∠a):
        #   delta = dVSM - a (angle difference)
        #   Ed_VSM = EVSM * cos(delta), Eq_VSM = EVSM * sin(delta)
        #   Id_VSM = ((Ed_VSM - v)*Rs + Eq_VSM*Xs) / Zs2
        #   Iq_VSM = (Eq_VSM*Rs - (Ed_VSM - v)*Xs) / Zs2

        self.Id_VSM = Algeb(tex_name='I_{d,VSM}',
                            info='GFM d-axis current',
                            v_str='0',
                            e_str='((EVSM * cos(dVSM - a) - v) * Rs + EVSM * sin(dVSM - a) * Xs) / Zs2 - Id_VSM',
                            )

        self.Iq_VSM = Algeb(tex_name='I_{q,VSM}',
                            info='GFM q-axis current',
                            v_str='0',
                            e_str='(EVSM * sin(dVSM - a) * Rs - (EVSM * cos(dVSM - a) - v) * Xs) / Zs2 - Iq_VSM',
                            )

        # GFM branch power at bus terminals (for bus injection)
        # In dq frame aligned with bus: Vd=v, Vq=0
        # P = Vd*Id + Vq*Iq = v*Id_VSM
        # Q = Vq*Id - Vd*Iq = -v*Iq_VSM (negative sign for generator convention)
        self.PGFM = Algeb(tex_name='P_{GFM}',
                          info='GFM branch active power at bus',
                          v_str='0',
                          e_str='v * Id_VSM - PGFM',
                          )

        self.QGFM = Algeb(tex_name='Q_{GFM}',
                          info='GFM branch reactive power at bus',
                          v_str='0',
                          e_str='-v * Iq_VSM - QGFM',
                          )

        # GFL branch power
        self.PGFL = Algeb(tex_name='P_{GFL}',
                          info='GFL branch active power',
                          v_str='p0',
                          e_str='v * Ip_GFL - PGFL',
                          )

        self.QGFL = Algeb(tex_name='Q_{GFL}',
                          info='GFL branch reactive power',
                          v_str='v * kgv * (Vref0 - v) + q0',
                          e_str='v * Iq_GFL - QGFL',
                          )

        # Total power injection
        self.Pe = Algeb(tex_name='P_e',
                        info='Total active power injection',
                        v_str='p0',
                        e_str='PGFM + PGFL - Pe',
                        )

        self.Qe = Algeb(tex_name='Q_e',
                        info='Total reactive power injection',
                        v_str='q0',
                        e_str='QGFM + QGFL - Qe',
                        )

    def v_numeric(self, **kwargs):
        """
        Disable the corresponding StaticGen.
        """
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class REGFMC1(REGFMC1Data, REGFMC1Model):
    """
    Hybrid Grid-Forming Converter Model (REGFMC1).

    This model represents a parallel combination of:
    - Grid-forming (GFM) voltage source with series impedance and VSM control
    - Grid-following (GFL) current source

    Notes
    -----
    - Current implementation has PLACEHOLDER sections for:
      - Voltage and frequency limiters
      - Current limiters with PQ priority
      - Deadband for GFL voltage error
      - Complete current limiting logic
      - Full GFM branch current calculations

    - Initialization:
      - GFM branch: Pref_GFM = 0, reactive power = 0
      - GFL branch: Pcmd_GFL = p0, Qcmd_GFL = q0
      - EVSM = V, dVSM = bus angle
    """

    def __init__(self, system, config):
        REGFMC1Data.__init__(self)
        REGFMC1Model.__init__(self, system, config)
