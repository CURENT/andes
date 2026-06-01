"""
REGFMC1 - Hybrid Grid-Forming Converter Model.

This model implements a parallel combination of:
- Grid-forming (GFM) voltage source with series impedance
- Grid-following (GFL) current source
"""

from andes.core import (
    Algeb, ConstService, ExtAlgeb, ExtService, IdxParam,
    Lag, Model, ModelData, NumParam, State, Switcher,
    Limiter, Piecewise
)

from andes.core.service import NumSelect, VarService
from andes.core.block import DeadBand1, PIController, Washout


class REGFMC1Data(ModelData):
    """
    REGFMC1 model data.
    """

    def __init__(self):
        ModelData.__init__(self)

        # --- General Parameters ---
        self.bus = IdxParam(model='Bus',
                            info="Inverter terminal/interface bus id",
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
        self.Rs = NumParam(default=0.05,
                           info="Series resistance for GFM branch",
                           z=True,
                           tex_name='R_s'
                           )
        self.Xs = NumParam(default=0.2,
                           info="Series reactance for GFM branch",
                           z=True,
                           tex_name='X_s'
                           )

        # --- GFM Voltage Control Parameters ---
        self.Tvr = NumParam(default=0.025,
                            tex_name='T_{vr}',
                            info='Time constant for Vref filter',
                            unit='s',
                            )
        self.Tomegam = NumParam(default=0.3183,
                                tex_name=r'T_{\omega m}',
                                info='Time constant for omegam filter',
                                unit='s',
                                )

        self.kq = NumParam(default=1,
                           tex_name='k_q',
                           info='Reactive power gain in voltage control',
                           current=True,
                           )
        self.mq = NumParam(default=0.4,
                           tex_name='m_q',
                           info='Reactive power measurement gain',
                           )
        self.kpE = NumParam(default=0.333,
                            tex_name='k_{pE}',
                            info='Proportional gain for voltage magnitude error',
                            ipower=True,
                            )
        self.kiE = NumParam(default=3.333,
                            tex_name='k_{iE}',
                            info='Integral gain for voltage magnitude error',
                            ipower=True,
                            )
        self.Tvsm = NumParam(default=0.318,
                            tex_name='T_{vsm}',
                            info='Time constant for voltage controller output filter',
                            unit='s',
                            )
        self.dEmax = NumParam(default=0.2083,
                              tex_name=r'\Delta E_{max}',
                              info='Maximum voltage magnitude deviation (PLACEHOLDER)',
                              )
        self.dEmin = NumParam(default=-0.2083,
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
        self.TIf = NumParam(default=0.02,
                            tex_name=r'T_{If}',
                            info='Time constant for I_d and I_q filter',
                            unit='s',
                            )

        self.mp = NumParam(default=0.041667,
                           tex_name='m_p',
                           info='Active power droop gain',
                           ipower=True,
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
        self.Hs = NumParam(default=1.6,
                           tex_name='H_s',
                           info='Inertia constant (2H)',
                           unit='s',
                           power=True,
                           )
        self.D1 = NumParam(default=5,
                           tex_name='D_1',
                           info='Primary damping coefficient',
                           power=True,
                           )
        self.D2 = NumParam(default=90/3.14,
                           tex_name='D_2',
                           info='Secondary damping coefficient',
                           power=True,
                           )
        self.omegaD = NumParam(default=3.14,
                               tex_name=r'\omega_D',
                               info='Damping filter frequency',
                               unit='rad/s',
                               )
        self.domegamax = NumParam(default=0.1667,
                                  tex_name=r'\Delta\omega_{max}',
                                  info='Maximum frequency deviation (PLACEHOLDER)',
                                  )
        self.domegamin = NumParam(default=-0.3333,
                                  tex_name=r'\Delta\omega_{min}',
                                  info='Minimum frequency deviation (PLACEHOLDER)',
                                  )
        self.dPGFMmax = NumParam(default=1.36,
                                 tex_name=r'\Delta P_{GFM,max}',
                                 info='Maximum active power deviation for GFM (PLACEHOLDER)',
                                 power=True,
                                 )
        self.dPGFMmin = NumParam(default=-1.36,
                                 tex_name=r'\Delta P_{GFM,min}',
                                 info='Minimum active power deviation for GFM (PLACEHOLDER)',
                                 power=True,
                                 )
        self.Tpf = NumParam(default=0.02,
                            tex_name='T_{pf}',
                            info='Time constant for frequency flag filter',
                            unit='s',
                            )
        self.FFFlag = NumParam(default=1.0,
                               tex_name='FF_{Flag}',
                               info='Frequency droop flag; 0-disable, 1-enable',
                               unit='bool',
                               )

        # --- GFL Control Parameters ---
        self.Tvf = NumParam(default=0.02,
                            tex_name='T_{vf}',
                            info='Time constant for voltage filter in GFL',
                            unit='s',
                            )
        self.kqv = NumParam(default=2,        # modified
                            tex_name='k_{qv}',
                            info='Voltage error gain in GFL',
                            current=True,
                            )
        self.dbVLI = NumParam(default=-0.12,
                              tex_name='db_{VLI}',
                              info='Voltage deadband lower limit (PLACEHOLDER)',
                              )
        self.dbVHI = NumParam(default=0.12,
                              tex_name='db_{VHI}',
                              info='Voltage deadband upper limit (PLACEHOLDER)',
                              )
        
        
        # --- Current Limiting Parameters ---
        
        self.Imax = NumParam(default=1.2,
                             tex_name='I_{max}',
                             info='Maximum total output current',
                             current=True,
                             )       
        
        self.PQFlag = NumParam(default=1.0,
                               tex_name='PQ_{Flag}',
                               info='1=P priority, 0=Q priority',
                               unit='bool',
                               )


        self.Vmin = NumParam(default=0.88,
                             tex_name='V_{min}',
                             info='Minimum voltage for current limiting (PLACEHOLDER)',
                             )
        
        self.Pcmd_GFL_max = NumParam(default=1.2,
                                     tex_name='P_{cmd,GFL,max}',
                                     info='Maximum active power command for GFL',
                                     unit='p.u.',
                                     power=True,
                                     )

        self.Pcmd_GFL_min = NumParam(default=-1.0,
                                     tex_name='P_{cmd,GFL,min}',
                                     info='Minimum active power command for GFL',
                                     unit='p.u.',
                                     power=True,
                                     )
        self.Qcmd_GFL_max = NumParam(default=0.6,
                                     tex_name='Q_{cmd,GFL,max}',
                                     info='Maximum reactive power command for GFL',
                                     unit='p.u.',
                                     power=True,
                                     )

        self.Qcmd_GFL_min = NumParam(default=-0.6,
                                     tex_name='Q_{cmd,GFL,min}',
                                     info='Minimum reactive power command for GFL',
                                     unit='p.u.',
                                     power=True,
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
                          info='Inverter terminal/interface bus voltage angle',
                          e_str='-u * Pe',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name='V',
                          info='Inverter terminal/interface bus voltage magnitude',
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
        # Power, current, and impedance parameters marked with ``power=True``,
        # ``current=True``, and ``z=True`` are converted by ANDES to system base.
        # Keep the internal controller equations on system base as well.
        self.p0 = ConstService(v_str='gammap * p0s',
                               tex_name='P_0',
                               info='Initial P for this device on system base',
                               )
        self.q0 = ConstService(v_str='gammaq * q0s',
                               tex_name='Q_0',
                               info='Initial Q for this device on system base',
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
                                info='GFM series impedance magnitude squared on system base',
                                )

        # --- External reference variables (to be controlled by plant controller) ---
        # GFM voltage reference (external input to voltage control)             
        self.Vref_GFM = Algeb(tex_name='V_{ref,GFM}',
                              info='Voltage reference for GFM branch (from plant controller)',
                              v_str='Vref0',  # modified
                              e_str='Vref0 - Vref_GFM',  # Default: maintain initial bus voltage
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


        # Reactive power measurement path (per diagram: Iq_GFM filtered through 1/(Tif*s+1))
        self.Iq_VSMLag = Lag(u='Iq_VSM', T=self.TIf, K=1, info='Filter for I_q GFM', name='Iq_VSMLag')

        
        # Proportional part
        # Voltage magnitude error
        self.Verr = Algeb(
            tex_name='V_{err}',
            info='Voltage magnitude error',
            v_str='0',
            e_str='(VrefLag_y - VinvLag_y) * kq / mq - Iq_VSMLag_y - Verr',
        )

        # Proportional part
        self.VmagP = Algeb(
            tex_name='V_{P}',
            info='Proportional part of voltage PI',
            v_str='0',
            e_str='kpE * Verr - VmagP',
        )

        # Integral state
        self.VmagI = State(
            tex_name='V_{I}',
            info='Integral state of voltage PI',
            v_str='0',
            e_str='kiE * Verr',
        )

        # Limit the integral output itself to [dEmin, dEmax]
        self.VmagI_lim = Limiter(
            u=self.VmagI,
            lower=self.dEmin,
            upper=self.dEmax,
            name='VmagI_lim'
        )

        self.VmagI_lim_val = Algeb(
            tex_name='V_{I,lim}',
            info='Limited integral output',
            v_str='0',
            e_str='VmagI * VmagI_lim_zi + dEmax * VmagI_lim_zu + dEmin * VmagI_lim_zl - VmagI_lim_val',
        )

        # Unsaturated PI output
        self.VmagPI_raw = Algeb(
            tex_name='VmagPI_{raw}',
            info='Raw PI output',
            v_str='0',
            e_str='VmagP + VmagI_lim_val - VmagPI_raw',
        )

        # Also limit the final PI output to [dEmin, dEmax]
        self.VmagPI_lim = Limiter(
            u=self.VmagPI_raw,
            lower=self.dEmin,
            upper=self.dEmax,
            name='VmagPI_lim'
        )

        self.VmagPI_lim_val = Algeb(
            tex_name='VmagPI_{sat,val}',
            info='Limited PI output',
            v_str='0',
            e_str='VmagPI_raw * VmagPI_lim_zi + dEmax * VmagPI_lim_zu + dEmin * VmagPI_lim_zl - VmagPI_lim_val',
        )

        # Final EVSM command
        self.EVSM_initial = Algeb(
            tex_name='E_{VSM_initial}',
            info='EVSM_initial',
            v_str='v',
            e_str='VmagPI_lim_val + VrefLag_y - EVSM_initial',
        )

        # Output filter for EVSM 
        self.EVSMLag = Lag(u='EVSM_initial',
                           T=self.Tvsm,
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

        # GFM branch power measurement (for droop feedback)
        self.Pinv_GFM = Algeb(            
            tex_name='P_{inv,GFM}',
            info='GFM inverter active power at limited internal voltage source',
            v_str='0',
            e_str='Ed_VSM_lim * Id_VSM_lim + Eq_VSM_lim * Iq_VSM_lim - Pinv_GFM',
        )

        # Frequency droop: converts frequency error to power command
        # dP_GFM_droop = (omegarefLag_y - omegamLag_y) / mp
        self.SWFF = Switcher(u=self.FFFlag, options=(0, 1), tex_name='SW_{FF}', cache=True)

        self.dP_GFM_droop = Algeb(tex_name=r'\Delta P_{GFM,droop}',
                                  info='Power command from frequency droop',
                                  v_str='0',
                                  e_str='SWFF_s1 * (OmegarefLag_y - omegamLag_y) / mp - dP_GFM_droop',
                                  )

        # Power command for GFM branch
        # Pcmd_GFM = Pref_GFM + dP_GFM_droop (reference + droop correction)
        self.Pcmd_GFM = Algeb(tex_name='P_{cmd,GFM}',
                              info='Power command for GFM branch',
                              v_str='Pref_GFM',
                              e_str='Pref_GFM + dP_GFM_droop - Pcmd_GFM',
                              )

        # Damping filter: sD2/(s+omegaD) using Washout
        # Washout implements sK/(1+sT), so K=D2, T=1/omegaD
        self.domegam = Algeb(                    
            name='domegam',
            tex_name=r'\Delta\omega',
            info='Frequency deviation (pu)',
            v_str='0.0',
            e_str='omegam - 1.0 - domegam',
        )

        self.DampWash = Washout(u='domegam',
                                T=self.Tdamp,
                                K=self.D2,  
                                info='Damping washout filter',
                                name='DampWash',
                                )

        # Inverter active power for GFM - measured at voltage source 
        self.PGFM_pre = Algeb(
            tex_name='P_{GFM,pre}',
            info='Pre-limited GFM branch active power at bus',
            v_str='0',
            e_str='v * Id_VSM - PGFM_pre',
        )       
              
        self.PGFM_pre_Lag = Lag(u='PGFM_pre',            
                              T=self.Tpf,
                              K=1,
                              info='Pre-limited GFM branch active power at bus',
                              name='PGFM_pre_Lag',
                              )

        # Virtual machine angular frequency (swing equation)
        # Power balance: 2*Hs*d(Δω)/dt = P_cmd - P_inv - D1*Δω - D2*d(Δω)/dt
        # Since omegam is absolute frequency: Δω = omegam - 1.0
        # Equation: d(omegam)/dt = (Pcmd_GFM - Pinv_GFM - D1*(omegam-1) - DampWash_y) / (2*Hs)
        self.omegam = State(
            info='Virtual machine angular frequency (pu)',
            tex_name=r'\omega_m',
            v_str='1.0',
            e_str='(Pcmd_GFM - PGFM_pre_Lag_y - D1 * domegam - DampWash_y) / 2',
            t_const=self.Hs,
        )

        self.omegamLag = Lag(u='omegam', T=self.Tomegam, K=1, info='Filter for omegam', name='omegamLag')

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
                              )   # same as VinvLag_y

        # Voltage error for GFL
        self.Verr_GFL = Algeb(tex_name='V_{err,GFL}',
                              info='Voltage error for GFL',
                              v_str='Vref0 - v',
                              e_str='Vref0 - VinvGFLLag_y - Verr_GFL',   # Vref0 or VrefLag_y? shouldbe Vref0!
                              )

        # Offset piecewise-linear deadband for GFL voltage error:
        # y = x-dbVHI (x > dbVHI), 0 (dbVLI <= x <= dbVHI), x-dbVLI (x < dbVLI)
        self.Verr_GFL_dbd = DeadBand1(u=self.Verr_GFL, center=0.0,
                                      lower=self.dbVLI, upper=self.dbVHI,
                                      name='Verr_GFL_dbd',
                                      tex_name='V_{err,GFL,dbd}',
                                      info='Deadbanded voltage error for GFL')

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

        # Current commands
        self.Ipcmd_GFL = Algeb(tex_name='I_{pcmd,GFL}',
                               info='Active current command for GFL',
                               v_str='Id0_GFL',
                               e_str='Pcmd_GFL / v - Ipcmd_GFL',
                               )

        self.Iqcmd_GFL = Algeb(tex_name='I_{qcmd,GFL}',
                               info='Reactive current command for GFL',
                               v_str='kqv * Verr_GFL_dbd_y + Iq0_GFL',
                               e_str='kqv * Verr_GFL_dbd_y + Qcmd_GFL / v - Iqcmd_GFL',
                               )
        
        self.Ipmaxsq0_GFL1 = ConstService(
            v_str='Piecewise((0, Le(Imax**2 - (kqv * (Vref0 - v) + Iq0_GFL) **2, 0.0)), (Imax**2 - (kqv * (Vref0 - v) + Iq0_GFL)**2 , True), evaluate=False)'
        )

        self.Iqmaxsq0_GFL1 = ConstService(
            v_str='Piecewise((0, Le(Imax**2 - Id0_GFL**2, 0.0)), (Imax**2 - Id0_GFL**2, True), evaluate=False)'
        )
                
        self.Ipmaxsq_GFL1 = VarService(
            v_str='Piecewise((0, Le(Imax**2 - Iqcmd_sat_val**2, 0.0)), (Imax**2 - Iqcmd_sat_val**2, True), evaluate=False)',
            tex_name='I_{p,max,GFL}^2',
        )

        self.Iqmaxsq_GFL1 = VarService(
            v_str='Piecewise((0, Le(Imax**2 - Ipcmd_sat_val**2, 0.0)), (Imax**2 - Ipcmd_sat_val**2, True), evaluate=False)',
            tex_name='I_{q,max,GFL}^2',
        )

        self.Ipmax_GFL1 = Algeb(
            tex_name='I_{p,max,GFL}',
            info='Ipmax_GFL for Current Limiting Algorithm',
            v_str='PQFlag * Imax + (1 - PQFlag) * sqrt(Ipmaxsq0_GFL1)',
            e_str='PQFlag * Imax + (1 - PQFlag) * sqrt(Ipmaxsq_GFL1) - Ipmax_GFL1'
        )

        self.Iqmax_GFL1 = Algeb(
            tex_name='I_{q,max,GFL}',
            info='Iqmax_GFL for Current Limiting Algorithm',
            v_str='(1 - PQFlag) * Imax + PQFlag * sqrt(Iqmaxsq0_GFL1)',
            e_str='(1 - PQFlag) * Imax + PQFlag * sqrt(Iqmaxsq_GFL1) - Iqmax_GFL1'
        )

        self.Ipmin_GFL1 = Algeb(
            tex_name='I_{p,min,GFL}',
            info='Active-current lower limit with P/Q priority selection',
            v_str='-Ipmax_GFL1',
            e_str='-Ipmax_GFL1 - Ipmin_GFL1'
        )

        self.Iqmin_GFL1 = Algeb(
            tex_name='I_{q,min,GFL}',
            info='Reactive-current lower limit with P/Q priority selection',
            v_str='-Iqmax_GFL1',
            e_str='-Iqmax_GFL1 - Iqmin_GFL1'
        )
        
        
        self.Ipcmd_sat = Limiter(u=self.Ipcmd_GFL, lower=self.Ipmin_GFL1, upper=self.Ipmax_GFL1, name='Ipcmd_sat')
        self.Iqcmd_sat = Limiter(u=self.Iqcmd_GFL, lower=self.Iqmin_GFL1, upper=self.Iqmax_GFL1, name='Iqcmd_sat')
        
        # get the Ip_GFL and Iq_GFL
        self.Ipcmd_sat_val = Algeb(tex_name='Ipcmd_{sat,val}',
                                  info='Limited Ipcmd',
                                  v_str='Id0_GFL',
                                  e_str='Ipcmd_GFL * Ipcmd_sat_zi + Ipmax_GFL1 * Ipcmd_sat_zu + Ipmin_GFL1 * Ipcmd_sat_zl - Ipcmd_sat_val',
                                  )

        self.Iqcmd_sat_val = Algeb(tex_name='Iqcmd_{sat,val}',
                                  info='Limited Iqcmd',
                                  v_str='kqv * (Vref0 - v) + Iq0_GFL',
                                  e_str='Iqcmd_GFL * Iqcmd_sat_zi + Iqmax_GFL1 * Iqcmd_sat_zu + Iqmin_GFL1 * Iqcmd_sat_zl - Iqcmd_sat_val',
                                  )


        # --- Current Calculation (PLACEHOLDER - simplified) ---
        # GFM branch current magnitude (simplified)
        self.IVSM_mag = VarService(tex_name='I_{VSM,mag}',
                                   info='GFM branch current magnitude',
                                   v_str='sqrt(Id_VSM_lim**2 + Iq_VSM_lim**2)',
                                   )

        # GFM branch current angle (simplified)
        self.IVSM_ang = VarService(tex_name=r'\phi_{VSM}',
                                    info='GFM branch current angle',
                                    v_str='a - atan2(Iq_VSM_lim, Id_VSM_lim + 1e-8)',
                                    )
        
        
        # GFL branch current magnitude
        # ---------- dq -> xy(αβ) rotation for GFL current ----------
        self.deltaV = VarService(v_str='a', tex_name=r'\delta_V', info='dq to xy rotation angle') # relative angle?

        self.Ialpha_GFL = Algeb(
            name='Ialpha_GFL', v_str='Id0_GFL* cos(a) + ( kqv * (Vref0 - v) + Iq0_GFL )* sin(a)  ',
            e_str='Ip_GFL_lim * cos(deltaV) + Iq_GFL_lim * sin(deltaV) - Ialpha_GFL',
            tex_name='I_{\alpha,GFL}', info='GFL current alpha'
        )
        self.Ibeta_GFL = Algeb(
            name='Ibeta_GFL', v_str='Id0_GFL * sin(a) - ( kqv * (Vref0 - v) + Iq0_GFL )* cos(a) ',
            e_str='Ip_GFL_lim * sin(deltaV) - Iq_GFL_lim * cos(deltaV) - Ibeta_GFL',
            tex_name='I_{\beta,GFL}', info='GFL current beta'
        )
        self.phi_gfl = Algeb(
            name='phi_gfl', v_str='atan2(Id0_GFL * sin(a) - (kqv * (Vref0 - v) + Iq0_GFL) * cos(a), Id0_GFL * cos(a) + (kqv * (Vref0 - v) + Iq0_GFL) * sin(a))',
            e_str='atan2(Ibeta_GFL, Ialpha_GFL) - phi_gfl',                         # might have problem! 
            tex_name=r'\phi_{GFL}', info='GFL current angle in xy'
        )

        self.IGFL_mag = Algeb(tex_name='I_{GFL,mag}',
                              info='GFL branch current magnitude',
                              v_str='sqrt(Id0_GFL**2 + (kqv * (Vref0 - v) + Iq0_GFL)**2)',
                              e_str='sqrt(Ip_GFL_lim**2 + Iq_GFL_lim**2) - IGFL_mag',
                              )

        # Total current magnitude (PLACEHOLDER - vector sum needed)
        self.Itotal = Algeb(tex_name='I_{total}',
                            info='Total current magnitude (PLACEHOLDER)',
                            v_str='sqrt(Id0_GFL**2 + (kqv * (Vref0 - v) + Iq0_GFL)**2)',
                            e_str='sqrt((Id_VSM + Ipcmd_sat_val)**2 + (Iq_VSM + Iqcmd_sat_val)**2 ) - Itotal',  # Simplified, should be vector sum
                            )

        self.k_scale = VarService(
            v_str='1.0 + Indicator(Itotal > Imax) * (Itotal / Imax - 1.0)',
            tex_name='k',
            info='Scaling factor (>=1) for total current limiting'
        )


        self.k_scale_out = Algeb(tex_name='k_{scale}',
                            info='Scaling factor (>=1) for total current limiting(output)',
                            v_str='k_scale',
                            e_str='k_scale - k_scale_out',
                            )

        # --- Limited branch currents (both branches divided by k_scale) ---
        self.Ed_VSM_lim = Algeb(
            tex_name='E_{d,VSM}^{lim}',
            info='Limited GFM internal voltage d-axis component',
            v_str='v',
            e_str='v + (Ed_VSM - v) / (k_scale) - Ed_VSM_lim',
        )

        self.Eq_VSM_lim = Algeb(
            tex_name='E_{q,VSM}^{lim}',
            info='Limited GFM internal voltage q-axis component',
            v_str='0',
            e_str='Eq_VSM / (k_scale) - Eq_VSM_lim',
        )


        self.Id_VSM_lim = Algeb(
            name='Id_VSM_lim',
            v_str='0.0',
            e_str='((Ed_VSM_lim - v) * Rs + Eq_VSM_lim * Xs) / Zs2 - Id_VSM_lim',
            tex_name='I_{d,VSM}^{lim}',
            info='Limited GFM d-axis current'
        )

        self.Iq_VSM_lim = Algeb(
            name='Iq_VSM_lim',
            v_str='0.0',
            e_str='((Ed_VSM_lim - v) * Xs - Eq_VSM_lim * Rs) / Zs2 - Iq_VSM_lim',
            tex_name='I_{q,VSM}^{lim}',
            info='Limited GFM q-axis current'
        )

        self.EVSM_lim = Algeb(
            tex_name='E_{VSM}^{lim}',
            info='Limited GFM internal voltage magnitude',
            v_str='v',
            e_str='sqrt(Ed_VSM_lim**2 + Eq_VSM_lim**2 + 1e-8) - EVSM_lim',
        )

        self.dVSM_lim = Algeb(
            tex_name='\\delta_{VSM}^{lim}',
            info='Limited GFM internal voltage angle',
            v_str='a',
            e_str='a + atan2(Eq_VSM_lim, Ed_VSM_lim + 1e-8) - dVSM_lim',
        )

        self.Ip_GFL_lim = Algeb(
            name='Ip_GFL_lim', v_str='Id0_GFL',
            e_str='Ipcmd_sat_val / (k_scale) - Ip_GFL_lim',
            tex_name='I_{p,GFL}^{lim}', info='Limited GFL p-axis current'
        )
        self.Iq_GFL_lim = Algeb(
            name='Iq_GFL_lim', v_str='kqv * (Vref0 - v) + Iq0_GFL',
            e_str='Iqcmd_sat_val / (k_scale) - Iq_GFL_lim',
            tex_name='I_{q,GFL}^{lim}', info='Limited GFL q-axis current'
        )

        # total limit current
        self.Itotal_lim = Algeb(
            name='Itotal_lim', v_str='sqrt(Id0_GFL**2+(kqv * (Vref0 - v) + Iq0_GFL)**2)',
            e_str='sqrt((Id_VSM_lim + Ip_GFL_lim)**2 + (Iq_VSM_lim + Iq_GFL_lim)**2) - Itotal_lim',
            tex_name='I_{total}^{lim}', info='Total output current after limiting'
        )
        

        self.Ed_VSM = Algeb(tex_name='E_{d,VSM}',
                            info='GFM d-axis voltage',
                            v_str='v',
                            e_str='EVSM * cos(dVSM - a) - Ed_VSM',
                            )

        self.Eq_VSM = Algeb(tex_name='E_{q,VSM}',
                            info='GFM q-axis voltage',
                            v_str='0',
                            e_str='EVSM * sin(dVSM - a) - Eq_VSM',
                            )

        self.Id_VSM = Algeb(tex_name='I_{d,VSM}',
                            info='GFM d-axis current',
                            v_str='0',
                            e_str='((EVSM * cos(dVSM - a) - v) * Rs + EVSM * sin(dVSM - a) * Xs) / Zs2 - Id_VSM',
                            )
        
        # should be negative?
        self.Iq_VSM = Algeb(tex_name='I_{q,VSM}',
                            info='GFM q-axis current',
                            v_str='0',
                            e_str='-(EVSM * sin(dVSM - a) * Rs - (EVSM * cos(dVSM - a) - v) * Xs) / Zs2 - Iq_VSM',
                            )
        

        # GFM branch power at bus terminals (for bus injection)
        # In dq frame aligned with bus: Vd=v, Vq=0
        # P = Vd*Id + Vq*Iq = v*Id_VSM
        # Q = Vq*Id - Vd*Iq = v*Iq_VSM 
        
        self.Vd = VarService(v_str='v*cos(a)')
        self.Vq = VarService(v_str='v*sin(a)')
        
        self.PGFM = Algeb(tex_name='P_{GFM}',
                          info='GFM branch active power at bus',
                          v_str='0',
                          e_str='v * Id_VSM_lim - PGFM',
                          )
        
        self.QGFM = Algeb(tex_name='Q_{GFM}',
                          info='GFM branch reactive power at bus',
                          v_str='0',
                          e_str='v * Iq_VSM_lim - QGFM',
                          )
        
        # GFL branch power
    
        self.PGFL = Algeb(tex_name='P_{GFL}',
                          info='GFL branch active power',
                          v_str='v * Id0_GFL',
                          e_str='v * Ip_GFL_lim - PGFL',
                          )

        self.QGFL = Algeb(tex_name='Q_{GFL}',
                          info='GFL branch reactive power',
                          v_str='v * (kqv * (Vref0 - v) + Iq0_GFL)',
                          e_str='v * Iq_GFL_lim - QGFL',
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
