"""
REPCGFMC1 - Plant Controller for REGFMC1 (Hybrid Grid-Forming Converter).

This model implements the plant controller that provides reference signals to REGFMC1:
- GFM Branch: Voltage reference (Vref_GFM) and frequency reference (fref_GFM)
- GFL Branch: Active power command (Pcmd_GFL) and reactive power command (Qcmd_GFL)
"""

from collections import OrderedDict

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        IdxParam, Lag, Limiter, Model, ModelData, NumParam,
                        Piecewise, State, Switcher)
from andes.core.block import DeadBand1, GainLimiter, PIController, Washout
from andes.core.service import NumSelect, VarService


class REPCGFMC1Data(ModelData):
    """
    REPCGFMC1 plant controller data.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.reg = IdxParam(model='RenGen',
                            info='REGFMC1 device idx',
                            mandatory=True,
                            )

        self.busr = IdxParam(model='Bus',
                             info='Optional remote bus for measurements',
                             default=None,
                             )

        # --- Site Measurement Parameters ---
        self.Tmeas = NumParam(default=0.02,
                              tex_name='T_{meas}',
                              info='Site voltage measurement time constant',
                              unit='s',
                              )

        self.Tfrq = NumParam(default=0.02,
                             tex_name='T_{frq}',
                             info='Site frequency measurement time constant',
                             unit='s',
                             )

        # --- GFM Frequency Reference Parameters ---
        self.frmax = NumParam(default=1.02,
                              tex_name='f_{rmax}',
                              info='Maximum frequency reference',
                              unit='p.u.',
                              )

        self.frmin = NumParam(default=0.98,
                              tex_name='f_{rmin}',
                              info='Minimum frequency reference',
                              unit='p.u.',
                              )

        self.Vfth = NumParam(default=0.9,
                             tex_name='V_{fth}',
                             info='Voltage threshold for frequency reference switching',
                             unit='p.u.',
                             )

        self.Tfref = NumParam(default=0.02,
                              tex_name='T_{fref}',
                              info='Frequency reference filter time constant',
                              unit='s',
                              )

        # --- GFM Voltage Reference Parameters ---
        self.Ptarget = NumParam(default=0.0,
                                tex_name='P_{target}',
                                info='Target active power',
                                unit='p.u.',
                                )

        self.Qtarget = NumParam(default=0.0,
                                tex_name='Q_{target}',
                                info='Target reactive power',
                                unit='p.u.',
                                )

        self.Rloss = NumParam(default=0.0,
                              tex_name='R_{loss}',
                              info='Loss compensation resistance',
                              unit='p.u.',
                              )

        self.Xloss = NumParam(default=0.0,
                              tex_name='X_{loss}',
                              info='Loss compensation reactance',
                              unit='p.u.',
                              )

        self.TVmeas = NumParam(default=0.02,
                               tex_name='T_{Vmeas}',
                               info='Voltage measurement time constant for GFM',
                               unit='s',
                               )

        self.TVlag = NumParam(default=0.02,
                              tex_name='T_{Vlag}',
                              info='Voltage lag filter time constant',
                              unit='s',
                              )

        self.VrefFlag = NumParam(default=1.0,
                                 tex_name='V_{refFlag}',
                                 info='Voltage reference flag (0 or 1)',
                                 unit='bool',
                                 )

        self.Vrefmax = NumParam(default=1.1,
                                tex_name='V_{refmax}',
                                info='Maximum voltage reference',
                                unit='p.u.',
                                )

        self.Vrefmin = NumParam(default=0.9,
                                tex_name='V_{refmin}',
                                info='Minimum voltage reference',
                                unit='p.u.',
                                )

        self.TVref = NumParam(default=0.02,
                              tex_name='T_{Vref}',
                              info='Voltage reference filter time constant',
                              unit='s',
                              )

        # --- GFL Active Power Path Parameters ---
        self.dbJLI = NumParam(default=-0.01,
                              tex_name='db_{JLI}',
                              info='Frequency deadband lower limit',
                              unit='p.u.',
                              )

        self.dbJHI = NumParam(default=0.01,
                              tex_name='db_{JHI}',
                              info='Frequency deadband upper limit',
                              unit='p.u.',
                              )

        self.Ddn = NumParam(default=20.0,
                            tex_name='D_{dn}',
                            info='Droop for frequency above deadband',
                            )

        self.Dup = NumParam(default=20.0,
                            tex_name='D_{up}',
                            info='Droop for frequency below deadband',
                            )

        self.Pfreq_max = NumParam(default=0.2,
                                  tex_name='P_{freq,max}',
                                  info='Maximum frequency droop output',
                                  unit='p.u.',
                                  )

        self.Pfreq_min = NumParam(default=-0.2,
                                  tex_name='P_{freq,min}',
                                  info='Minimum frequency droop output',
                                  unit='p.u.',
                                  )

        self.Pref_max = NumParam(default=1.0,
                                 tex_name='P_{ref,max}',
                                 info='Maximum site power reference',
                                 unit='p.u.',
                                 )

        self.Pref_min = NumParam(default=0.0,
                                 tex_name='P_{ref,min}',
                                 info='Minimum site power reference',
                                 unit='p.u.',
                                 )

        self.Perr_rmax = NumParam(default=0.1,
                                  tex_name='P_{err,rmax}',
                                  info='Maximum power error for rate limiter',
                                  unit='p.u.',
                                  )

        self.Perr_rmin = NumParam(default=-0.1,
                                  tex_name='P_{err,rmin}',
                                  info='Minimum power error for rate limiter',
                                  unit='p.u.',
                                  )

        self.Perr_max = NumParam(default=0.5,
                                 tex_name='P_{err,max}',
                                 info='Maximum power error',
                                 unit='p.u.',
                                 )

        self.Perr_min = NumParam(default=-0.5,
                                 tex_name='P_{err,min}',
                                 info='Minimum power error',
                                 unit='p.u.',
                                 )

        self.Kip = NumParam(default=1.0,
                            tex_name='K_{ip}',
                            info='Proportional gain for active power PI controller',
                            )

        self.Kii = NumParam(default=0.1,
                            tex_name='K_{ii}',
                            info='Integral gain for active power PI controller',
                            )

        self.Tplag = NumParam(default=0.02,
                              tex_name='T_{plag}',
                              info='Active power command lag time constant',
                              unit='s',
                              )

        self.FFRFlag = NumParam(default=0.0,
                                tex_name='FFR_{Flag}',
                                info='FFR flag (0 or 1)',
                                unit='bool',
                                )

        self.Pcmd_GFL_max = NumParam(default=1.0,
                                     tex_name='P_{cmd,GFL,max}',
                                     info='Maximum active power command for GFL',
                                     unit='p.u.',
                                     )

        self.Pcmd_GFL_min = NumParam(default=0.0,
                                     tex_name='P_{cmd,GFL,min}',
                                     info='Minimum active power command for GFL',
                                     unit='p.u.',
                                     )

        # --- GFL Reactive Power Path Parameters ---
        self.Qref_max = NumParam(default=0.5,
                                 tex_name='Q_{ref,max}',
                                 info='Maximum reactive power reference',
                                 unit='p.u.',
                                 )

        self.Qref_min = NumParam(default=-0.5,
                                 tex_name='Q_{ref,min}',
                                 info='Minimum reactive power reference',
                                 unit='p.u.',
                                 )

        self.Kiq = NumParam(default=0.1,
                            tex_name='K_{iq}',
                            info='Reactive power gain',
                            )

        self.Tqlag = NumParam(default=0.02,
                              tex_name='T_{qlag}',
                              info='Reactive power lag time constant',
                              unit='s',
                              )

        self.Verr_max = NumParam(default=0.1,
                                 tex_name='V_{err,max}',
                                 info='Maximum voltage error',
                                 unit='p.u.',
                                 )

        self.Verr_min = NumParam(default=-0.1,
                                 tex_name='V_{err,min}',
                                 info='Minimum voltage error',
                                 unit='p.u.',
                                 )

        self.dbVLI = NumParam(default=-0.02,
                              tex_name='db_{VLI}',
                              info='Voltage deadband lower limit',
                              unit='p.u.',
                              )

        self.dbVHI = NumParam(default=0.02,
                              tex_name='db_{VHI}',
                              info='Voltage deadband upper limit',
                              unit='p.u.',
                              )

        self.Kp_vc = NumParam(default=1.0,
                              tex_name='K_{p,vc}',
                              info='Voltage control proportional gain',
                              )

        self.Tvc = NumParam(default=0.02,
                            tex_name='T_{vc}',
                            info='Voltage control time constant',
                            unit='s',
                            )

        self.Qvc_max = NumParam(default=0.5,
                                tex_name='Q_{vc,max}',
                                info='Maximum voltage control output',
                                unit='p.u.',
                                )

        self.Qvc_min = NumParam(default=-0.5,
                                tex_name='Q_{vc,min}',
                                info='Minimum voltage control output',
                                unit='p.u.',
                                )

        self.Qcmd_GFL_max = NumParam(default=0.5,
                                     tex_name='Q_{cmd,GFL,max}',
                                     info='Maximum reactive power command for GFL',
                                     unit='p.u.',
                                     )

        self.Qcmd_GFL_min = NumParam(default=-0.5,
                                     tex_name='Q_{cmd,GFL,min}',
                                     info='Minimum reactive power command for GFL',
                                     unit='p.u.',
                                     )

        self.VFlag = NumParam(default=1.0,
                              tex_name='V_{Flag}',
                              info='Voltage control flag (1-enable, 0-disable)',
                              unit='bool',
                              )

        self.Kl_xc = NumParam(default=1.0,
                              tex_name='K_{l,xc}',
                              info='Cross-coupling gain',
                              )


class REPCGFMC1Model(Model):
    """
    REPCGFMC1 plant controller model implementation.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.group = 'RenPlant'
        self.flags.tds = True

        # --- External Parameters from REGFMC1 ---
        self.bus = ExtParam(model='RenGen', src='bus', indexer=self.reg, export=False,
                            info='Retrieved bus idx', vtype=str, default=None,
                            )

        # Select bus for measurements (remote bus if provided, otherwise converter bus)
        from andes.core.service import DataSelect
        self.buss = DataSelect(self.busr, self.bus, info='Selected bus for measurements')

        # --- External Variables from Bus ---
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.buss, tex_name='V',
                          info='Bus (or busr, if given) terminal voltage',
                          )

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.buss, tex_name=r'\theta',
                          info='Bus (or busr, if given) phase angle',
                          )

        self.v0 = ExtService(model='Bus', src='v', indexer=self.buss, tex_name="V_0",
                             info='Initial bus voltage',
                             )

        # --- External Variables from REGFMC1 ---
        self.Vref_GFM = ExtAlgeb(model='RenGen', src='Vref_GFM', indexer=self.reg,
                                 tex_name='V_{ref,GFM}',
                                 info='Voltage reference for GFM branch',
                                 )

        self.fref_GFM = ExtAlgeb(model='RenGen', src='fref_GFM', indexer=self.reg,
                                 tex_name='f_{ref,GFM}',
                                 info='Frequency reference for GFM branch',
                                 )

        self.Pcmd_GFL = ExtAlgeb(model='RenGen', src='Pcmd_GFL', indexer=self.reg,
                                 tex_name='P_{cmd,GFL}',
                                 info='Active power command for GFL branch',
                                 )

        self.Qcmd_GFL = ExtAlgeb(model='RenGen', src='Qcmd_GFL', indexer=self.reg,
                                 tex_name='Q_{cmd,GFL}',
                                 info='Reactive power command for GFL branch',
                                 )

        self.Pe = ExtAlgeb(model='RenGen', src='Pe', indexer=self.reg, export=False,
                           info='Active power output of REGFMC1',
                           )

        self.Qe = ExtAlgeb(model='RenGen', src='Qe', indexer=self.reg, export=False,
                           info='Reactive power output of REGFMC1',
                           )

        self.p0 = ExtService(model='RenGen', src='p0', indexer=self.reg, tex_name='P_0',
                             info='Initial active power of REGFMC1',
                             )

        self.q0 = ExtService(model='RenGen', src='q0', indexer=self.reg, tex_name='Q_0',
                             info='Initial reactive power of REGFMC1',
                             )

        # Internal reference values from power flow
        self.Pref_site_0 = ConstService(v_str='p0',
                                        tex_name='P_{ref,site,0}',
                                        info='Initial site active power reference from power flow',
                                        )

        self.Qref_site_0 = ConstService(v_str='q0',
                                        tex_name='Q_{ref,site,0}',
                                        info='Initial site reactive power reference from power flow',
                                        )

        self.fref_site_0 = ConstService(v_str='1.0',
                                        tex_name='f_{ref,site,0}',
                                        info='Initial site frequency reference (nominal)',
                                        )

        self.Vref_site_0 = ConstService(v_str='v',
                                        tex_name='V_{ref,site,0}',
                                        info='Initial site voltage reference from power flow',
                                        )

        self.Ptarget_0 = NumSelect(self.Ptarget, self.p0,
                                    tex_name='P_{target,0}',
                                    info='Actual Ptarget (defaults to p0 if Ptarget=0)',
                                    )

        self.Qtarget_0 = NumSelect(self.Qtarget, self.q0,
                                    tex_name='Q_{target,0}',
                                    info='Actual Qtarget (defaults to q0 if Qtarget=0)',
                                    )

        # --- Internal reference Algeb variables (can be modified by external controllers) ---
        self.Pref_site = Algeb(v_str='Pref_site_0',
                               e_str='Pref_site_0 - Pref_site',
                               tex_name='P_{ref,site}',
                               info='Site active power reference (internal variable)',
                               )

        self.Qref_site = Algeb(v_str='Qref_site_0',
                               e_str='Qref_site_0 - Qref_site',
                               tex_name='Q_{ref,site}',
                               info='Site reactive power reference (internal variable)',
                               )

        self.fref_site = Algeb(v_str='fref_site_0',
                               e_str='fref_site_0 - fref_site',
                               tex_name='f_{ref,site}',
                               info='Site frequency reference (internal variable)',
                               )

        self.Vref_site = Algeb(v_str='Vref_site_0',
                               e_str='Vref_site_0 - Vref_site',
                               tex_name='V_{ref,site}',
                               info='Site voltage reference (internal variable)',
                               )

        # --- Site Measurements ---
        # Site voltage measurement
        self.Vsite = Lag(u='v', T=self.Tmeas, K=1,
                         info='Site voltage measurement',
                         tex_name='V_{site}',
                         )

        # Site frequency measurement (PLACEHOLDER - simplified to 1.0)
        self.fsite = Lag(u='1.0', T=self.Tfrq, K=1,
                         info='Site frequency measurement',
                         tex_name='f_{site}',
                         )

        # --- GFM Frequency Reference Generator (Image 3) ---
        # Frequency reference filter (using internal fref_site Algeb)
        self.frefLag = Lag(u='fref_site', T=self.Tfref, K=1,
                           info='Frequency reference filter',
                           tex_name='f_{ref}',
                           )

        # Output to REGFMC1 frequency reference
        fref_out = 'frefLag_y'
        self.fref_GFM.e_str = f'{fref_out} - 1.0'

        # --- GFM Voltage Reference Generator (Image 3) ---
        # Site voltage measurement (already defined above as Vsite)

        # Loss compensation calculation
        # Vdrop = (Rloss + jXloss) * (Ptarget - jQtarget) / Vsite_meas
        # Note: This is a simplified version; full implementation needs complex calculations
        self.Vsite_meas2 = Lag(u='v', T=self.TVmeas, K=1,
                               info='Voltage measurement for loss compensation',
                               tex_name='V_{site,meas}',
                               )

        # Loss compensation (simplified - real part only for now)
        # ΔV_loss ≈ (Rloss * Ptarget + Xloss * Qtarget) / Vsite_meas
        self.dVloss = VarService(v_str='(Rloss * Ptarget_0 + Xloss * Qtarget_0) / (Vsite_meas2_y + 1e-8)',
                                 tex_name=r'\Delta V_{loss}',
                                 info='Voltage drop due to losses',
                                 )

        # Voltage calculation with loss compensation
        self.Vcalc = Algeb(tex_name='V_{calc}',
                           info='Calculated voltage with loss compensation',
                           v_str='v0',
                           e_str='Vsite_meas2_y + dVloss - Vcalc',
                           )

        # Inverter voltage measurement (another filter)
        self.Vinv_meas = Lag(u='v', T=self.TVlag, K=1,
                             info='Inverter voltage measurement',
                             tex_name='V_{inv,meas}',
                             )

        # Voltage reference selector based on VrefFlag
        self.VrefSW = Switcher(u=self.VrefFlag, options=(0, 1), tex_name='V_{refSW}')

        # When VFlag=1, use V_GFM_ref (complex calculation); when 0, use initial voltage
        self.VGFM_ref = Algeb(tex_name='V_{GFM,ref}',
                              info='GFM voltage reference before filter',
                              v_str='v0',
                              e_str='VrefSW_s1 * Vcalc + VrefSW_s0 * v0 - VGFM_ref',
                              )

        # Apply limits
        self.VrefLim = Limiter(u=self.VGFM_ref, lower=self.Vrefmin, upper=self.Vrefmax,
                               tex_name='V_{refLim}',
                               )

        self.VGFM_ref_lim = Algeb(tex_name='V_{GFM,ref,lim}',
                                  info='Limited GFM voltage reference',
                                  v_str='v0',
                                  e_str='VGFM_ref * VrefLim_zi + Vrefmax * VrefLim_zu + Vrefmin * VrefLim_zl - VGFM_ref_lim',
                                  )

        # Voltage reference filter
        self.VrefGFMLag = Lag(u='VGFM_ref_lim', T=self.TVref, K=1,
                              info='GFM voltage reference filter',
                              tex_name='V_{ref,GFM,lag}',
                              )

        # Output to REGFMC1 voltage reference
        Vref_out = 'VrefGFMLag_y'
        self.Vref_GFM.e_str = f'{Vref_out} - v'

        # --- GFL Active Power Path (Image 5) ---
        # Frequency deadband
        self.fsite_err = Algeb(tex_name='f_{site,err}',
                               info='Site frequency error',
                               v_str='0',
                               e_str='1.0 - fsite_y - fsite_err',
                               )

        self.fdbd = DeadBand1(u=self.fsite_err, center=0.0,
                              lower=self.dbJLI, upper=self.dbJHI,
                              tex_name='f_{dbd}',
                              info='Frequency deadband',
                              )

        # Frequency droop: use Ddn when freq is low (error > 0), Dup when freq is high (error < 0)
        # Pfreq_droop = Ddn * fdbd_y (when fdbd_y > 0) or Dup * fdbd_y (when fdbd_y < 0)
        self.fdbd_sign = VarService(v_str='Indicator(fdbd_y >= 0)',
                                    tex_name='f_{dbd,sign}',
                                    )

        self.Pfreq_droop = Algeb(tex_name='P_{freq,droop}',
                                 info='Frequency droop output',
                                 v_str='0',
                                 e_str='fdbd_sign * Ddn * fdbd_y + (1 - fdbd_sign) * Dup * fdbd_y - Pfreq_droop',
                                 )

        # Apply frequency droop limits
        self.Pfreq_lim = Limiter(u=self.Pfreq_droop, lower=self.Pfreq_min, upper=self.Pfreq_max,
                                 tex_name='P_{freq,lim}',
                                 )

        self.Pfreq_droop_lim = Algeb(tex_name='P_{freq,droop,lim}',
                                     info='Limited frequency droop output',
                                     v_str='0',
                                     e_str='Pfreq_droop * Pfreq_lim_zi + Pfreq_max * Pfreq_lim_zu + Pfreq_min * Pfreq_lim_zl - Pfreq_droop_lim',
                                     )

        # Site power measurement
        self.Psite = Lag(u='Pe', T=self.Tfrq, K=1,
                         info='Site power measurement',
                         tex_name='P_{site}',
                         )

        # FFR flag selector (PLACEHOLDER - simplified)
        self.FFRCSW = Switcher(u=self.FFRFlag, options=(0, 1), tex_name='FFR_{SW}')

        # Site power reference limits
        self.Pref_site_lim = Limiter(u=self.Pref_site, lower=self.Pref_min, upper=self.Pref_max,
                                     tex_name='P_{ref,site,lim}',
                                     )

        self.Pref_site_lim_val = Algeb(tex_name='P_{ref,site,lim}',
                                       info='Limited site power reference',
                                       v_str='Pref_site_0',
                                       e_str='Pref_site * Pref_site_lim_zi + Pref_max * Pref_site_lim_zu + Pref_min * Pref_site_lim_zl - Pref_site_lim_val',
                                       )

        # Active power reference with frequency droop
        # Ptarget is a parameter, so Pref = Ptarget
        self.Ptarget_val = ConstService(v_str='Ptarget',
                                        tex_name='P_{target,val}',
                                        )

        # Power error calculation
        self.Perr = Algeb(tex_name='P_{err}',
                          info='Site power error',
                          v_str='0',
                          e_str='Pref_site_lim_val - Psite_y + FFRCSW_s0 * Pfreq_droop_lim - Perr',
                          )

        # Apply error limits (for rate limiter and PI)
        self.Perr_lim = Limiter(u=self.Perr, lower=self.Perr_min, upper=self.Perr_max,
                                tex_name='P_{err,lim}',
                                )

        self.Perr_lim_val = Algeb(tex_name='P_{err,lim}',
                                  info='Limited power error',
                                  v_str='0',
                                  e_str='Perr * Perr_lim_zi + Perr_max * Perr_lim_zu + Perr_min * Perr_lim_zl - Perr_lim_val',
                                  )

        # Integrator state for PI controller
        self.xpwr = State(tex_name='x_{pwr}',
                          info='Integrator state for active power PI',
                          v_str='Pref_site_0',
                          e_str='Kii * Perr_lim_val',
                          )

        # PI controller for active power
        self.Pcmd_pi = Algeb(tex_name='P_{cmd,pi}',
                             info='Active power PI output',
                             v_str='Pref_site_0',
                             e_str='Kip * Perr_lim_val + xpwr - Pcmd_pi',
                             )

        # Active power command with lag filter
        self.Pcmd_GFL_lag = Lag(u='Pcmd_pi + FFRCSW_s1 * Ptarget * 2',
                                T=self.Tplag, K=1,
                                info='Active power command lag filter',
                                tex_name='P_{cmd,GFL,lag}',
                                )

        # Apply Pcmd limits
        self.Pcmd_lim = Limiter(u=self.Pcmd_GFL_lag_y, lower=self.Pcmd_GFL_min, upper=self.Pcmd_GFL_max,
                                tex_name='P_{cmd,lim}',
                                )

        # Output to REGFMC1 active power command
        Pcmd_out = 'Pcmd_GFL_lag_y * Pcmd_lim_zi + Pcmd_GFL_max * Pcmd_lim_zu + Pcmd_GFL_min * Pcmd_lim_zl'
        self.Pcmd_GFL.e_str = f'{Pcmd_out} - Pcmd_GFL'

        # --- GFL Reactive Power Path (Image 4) ---
        # Voltage control path (using internal Vref_site Algeb)
        self.Verr_site = Algeb(tex_name='V_{err,site}',
                               info='Site voltage error',
                               v_str='0',
                               e_str='Vref_site - Vsite_y - Verr_site',
                               )

        # Voltage deadband
        self.Vdbd = DeadBand1(u=self.Verr_site, center=0.0,
                              lower=self.dbVLI, upper=self.dbVHI,
                              tex_name='V_{dbd}',
                              info='Voltage deadband',
                              )

        # Apply voltage error limits
        self.Verr_lim = Limiter(u=self.Vdbd_y, lower=self.Verr_min, upper=self.Verr_max,
                                tex_name='V_{err,lim}',
                                )

        self.Verr_lim_val = Algeb(tex_name='V_{err,lim}',
                                  info='Limited voltage error',
                                  v_str='0',
                                  e_str='Vdbd_y * Verr_lim_zi + Verr_max * Verr_lim_zu + Verr_min * Verr_lim_zl - Verr_lim_val',
                                  )

        # Voltage control with gain
        self.Qvc = Algeb(tex_name='Q_{vc}',
                         info='Voltage control output',
                         v_str='0',
                         e_str='Kp_vc * Verr_lim_val - Qvc',
                         )

        # Apply voltage control limits
        self.Qvc_lim = Limiter(u=self.Qvc, lower=self.Qvc_min, upper=self.Qvc_max,
                               tex_name='Q_{vc,lim}',
                               )

        self.Qvc_lim_val = Algeb(tex_name='Q_{vc,lim}',
                                 info='Limited voltage control output',
                                 v_str='0',
                                 e_str='Qvc * Qvc_lim_zi + Qvc_max * Qvc_lim_zu + Qvc_min * Qvc_lim_zl - Qvc_lim_val',
                                 )

        # Voltage control filter
        self.Qvc_lag = Lag(u='Qvc_lim_val', T=self.Tvc, K=1,
                           info='Voltage control lag filter',
                           tex_name='Q_{vc,lag}',
                           )

        # Reactive power reference path (simplified)
        self.Qref_site_lim = Limiter(u=self.Qref_site, lower=self.Qref_min, upper=self.Qref_max,
                                     tex_name='Q_{ref,site,lim}',
                                     )

        self.Qref_site_lim_val = Algeb(tex_name='Q_{ref,site,lim}',
                                       info='Limited reactive power reference',
                                       v_str='Qref_site_0',
                                       e_str='Qref_site * Qref_site_lim_zi + Qref_max * Qref_site_lim_zu + Qref_min * Qref_site_lim_zl - Qref_site_lim_val',
                                       )

        # Reactive power control with lag
        self.Qsite_filt = Lag(u='Qe', T=self.Tqlag, K=1,
                              info='Site reactive power measurement',
                              tex_name='Q_{site,filt}',
                              )

        # VFlag selector
        self.VFlagSW = Switcher(u=self.VFlag, options=(0, 1), tex_name='V_{FlagSW}')

        # When VFlag=1, use Qvc_lag (voltage control); when 0, use Q reference
        self.Qerr = Algeb(tex_name='Q_{err}',
                          info='Reactive power error',
                          v_str='0',
                          e_str='Qref_site_lim_val - Qsite_filt_y + VFlagSW_s1 * Kiq * Qvc_lag_y - Qerr',
                          )

        # Reactive power command (simplified - direct output)
        self.Qcmd_GFL_val = Algeb(tex_name='Q_{cmd,GFL,val}',
                                  info='Reactive power command value',
                                  v_str='Qtarget_0',
                                  e_str='Qtarget_0 + VFlagSW_s1 * Qvc_lag_y + VFlagSW_s0 * Qerr - Qcmd_GFL_val',
                                  )

        # Apply Qcmd limits
        self.Qcmd_lim = Limiter(u=self.Qcmd_GFL_val, lower=self.Qcmd_GFL_min, upper=self.Qcmd_GFL_max,
                                tex_name='Q_{cmd,lim}',
                                )

        # Output to REGFMC1 reactive power command
        Qcmd_out = 'Qcmd_GFL_val * Qcmd_lim_zi + Qcmd_GFL_max * Qcmd_lim_zu + Qcmd_GFL_min * Qcmd_lim_zl'
        self.Qcmd_GFL.e_str = f'{Qcmd_out} - Qcmd_GFL'


class REPCGFMC1(REPCGFMC1Data, REPCGFMC1Model):
    """
    REPCGFMC1: Plant controller for REGFMC1 (hybrid grid-forming converter).

    This model provides reference signals to REGFMC1:
    - GFM Branch: Voltage reference (Vref_GFM) and frequency reference (fref_GFM)
    - GFL Branch: Active power command (Pcmd_GFL) and reactive power command (Qcmd_GFL)

    The controller implements:
    1. GFM frequency reference generator with voltage-based switching
    2. GFM voltage reference generator with loss compensation
    3. GFL active power path with frequency droop and FFR
    4. GFL reactive power and voltage control

    Notes:
    - Site measurements are taken from the converter bus or optional remote bus
    - Frequency measurement is simplified (uses constant 1.0 pu)
    - Loss compensation uses simplified real-part calculation
    """

    def __init__(self, system, config):
        REPCGFMC1Data.__init__(self)
        REPCGFMC1Model.__init__(self, system, config)
