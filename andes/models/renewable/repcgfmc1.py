"""
REPCGFMC1 - Plant Controller for REGFMC1 (Hybrid Grid-Forming Converter).

This model implements the plant controller that provides reference signals to REGFMC1:
- GFM Branch: Voltage reference (Vref_GFM) and frequency reference (fref_GFM)
- GFL Branch: Active power command (Pcmd_GFL) and reactive power command (Qcmd_GFL)
"""
import numpy as np
from collections import OrderedDict

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtParam, ExtService,
                        IdxParam, Lag, Limiter, Model, ModelData, NumParam,
                        Piecewise, State, Switcher)
from andes.core.block import DeadBand1, GainLimiter, IntegratorAntiWindup, PIController, Washout


from andes.core.service import (CurrentSign, NumSelect, VarService,
                                EventFlag, ExtendedEvent, VarHold)

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
                             info='Plant/PCC measurement bus',
                             mandatory=True,
                             )

        self.line = IdxParam(info='Monitored branch between inverter terminal bus and plant/PCC measurement bus',
                             model='ACLine',
                             mandatory=True,
                             )
        self.Sn = NumParam(default=100.0, tex_name='S_n',
                           info='Model MVA base',
                           unit='MVA',
                           )
        self.busf = IdxParam(model='BusFreq',
                            info='Optional BusFreq device (if None, auto-created)',
                            default=None,
                            )
        self.busrocof = IdxParam(model='BusROCOF',
                             info='Optional BusROCOF device (if None, auto-created)',
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
        self.frmax = NumParam(default=1.05,
                              tex_name='f_{rmax}',
                              info='Maximum frequency reference',
                              unit='p.u.',
                              )

        self.frmin = NumParam(default=0.95,
                              tex_name='f_{rmin}',
                              info='Minimum frequency reference',
                              unit='p.u.',
                              )

        self.Vfth = NumParam(default=0,
                             tex_name='V_{fth}',
                             info='Voltage threshold for frequency reference switching',
                             unit='p.u.',
                             )

        self.Tfref = NumParam(default=0.3,   
                              tex_name='T_{fref}',
                              info='Frequency reference filter time constant',
                              unit='s',
                              )

        # --- GFM Voltage Reference Parameters ---
        self.Ptarget = NumParam(default=0.0,
                                tex_name='P_{target}',
                                info='Target active power',
                                unit='p.u.',
                                power=True,
                                )

        self.Qtarget = NumParam(default=0.0,
                                tex_name='Q_{target}',
                                info='Target reactive power',
                                unit='p.u.',
                                power=True,
                                )

        self.Rloss = NumParam(default=0.0,  
                              tex_name='R_{loss}',
                              info='Loss compensation resistance',
                              unit='p.u.',
                              z=True,
                              )

        self.Xloss = NumParam(default=0.0,   
                              tex_name='X_{loss}',
                              info='Loss compensation reactance',
                              unit='p.u.',
                              z=True,
                              )

        self.TVmeas = NumParam(default=0.01,
                               tex_name='T_{Vmeas}',
                               info='Voltage measurement time constant for GFM',
                               unit='s',
                               )

        self.TVlag = NumParam(default=0.2,
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

        self.Vrefmin = NumParam(default=0.85,
                                tex_name='V_{refmin}',
                                info='Minimum voltage reference',
                                unit='p.u.',
                                )

        self.TVref = NumParam(default=0.3,
                              tex_name='T_{Vref}',
                              info='Voltage reference filter time constant',
                              unit='s',
                              )

        # --- GFL Active Power Path Parameters ---
        self.dbJLI = NumParam(default=-0.0005,
                              tex_name='db_{JLI}',
                              info='Frequency deadband lower limit',
                              unit='p.u.',
                              )

        self.dbJHI = NumParam(default=0.0005,
                              tex_name='db_{JHI}',
                              info='Frequency deadband upper limit',
                              unit='p.u.',
                              )

        self.Ddn = NumParam(default=20.0,
                            tex_name='D_{dn}',
                            info='Droop for frequency above deadband',
                            power=True,
                            )

        self.Dup = NumParam(default=20.0,
                            tex_name='D_{up}',
                            info='Droop for frequency below deadband',
                            power=True,
                            )

        self.Pfreq_max = NumParam(default=1,
                                  tex_name='P_{freq,max}',
                                  info='Maximum frequency droop output',
                                  unit='p.u.',
                                  power=True,
                                  )

        self.Pfreq_min = NumParam(default=-1,
                                  tex_name='P_{freq,min}',
                                  info='Minimum frequency droop output',
                                  unit='p.u.',
                                  power=True,
                                  )

        self.Pref_max = NumParam(default=1.0,
                                 tex_name='P_{ref,max}',
                                 info='Maximum site power reference',
                                 unit='p.u.',
                                 power=True,
                                 )

        self.Pref_min = NumParam(default=-1.0,
                                 tex_name='P_{ref,min}',
                                 info='Minimum site power reference',
                                 unit='p.u.',
                                 power=True,
                                 )

        self.Perr_rmax = NumParam(default=0.1,
                                  tex_name='P_{err,rmax}',
                                  info='Maximum power error for rate limiter',
                                  unit='p.u.',
                                  power=True,
                                  )

        self.Perr_rmin = NumParam(default=-0.1,
                                  tex_name='P_{err,rmin}',
                                  info='Minimum power error for rate limiter',
                                  unit='p.u.',
                                  power=True,
                                  )

        self.Perr_max = NumParam(default=0.1,
                                 tex_name='P_{err,max}',
                                 info='Maximum power error',
                                 unit='p.u.',
                                 power=True,
                                 )

        self.Perr_min = NumParam(default=-0.1,
                                 tex_name='P_{err,min}',
                                 info='Minimum power error',
                                 unit='p.u.',
                                 power=True,
                                 )

        self.Kip = NumParam(default=0.1,   
                            tex_name='K_{ip}',
                            info='Proportional gain for active power PI controller',
                            )

        self.Kip_Perr = NumParam(default=0.1,
                            tex_name='K_{ip_Perr}',
                            info='Integral gain for active power PI controller',
                            )

        self.Tplag = NumParam(default=0.04,
                              tex_name='T_{plag}',
                              info='Active power command lag time constant',
                              unit='s',
                              )

        self.FFRFlag = NumParam(default=1.0,
                                tex_name='FFR_{Flag}',
                                info='FFR flag (0 or 1)',
                                unit='bool',
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

        # --- GFL Reactive Power Path Parameters ---
        self.Qref_max = NumParam(default=0.6,               
                                 tex_name='Q_{ref,max}',
                                 info='Maximum reactive power reference',
                                 unit='p.u.',
                                 power=True,
                                 )

        self.Qref_min = NumParam(default=-0.6,
                                 tex_name='Q_{ref,min}',
                                 info='Minimum reactive power reference',
                                 unit='p.u.',
                                 power=True,
                                 )

        self.Kiq = NumParam(default=0.1,
                            tex_name='K_{iq}',
                            info='Reactive power gain',
                            )


        self.Tqlag = NumParam(default=0.04,
                              tex_name='T_{qlag}',
                              info='Reactive power lag time constant',
                              unit='s',
                              )

        self.Verr_max = NumParam(default=0.3,
                                 tex_name='V_{err,max}',
                                 info='Maximum voltage error',
                                 unit='p.u.',
                                 )

        self.Verr_min = NumParam(default=-0.3,
                                 tex_name='V_{err,min}',
                                 info='Minimum voltage error',
                                 unit='p.u.',
                                 )

        self.dbVLI = NumParam(default=-0.01,
                              tex_name='db_{VLI}',
                              info='Voltage deadband lower limit',
                              unit='p.u.',
                              )

        self.dbVHI = NumParam(default=0.01,
                              tex_name='db_{VHI}',
                              info='Voltage deadband upper limit',
                              unit='p.u.',
                              )

        self.Kp_vc = NumParam(default=2,  
                              tex_name='K_{p,vc}',
                              info='Voltage control proportional gain',
                              power=True,
                              )

        self.Ki_vc = NumParam(default=6, 
                              tex_name='K_{i,vc}',
                              info='Voltage control integral gain',
                              power=True,
                              )

        self.Tvc = NumParam(default=0.02,
                            tex_name='T_{vc}',
                            info='Voltage control time constant',
                            unit='s',
                            )

        self.Qvc_max = NumParam(default=0.6,             
                                tex_name='Q_{vc,max}',
                                info='Maximum voltage control output',
                                unit='p.u.',
                                power=True,
                                )

        self.Qvc_min = NumParam(default=-0.6,
                                tex_name='Q_{vc,min}',
                                info='Minimum voltage control output',
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

        self.VFlag = NumParam(default=1.0,
                              tex_name='V_{Flag}',
                              info='Voltage control flag (1-enable, 0-disable)',
                              unit='bool',
                              )

        self.Kl_xc = NumParam(default=1.0,
                              tex_name='K_{l,xc}',
                              info='Cross-coupling gain',
                              )

        self.Qerr_max = NumParam(default=0.1,         
                                     tex_name='Q_{err,max}',
                                     info='Maximum reactive power error limit',
                                     unit='p.u.',
                                     power=True,
                                     )
        self.Qerr_min = NumParam(default=-0.1,         
                                     tex_name='Q_{err,min}',
                                     info='Minimum reactive power error limit',
                                     unit='p.u.',
                                     power=True,
                                     )
        
        
        # --- FFR Parameters (per REPCGFM_C1 specification) ---
        self.fFFR_low = NumParam(default=0.998,
                                 tex_name='f_{FFR,low}',
                                 info='Lower threshold of the FFR function',
                                 unit='p.u.',
                                 )

        self.fFFR_high = NumParam(default=1.002,
                                  tex_name='f_{FFR,high}',
                                  info='Upper threshold of the FFR function',
                                  unit='p.u.',
                                  )

        self.PFFR_low = NumParam(default=0.05,
                                 tex_name='P_{FFR,low}',
                                 info='FFR power command when frequency is below fFFR_low',
                                 unit='p.u.',
                                 power=True,
                                 )

        self.PFFR_high = NumParam(default=-0.05,
                                  tex_name='P_{FFR,high}',
                                  info='FFR power command when frequency is above fFFR_high',
                                  unit='p.u.',
                                  power=True,
                                  )

        self.DFFR = NumParam(default=0.01,
                             tex_name='D_{FFR}',
                             info='Ramp rate for FFR to quit operation',
                             unit='p.u./s',
                             power=True,
                             )

        self.TFFR = NumParam(default=20, # 300
                             tex_name='T_{FFR}',
                             info='Time duration of the FFR hold period',
                             unit='s',
                             )





class REPCGFMC1Model(Model):
    """
    REPCGFMC1 plant controller model implementation.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.group = 'RenPlant'
        
        self.flags.tds = True
        
        self.flags.update({'v_num': True, 'g_num': True})
        
        
        # --- External Parameters from REGFMC1 ---
        self.bus = ExtParam(model='RenGen', src='bus', indexer=self.reg, export=False,
                            info='Retrieved bus idx', vtype=str, default=None,
                            )

        # Plant/PCC measurement bus. Frequency is measured at this same bus.
        from andes.core.service import DeviceFinder

        self.busfreq = DeviceFinder(self.busf, link=self.busr, idx_name='bus', default_model='BusFreq')
        self.busRocof = DeviceFinder(self.busrocof, link=self.busr, idx_name='bus', default_model='BusROCOF')
        # --- External Variables from Bus ---
        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.busr, tex_name='V',
                          info='Plant/PCC measurement bus voltage',
                          e_str='0',
                          )

        self.vsite = ExtAlgeb(model='Bus', src='v', indexer=self.busr, tex_name='V_{site}',
                              info='Plant/PCC measurement bus voltage',
                              e_str='0',
                              )

        self.vinv = ExtAlgeb(model='Bus', src='v', indexer=self.bus, tex_name='V_{inv}',
                             info='Inverter terminal voltage',
                             e_str='0',
                             )

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.busr, tex_name=r'\theta',
                          info='Plant/PCC measurement bus phase angle',
                          e_str='0',
                          )

        self.v0 = ExtService(model='Bus', src='v', indexer=self.busr, tex_name="V_0",
                             info='Initial plant/PCC measurement bus voltage',
                             )

        self.f = ExtAlgeb(model='BusFreq', src='f', indexer=self.busfreq,
                          tex_name="f", info='Plant/PCC measurement bus frequency (p.u.)')

        self.rocof = ExtService(model='BusROCOF', src='Wf_y', indexer=self.busRocof, tex_name="ROCOF",
                             info='bus ROCOF',
                             )

        # --- Monitored branch for plant/PCC power measurement ---
        self.bus1 = ExtParam(model='ACLine', src='bus1', indexer=self.line, export=False,
                             info='Retrieved monitored branch Line.bus1 idx', vtype=str,
                             )

        self.bus2 = ExtParam(model='ACLine', src='bus2', indexer=self.line, export=False,
                             info='Retrieved monitored branch Line.bus2 idx', vtype=str,
                             )

        self.line_phi = ExtParam(model='ACLine', src='phi', indexer=self.line, export=False,
                                 info='Retrieved monitored branch phase shift', vtype=float,
                                 )

        self.v1 = ExtAlgeb(model='ACLine', src='v1', indexer=self.line, tex_name='V_1',
                           info='Voltage at monitored branch Line.bus1',
                           )

        self.v2 = ExtAlgeb(model='ACLine', src='v2', indexer=self.line, tex_name='V_2',
                           info='Voltage at monitored branch Line.bus2',
                           )

        self.a1 = ExtAlgeb(model='ACLine', src='a1', indexer=self.line, tex_name=r'\theta_1',
                           info='Angle at monitored branch Line.bus1',
                           )

        self.a2 = ExtAlgeb(model='ACLine', src='a2', indexer=self.line, tex_name=r'\theta_2',
                           info='Angle at monitored branch Line.bus2',
                           )

        self.gh = ExtService(model='ACLine', src='gh', indexer=self.line,
                             info='Retrieved monitored branch Line.gh',
                             )

        self.bh = ExtService(model='ACLine', src='bh', indexer=self.line,
                             info='Retrieved monitored branch Line.bh',
                             )

        self.gk = ExtService(model='ACLine', src='gk', indexer=self.line,
                             info='Retrieved monitored branch Line.gk',
                             )

        self.bk = ExtService(model='ACLine', src='bk', indexer=self.line,
                             info='Retrieved monitored branch Line.bk',
                             )

        self.ghk = ExtService(model='ACLine', src='ghk', indexer=self.line,
                              info='Retrieved monitored branch Line.ghk',
                              )

        self.bhk = ExtService(model='ACLine', src='bhk', indexer=self.line,
                              info='Retrieved monitored branch Line.bhk',
                              )

        self.itap = ExtService(model='ACLine', src='itap', indexer=self.line,
                               info='Retrieved monitored branch Line.itap',
                               )

        self.itap2 = ExtService(model='ACLine', src='itap2', indexer=self.line,
                                info='Retrieved monitored branch Line.itap2',
                                )

        self.MeasBusSign = CurrentSign(self.busr, self.bus1, self.bus2,
                                       tex_name='I_{site,sign}',
                                       info='Sign of monitored branch current outflow at the plant/PCC measurement bus',
                                       )

        Pij = ('v1 ** 2 * (gh + ghk) * itap2 - '
               'v1 * v2 * (ghk * cos(a1 - a2 - line_phi) + '
               'bhk * sin(a1 - a2 - line_phi)) * itap')
        Qij = ('-v1 ** 2 * (bh + bhk) * itap2 - '
               'v1 * v2 * (ghk * sin(a1 - a2 - line_phi) - '
               'bhk * cos(a1 - a2 - line_phi)) * itap')
        Pji = ('v2 ** 2 * (gk + ghk) - '
               'v1 * v2 * (ghk * cos(a1 - a2 - line_phi) - '
               'bhk * sin(a1 - a2 - line_phi)) * itap')
        Qji = ('-v2 ** 2 * (bk + bhk) + '
               'v1 * v2 * (ghk * sin(a1 - a2 - line_phi) + '
               'bhk * cos(a1 - a2 - line_phi)) * itap')

        self.Pline_bus1 = Algeb(tex_name='P_{ij}',
                                info='Monitored branch active power out of Line.bus1',
                                v_str=Pij,
                                e_str=f'{Pij} - Pline_bus1',
                                )

        self.Qline_bus1 = Algeb(tex_name='Q_{ij}',
                                info='Monitored branch reactive power out of Line.bus1',
                                v_str=Qij,
                                e_str=f'{Qij} - Qline_bus1',
                                )

        self.Pline_bus2 = Algeb(tex_name='P_{ji}',
                                info='Monitored branch active power out of Line.bus2',
                                v_str=Pji,
                                e_str=f'{Pji} - Pline_bus2',
                                )

        self.Qline_bus2 = Algeb(tex_name='Q_{ji}',
                                info='Monitored branch reactive power out of Line.bus2',
                                v_str=Qji,
                                e_str=f'{Qji} - Qline_bus2',
                                )

        Psite_raw = ('-(0.5 * (1 + MeasBusSign) * Pline_bus1 + '
                     '0.5 * (1 - MeasBusSign) * Pline_bus2)')
        Qsite_raw = ('-(0.5 * (1 + MeasBusSign) * Qline_bus1 + '
                     '0.5 * (1 - MeasBusSign) * Qline_bus2)')

        self.Psite_raw = Algeb(tex_name='P_{site,raw}',
                               info='Plant/PCC active power injection from monitored branch; positive means plant output',
                               v_str=Psite_raw,
                               e_str=f'{Psite_raw} - Psite_raw',
                               )

        self.Qsite_raw = Algeb(tex_name='Q_{site,raw}',
                               info='Plant/PCC reactive power injection from monitored branch; positive means plant output',
                               v_str=Qsite_raw,
                               e_str=f'{Qsite_raw} - Qsite_raw',
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
        self.Vref0= ExtService(model='RenGen', src='Vref0', indexer=self.reg, tex_name='V_{ref0}',  
                           info='Vref0 of REGFMC1',
                           )
        self.p0 = ExtService(model='RenGen', src='p0', indexer=self.reg, tex_name='P_0',
                             info='Initial active power of REGFMC1',
                             )

        self.q0 = ExtService(model='RenGen', src='q0', indexer=self.reg, tex_name='Q_0',
                             info='Initial reactive power of REGFMC1',
                             )

        self.Ploss_0 = ConstService(v_str='(p0**2 + q0**2) * Rloss',
                                    tex_name='P_{loss,0}',
                                    info='Initial active power loss compensation',
                                    )

        self.Qloss_0 = ConstService(v_str='(p0**2 + q0**2) * Xloss',
                                    tex_name='Q_{loss,0}',
                                    info='Initial reactive power loss compensation',
                                    )

        # Internal reference values from power flow, expressed at the PCC.
        self.Pref_site_0 = ConstService(v_str='p0 - Ploss_0',
                                        tex_name='P_{ref,site,0}',
                                        info='Initial site active power reference from power flow',
                                        )

        self.Qref_site_0 = ConstService(v_str='q0 - Qloss_0',
                                        tex_name='Q_{ref,site,0}',
                                        info='Initial site reactive power reference from power flow',
                                        )

        self.fref_site_0 = ConstService(v_str='1.0',
                                        tex_name='f_{ref,site,0}',
                                        info='Initial site frequency reference (nominal)',
                                        )

        self.Ki_vc_nonzero = ConstService(v_str='Indicator(Ki_vc > 0) + Indicator(Ki_vc < 0)',
                                          tex_name='z_{Ki,vc}',
                                          info='Voltage-control integral gain is nonzero',
                                          )

        self.Qref_site_pos = ConstService(v_str='Indicator(Qref_site_0 > 0)',
                                          tex_name='z_{Qref>0}',
                                          info='Initial site reactive power is positive',
                                          )

        self.Qref_site_neg = ConstService(v_str='Indicator(Qref_site_0 < 0)',
                                          tex_name='z_{Qref<0}',
                                          info='Initial site reactive power is negative',
                                          )

        self.Vref_site_0 = ConstService(v_str='Ki_vc_nonzero * v + (1 - Ki_vc_nonzero) * '
                                              '((1 - Qref_site_pos - Qref_site_neg) * v + '
                                              'Qref_site_pos * (v + dbVHI + Qref_site_0 / Kp_vc) + '
                                              'Qref_site_neg * (v + dbVLI + Qref_site_0 / Kp_vc))',
                                        tex_name='V_{ref,site,0}',
                                        info='Initial site voltage reference from power flow',
                                        )

        self.Vest_0 = ConstService(v_str='sqrt((v0 + Rloss * Pref_site_0 + Xloss * Qref_site_0)**2 + '
                                         '(Pref_site_0 * Xloss - Qref_site_0 * Rloss)**2)',
                                   tex_name='V_{est,0}',
                                   info='Initial estimated inverter terminal voltage from site quantities',
                                   )

        self.dVerr = ConstService(v_str='Vref0 - Vest_0',
                                  tex_name=r'\Delta V_{err}',
                                  info='Initialization offset for GFM voltage reference',
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

        # Site frequency measurement 
        self.fsite = Lag(u='f', T=self.Tfrq, K=1,
                         info='Site frequency measurement',
                         tex_name='f_{site}',              
                         )

        # --- GFM Frequency Reference Generator ---
        # Frequency reference filter (using internal fref_site Algeb)
    
        self.Vsite_gate = Limiter(
            self.Vsite_y,
            lower=self.Vfth,
            upper=1.5,
            name='Vsite_gate'
        )

        self.fsite_meas = Algeb(
            name='fsite_meas',
            tex_name='f_{site,meas}',
            info='Site frequency after voltage-threshold selection',
            v_str='1.0',
            e_str='(fsite_y) * (Vsite_gate_zi + Vsite_gate_zu) + (fref_site) * (Vsite_gate_zl) - fsite_meas'
        )
        self.frefLag = Lag(u='fsite_meas', T=self.Tfref, K=1,
                           info='Frequency reference filter',
                           tex_name='f_{ref}',
                           )
        # Output to REGFMC1 frequency reference
        fref_out = 'frefLag_y'
        self.fref_GFM.e_str = f'{fref_out} -1'

        # --- GFM Voltage Reference Generator ---
        # Site voltage measurement (already defined above as Vsite)

        # Loss compensation calculation
        # Vdrop = (Rloss + jXloss) * (Ptarget - jQtarget) / Vsite_meas
        # Note: This is a simplified version; full implementation needs complex calculations
        self.Vsite_meas2 = Lag(u='v', T=self.TVmeas, K=1,
                               info='Site voltage measurement for loss compensation',
                               tex_name='V_{site,meas}',
                               )

        self.Vest = Algeb(tex_name='V_{est}',
                          info='Estimated inverter terminal voltage from site voltage and loss compensation',
                          v_str='Vest_0',
                          e_str='sqrt((Vsite_meas2_y + Rloss * Ptarget_1 + Xloss * Qtarget_1)**2 + '
                                '(Ptarget_1 * Xloss - Qtarget_1 * Rloss)**2) - Vest',
                          )

        # Voltage calculation with loss compensation
        self.Vcalc = Algeb(tex_name='V_{calc}',
                           info='Calculated voltage with loss compensation',
                           v_str='Vref0',
                           e_str='Vest + dVerr - Vcalc',
                           )

        # Inverter voltage measurement (another filter)
        self.Vinv_meas0 = Lag(u='vinv', T=self.TVmeas, K=1,
                              info='Inverter terminal voltage measurement',
                              tex_name='V_{inv,meas0}',
                              )

        self.Vinv_meas = Lag(u='Vinv_meas0_y', T=self.TVlag, K=1,
                             info='Delayed inverter terminal voltage measurement',
                             tex_name='V_{inv,meas}',
                             )

        # Voltage reference selector based on VrefFlag
        self.VrefSW = Switcher(u=self.VrefFlag, options=(0, 1), tex_name='V_{refSW}')

        # When VFlag=1, use V_GFM_ref (complex calculation); when 0, use initial voltage
        self.VGFM_ref = Algeb(tex_name='V_{GFM,ref}',
                              info='GFM voltage reference before filter',
                              v_str='Vref0',
                              e_str='VrefSW_s1 * Vcalc + VrefSW_s0 * Vinv_meas_y - VGFM_ref',  
                              )

        # Apply limits
        self.VrefLim = Limiter(self.VGFM_ref, lower=self.Vrefmin, upper=self.Vrefmax,
                               tex_name='V_{refLim}',
                               )

        self.VGFM_ref_lim = Algeb(tex_name='V_{GFM,ref,lim}',
                                  info='Limited GFM voltage reference',
                                  v_str='Vref0',
                                  e_str='VGFM_ref * VrefLim_zi + Vrefmax * VrefLim_zu + Vrefmin * VrefLim_zl - VGFM_ref_lim',
                                  )

        # Voltage reference filter
        self.VrefGFMLag = Lag(u='VGFM_ref_lim', T=self.TVref, K=1,
                              info='GFM voltage reference filter',
                              tex_name='V_{ref,GFM,lag}',
                              )

        # Output to REGFMC1 voltage reference
        Vref_out = 'VrefGFMLag_y'
        self.Vref_GFM.e_str = f'{Vref_out} - (Vref0)'

        # --- GFL Active Power Path ---
        # Frequency deadband
        self.fsite_err = Algeb(tex_name='f_{site,err}',
                               info='Site frequency error',
                               v_str='0',
                               e_str='fref_site - fsite_meas - fsite_err',
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
        self.Pfreq_lim = Limiter(self.Pfreq_droop, lower=self.Pfreq_min, upper=self.Pfreq_max,
                                 tex_name='P_{freq,lim}',
                                 )

        self.Pfreq_droop_lim = Algeb(tex_name='P_{freq,droop,lim}',
                                     info='Limited frequency droop output',
                                     v_str='0',
                                     e_str='Pfreq_droop * Pfreq_lim_zi + Pfreq_max * Pfreq_lim_zu + Pfreq_min * Pfreq_lim_zl - Pfreq_droop_lim',
                                     )
        
        # FFR module
        self.f_rocof = Algeb(v_str='rocof',
                               e_str='rocof - f_rocof',
                               tex_name='f_{rocof,freq}',
                               info='ROCOF',
                               )
        
        
        self.FFRCSW = Switcher(u=self.FFRFlag, options=(0, 1), tex_name='FFR_{SW}')


        # algebraic output placeholder, final value still updated in g_numeric()
        self.P_FFR = Algeb(
            tex_name='P_{FFR}',
            info='FFR contribution added to active power reference',
            v_str='0.0',
            e_str='0.0 - P_FFR',
            diag_eps=True
        )

        self.Paux = Algeb(tex_name='P_{aux}',
                          info='Auxiliary active power reference',
                          v_str='0.0',
                          e_str='0.0 - Paux',
                          diag_eps=True,
                          )

        # P target
        self.Ptarget_1_initial = Algeb(tex_name='P_{target1_initial}',  # modified
                                       info='active power P target_initial',
                                       v_str='Pref_site_0',
                                       e_str='Pfreq_droop_lim + Pref_site + FFRCSW_s1 * P_FFR + Paux - Ptarget_1_initial',
                                       )
        self.Ptarget_1_lim = Limiter(self.Ptarget_1_initial, lower=self.Pref_min, upper=self.Pref_max,  # modified
                                     tex_name='P_{target1_limit}',
                                     )
        self.Ptarget_1 = Algeb(tex_name='P_{target1}',  # modified
                               info='active power P target',
                               v_str='Pref_site_0',
                               e_str='Ptarget_1_initial * Ptarget_1_lim_zi + Pref_max * Ptarget_1_lim_zu + Pref_min * Ptarget_1_lim_zl - Ptarget_1',
                               )
        # Site power measurement from the monitored branch at the PCC side.
        self.Psite = Lag(u='Psite_raw', T=self.Tfrq, K=1,
                         info='Site active power measurement from monitored branch',
                         tex_name='P_{site}',
                         )

        # Site power reference limits
        
        # Active power reference with frequency droop
        # Ptarget is a parameter, so Pref = Ptarget
        # Power error calculation
        self.Perr = Algeb(tex_name='P_{err}',
                          info='Site power error',
                          v_str='Ptarget_1 - Psite_y',
                          e_str='Ptarget_1 - Psite_y  - Perr',  # mistake
                          )

        # Apply error limits (for rate limiter and PI)
        self.Perr_lim = Limiter(self.Perr, lower=self.Perr_min, upper=self.Perr_max,
                                tex_name='P_{err,lim}',
                                )

        self.Perr_lim_val = Algeb(tex_name='P_{err,lim}',
                                  info='Limited power error',
                                  v_str='Perr * Perr_lim_zi + Perr_max * Perr_lim_zu + Perr_min * Perr_lim_zl',
                                  e_str='Perr * Perr_lim_zi + Perr_max * Perr_lim_zu + Perr_min * Perr_lim_zl - Perr_lim_val',
                                  )

        # Integrator state for PI controller       
        self.xpwr = State(tex_name='x_{pwr}',
                          info='Integrator state for active power PI',
                          v_str='p0 - (Pref_site_0**2 + Qref_site_0**2) * Rloss - Pref_site_0',
                          e_str='Kip_Perr * Perr_lim_val',
                          )


        # Ploss   
        self.Ploss = Algeb(tex_name='P_{loss,GFL}',  
                                info='active power loss value',
                                v_str='(Pref_site_0**2 + Qref_site_0**2) * Rloss',
                                e_str='(Ptarget_1**2+ Qtarget_1**2)*Rloss - Ploss',
                           )


        # Active power command with lag filter

        self.Pcmd_sum = Algeb(tex_name='P_{cmd,sum}',    
                              info='Sum for active power command before lag',
                              v_str='p0',
                              e_str='Ploss + xpwr + Ptarget_1 - Pcmd_sum')

        self.Pcmd_GFL_lag = Lag(u='Pcmd_sum',
                                T=self.Tplag, K=1,
                                info='Active power command lag filter',
                                tex_name='P_{cmd,GFL,lag}')

        # Apply Pcmd limits
        self.Pcmd_lim = Limiter(self.Pcmd_GFL_lag_y, lower=self.Pcmd_GFL_min, upper=self.Pcmd_GFL_max,
                                tex_name='P_{cmd,lim}',
                                )

        # Output to REGFMC1 active power command
        Pcmd_out = 'Pcmd_GFL_lag_y * Pcmd_lim_zi +Pcmd_GFL_max * Pcmd_lim_zu + Pcmd_GFL_min * Pcmd_lim_zl'
        self.Pcmd_GFL.e_str = f'{Pcmd_out} - (p0)'

        # --- GFL Reactive Power Path  ---
        # Voltage control path 
        self.Verr_site = Algeb(tex_name='V_{err,site}',
                               info='Site voltage error',
                               v_str='Vref_site_0 - v',
                               e_str='Vref_site - Vsite_y - Verr_site',
                               )

        # Voltage deadband
        self.Vdbd = DeadBand1(u=self.Verr_site, center=0.0,
                              lower=self.dbVLI, upper=self.dbVHI,
                              tex_name='V_{dbd}',
                              info='Voltage deadband',
                              )

        # Apply voltage error limits
        self.Verr_lim = Limiter(self.Vdbd_y, lower=self.Verr_min, upper=self.Verr_max,
                                tex_name='V_{err,lim}',
                                )

        self.Verr_lim_val = Algeb(tex_name='V_{err,lim}',
                                  info='Limited voltage error',
                                  v_str='Vdbd_y * Verr_lim_zi + Verr_max * Verr_lim_zu + Verr_min * Verr_lim_zl',
                                  e_str='Vdbd_y * Verr_lim_zi + Verr_max * Verr_lim_zu + Verr_min * Verr_lim_zl - Verr_lim_val',
                                  )

        # Integral path with anti-windup
        self.Qvc_int = IntegratorAntiWindup(u=self.Verr_lim_val,
                                            T=1.0,
                                            K=self.Ki_vc,
                                            y0='Ki_vc_nonzero * Qref_site_0',
                                            lower=self.Qvc_min,
                                            upper=self.Qvc_max,
                                            name='Qvc_int',
                                            tex_name='Q_{vc,int}',
                                            info='Voltage control integrator with anti-windup',
                                            )

        # Voltage control PI output 
        self.Qvc = Algeb(tex_name='Q_{vc}',
                         info='Voltage control PI output',
                         v_str='Qref_site_0',
                         e_str='Kp_vc * Verr_lim_val + Qvc_int_y - Qvc',
                         )

        # Apply voltage control limits
        self.Qvc_lim = Limiter(self.Qvc, lower=self.Qvc_min, upper=self.Qvc_max,
                               tex_name='Q_{vc,lim}',
                               )

        self.Qvc_lim_val = Algeb(tex_name='Q_{vc,lim}',
                                 info='Limited voltage control output',
                                 v_str='Qvc * Qvc_lim_zi + Qvc_max * Qvc_lim_zu + Qvc_min * Qvc_lim_zl',
                                 e_str='Qvc * Qvc_lim_zi + Qvc_max * Qvc_lim_zu + Qvc_min * Qvc_lim_zl - Qvc_lim_val',
                                 )

        # Voltage control filter
        self.Qvc_lag = Lag(u='Qvc_lim_val', T=self.Tvc, K=1,
                           info='Voltage control lag filter',
                           tex_name='Q_{vc,lag}',
                           )

        # Reactive power control with lag
        self.Qsite = Lag(u='Qsite_raw', T=self.Tqlag, K=1,
                         info='Site reactive power measurement from monitored branch',
                         tex_name='Q_{site}',
                         )

        # VFlag selector
        self.VFlagSW = Switcher(u=self.VFlag, options=(0, 1), tex_name='V_{FlagSW}')

        self.Qaux = Algeb(tex_name='Q_{aux}',
                          info='Auxiliary reactive power reference',
                          v_str='0.0',
                          e_str='0.0 - Qaux',
                          diag_eps=True,
                          )


        self.Qtarget_1_initial = Algeb(tex_name='Q_{target1_initial}',          
                               info='Reactive power Q target_initial',
                               v_str='Qref_site_0 ',
                               e_str='VFlagSW_s1 * Qvc_lag_y + VFlagSW_s0 * Qref_site + Qaux - Qtarget_1_initial ',
                               )
        self.Qtarget_1_lim = Limiter(self.Qtarget_1_initial, lower=self.Qref_min, upper=self.Qref_max, 
                                tex_name='Q_{target1_limit}',
                                )
        self.Qtarget_1= Algeb(tex_name='Q_{target1}', 
                                       info='Reactive power Q target',
                                       v_str='Qref_site_0',
                                       e_str='Qtarget_1_lim_zi * Qtarget_1_initial + Qref_max * Qtarget_1_lim_zu + Qref_min * Qtarget_1_lim_zl - Qtarget_1',
                                       )
        self.Qerr0 = Algeb(tex_name='Q_{err0}',
                          info='Reactive power error_0',
                          v_str='Qtarget_1 - Qsite_y',
                          e_str='Qtarget_1 - Qsite_y  - Qerr0',
                          )

        # Integral path with anti-windup
        self.Qerr_int = IntegratorAntiWindup(u=self.Qerr0,
                                            T=1.0,
                                            K=self.Kiq,
                                            y0='q0 - (Pref_site_0**2 + Qref_site_0**2) * Xloss - Qref_site_0',
                                            lower=self.Qerr_min,
                                            upper=self.Qerr_max,
                                            name='Qerr_int',
                                            tex_name='Q_{err_integral}',
                                            info='Integrator state for Qerror',
                                            )

        # Voltage control PI output 
        self.Qerr_pi = Algeb(tex_name='Q_{vc}',
                         info='Qerror PI output',
                         v_str='q0 - (Pref_site_0**2 + Qref_site_0**2) * Xloss - Qref_site_0',
                         e_str='Qerr_int_y - Qerr_pi',
                         )

        self.Qerr_lim = Limiter(self.Qerr_pi, lower=self.Qerr_min, upper=self.Qerr_max,  
                               tex_name='Q_{err,lim}',
                               )

        self.Qerr_lim_val = Algeb(tex_name='Q_{vc,lim}',     
                                 info='Limited voltage control output',
                                 v_str='q0 - (Pref_site_0**2 + Qref_site_0**2) * Xloss - Qref_site_0',
                                 e_str='Qerr_pi * Qerr_lim_zi + Qerr_max * Qerr_lim_zu + Qerr_min * Qerr_lim_zl - Qerr_lim_val',
                                 )
        self.Qloss = Algeb(tex_name='Q_{loss,GFL}',  
                                info='Reactive power loss value',
                                v_str='(Pref_site_0**2 + Qref_site_0**2) * Xloss',
                                e_str='(Ptarget_1**2+ Qtarget_1**2)*Xloss - Qloss',
                                )
        self.Qcmd_GFL_0 = Algeb(tex_name='Q_{cmd,GFL,0}',              
                          info='Reactive power command value(no lag)',
                          v_str='q0',
                          e_str='Qloss + Qerr_lim_val + Qtarget_1 - Qcmd_GFL_0',
                          )


        self.Qcmd_GFLLag = Lag(u='Qcmd_GFL_0', T=self.Tqlag , K=1,  
                           info='Reactive power command value',
                           tex_name='Q_{cmd,GFL}',
                           )

        self.Qcmd_lim = Limiter(self.Qcmd_GFLLag_y, lower=self.Qcmd_GFL_min, upper=self.Qcmd_GFL_max,
                                tex_name='Q_{cmd,lim}',
                                )

        Qcmd_out = 'Qcmd_GFLLag_y * Qcmd_lim_zi + Qcmd_GFL_max * Qcmd_lim_zu + Qcmd_GFL_min * Qcmd_lim_zl'
        
      # Output to REGFMC1 reactive power command
        self.Qcmd_GFL.e_str = f'{Qcmd_out} - (q0)'
      
      
    # # FFR update function 
    def v_numeric(self, **kwargs):
        self._ffr_cmd = 0.0          # ffr output now
        self._ffr_hold_remain = 0.0  # remaining time
        self._ffr_busy = 0           # 0=armed, 1=busy
        self._ffr_last_t = None
        self._ffr_eps = 1e-8

    def g_numeric(self, **kwargs):
        dae = self.system.dae
        t_now = float(getattr(dae, 't', 0.0))

        if not hasattr(self, '_ffr_cmd'):
            self._ffr_cmd = 0.0
            self._ffr_hold_remain = 0.0
            self._ffr_busy = 0
            self._ffr_last_t = None
            self._ffr_eps = 1e-8

        if self._ffr_last_t is None:
            dt = 0.0
        else:
            dt = max(0.0, t_now - self._ffr_last_t)

        enabled = bool(self.FFRFlag.v[0] > 0.5)
        f_meas = float(self.fsite_meas.v[0])

        f_low = float(self.fFFR_low.v[0])
        f_high = float(self.fFFR_high.v[0])

        p_low = float(self.PFFR_low.v[0])
        p_high = float(self.PFFR_high.v[0])

        dffr = float(self.DFFR.v[0])
        tffr = float(self.TFFR.v[0])

        if (self._ffr_last_t is None) or (dt > 0.0):

            if not enabled:
                self._ffr_cmd = 0.0
                self._ffr_hold_remain = 0.0
                self._ffr_busy = 0

            else:
                # 1) armed: check trigger only when not busy
                if self._ffr_busy == 0:
                    if f_meas < f_low:
                        self._ffr_cmd = p_low
                        self._ffr_hold_remain = tffr
                        self._ffr_busy = 1
                    elif f_meas > f_high:
                        self._ffr_cmd = p_high
                        self._ffr_hold_remain = tffr
                        self._ffr_busy = 1

                # 2) busy: hold stage
                else:
                    if self._ffr_hold_remain > 0.0:
                        self._ffr_hold_remain = max(0.0, self._ffr_hold_remain - dt)

                    # 3) ramp-back stage
                    else:
                        if self._ffr_cmd > 0.0:
                            self._ffr_cmd = max(0.0, self._ffr_cmd - dffr * dt)
                        elif self._ffr_cmd < 0.0:
                            self._ffr_cmd = min(0.0, self._ffr_cmd + dffr * dt)

                        # 4) re-arm only after returning to zero
                        if abs(self._ffr_cmd) <= self._ffr_eps:
                            self._ffr_cmd = 0.0
                            self._ffr_busy = 0

            self._ffr_last_t = t_now

        # algebraic residual: force P_FFR = _ffr_cmd
        self.P_FFR.e[:] = np.array([self._ffr_cmd]) - self.P_FFR.v   


class REPCGFMC1(REPCGFMC1Data, REPCGFMC1Model):
    """
    REPCGFMC1: Plant controller for REGFMC1 (hybrid GFL/GFM converter).

    This model provides reference signals to REGFMC1:
    - GFM Branch: Voltage reference (Vref_GFM) and frequency reference (fref_GFM)
    - GFL Branch: Active power command (Pcmd_GFL) and reactive power command (Qcmd_GFL)

    The controller implements:
    1. GFM frequency reference generator with voltage-based switching
    2. GFM voltage reference generator with loss compensation
    3. GFL active power path with frequency droop and FFR
    4. GFL reactive power and voltage control

    Notes:
    - Voltage and frequency measurements are taken from ``busr``.
    - Site P/Q measurements are taken from the monitored branch at ``busr``.
    - ``Rloss`` and ``Xloss`` are user-specified loss-compensation values.
    """

    def __init__(self, system, config):
        REPCGFMC1Data.__init__(self)
        REPCGFMC1Model.__init__(self, system, config)
