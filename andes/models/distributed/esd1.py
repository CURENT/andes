"""Distributed energy storage system model"""

from andes.core.block import Integrator
from andes.core.discrete import HardLimiter, LessThan
from andes.core.param import NumParam
from andes.core.var import Algeb, AliasState
from andes.models.distributed.pvd1 import PVD1Data, PVD1Model


class ESD1Data(PVD1Data):
    """
    Data for energy storage distributed model.
    """

    def __init__(self):
        PVD1Data.__init__(self)
        self.Tf = NumParam(default=1.0, tex_name='T_f',
                           info='Integrator constant for SOC model',
                           )
        self.SOCmin = NumParam(default=0.0, tex_name='SOC_{min}',
                               info='Minimum required value for SOC in limiter',
                               )

        self.SOCmax = NumParam(default=1.0, tex_name='SOC_{max}',
                               info='Maximum allowed value for SOC in limiter',
                               )

        self.SOCinit = NumParam(default=0.5, tex_name='SOC_{init}',
                                info='Initial state of charge',
                                )

        self.En = NumParam(default=100.0, tex_name='E_n',
                           info='Rated energy capacity',
                           unit="MWh"
                           )

        self.EtaC = NumParam(default=1.0, tex_name='Eta_C',
                             info='Efficiency during charging',
                             vrange=(0, 1),
                             )

        self.EtaD = NumParam(default=1.0, tex_name='Eta_D',
                             info='Efficiency during discharging',
                             vrange=(0, 1),
                             )


class ESD1Model(PVD1Model):
    """
    Model implementation of ESD1.
    """

    def __init__(self, system, config):
        PVD1Model.__init__(self, system, config)

        # --- Determine whether the energy storage is in charging or discharging mode ---
        self.LTN = LessThan(self.Ipout_y, 0.0)

        # --- Add integrator. Assume that state-of-charge is the initial condition ---
        self.pIG = Integrator(u='-LTN_z1*(v * Ipout_y)*EtaC - LTN_z0*(v * Ipout_y)/EtaD',
                              T=self.Tf, K='sys_mva / 3600 / En', y0=self.SOCinit,
                              check_init=False,
                              )
        self.pIG.info = 'State of charge'

        self.SOC = AliasState(self.pIG_y)
        self.SOC.info = 'Alias for state of charge'

        # --- Add hard limiter for SOC ---
        self.SOClim = HardLimiter(u=self.pIG_y, lower=self.SOCmin, upper=self.SOCmax)

        # --- Add Ipmax, Ipmin, and Ipcmd ---
        self.Ipmax.v_str = '(1-SOClim_zl)*(SWPQ_s1 * ialim + SWPQ_s0 * sqrt(Ipmaxsq0))'
        self.Ipmax.e_str = '(1-SOClim_zl)*(SWPQ_s1 * ialim + SWPQ_s0 * sqrt(Ipmaxsq)) - Ipmax'

        self.Ipmin = Algeb(info='Minimum value of Ip',
                           v_str='-(1-SOClim_zu) * (SWPQ_s1 * ialim + SWPQ_s0 * sqrt(Ipmaxsq0))',
                           e_str='-(1-SOClim_zu) * (SWPQ_s1 * ialim + SWPQ_s0 * sqrt(Ipmaxsq)) - Ipmin',
                           )

        self.Ipcmd.lim.lower = self.Ipmin
        self.Ipcmd.y.deps = ['Ipmin']


class ESD1(ESD1Data, ESD1Model):
    """
    Distributed energy storage model.

    A state-of-charge limit is added to the PVD1 model.
    This limit is applied to Ipmax and Ipmin.
    The state of charge is in state variable ``SOC``,
    which is an alias of ``pIG_y``.

    Reference:
    [1] Powerworld, Renewable Energy Electrical Control Model REEC_C
    Available:

    https://www.powerworld.com/WebHelp/Content/TransientModels_HTML/Exciter%20REEC_C.htm
    """
    def __init__(self, system, config):
        ESD1Data.__init__(self)
        ESD1Model.__init__(self, system, config)
