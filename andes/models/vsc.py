"""Voltage-source converter models"""

from andes.models.dcbase import ACDC2Term
from andes.core.param import NumParam
from andes.core.var import Algeb, State, ExtState, ExtAlgeb  # NOQA
from andes.core.service import ConstService, ExtService  # NOQA
from andes.core.discrete import HardLimiter, DeadBand  # NOQA


class VSCStatic(ACDC2Term):
    """Data for VSC in power flow"""
    def __init__(self, system, config):
        ACDC2Term.__init__(self, system, config)
        self.rsh = NumParam(default=0.0025, info="AC interface resistance", unit="ohm")
        self.xsh = NumParam(default=0.06, info="AC interface reactance", unit="ohm")

        self.control = NumParam(info="Control method of this VSC in PQ, PV, vQ or vV", mandatory=True)
        self.v0 = NumParam(default=1.0, info="AC voltage setting (PV or vV) or initial guess (PQ or vQ)")
        self.p0 = NumParam(default=0.0, info="AC active power setting", unit="pu")
        self.q0 = NumParam(default=0.0, info="AC reactive power setting", unit="pu")
        self.vdc0 = NumParam(default=1.0, info="DC voltage setting", unit="pu")

        self.k0 = NumParam(default=0.0, info="Loss coefficient - constant")
        self.k1 = NumParam(default=0.0, info="Loss coefficient - linear")
        self.k2 = NumParam(default=0.0, info="Loss coefficient - quadratic")

        self.droop = NumParam(default=0.0, info="Enable dc voltage droop control", unit="boolean")
        self.K = NumParam(default=0.0, info="Droop coefficient")
        self.vhigh = NumParam(default=9999, info="Upper voltage threshold in droop control", unit="pu")
        self.vlow = NumParam(default=0.0, info="Lower voltage threshold in droop control", unit="pu")

        self.vshmax = NumParam(default=1.1, info="Maximum ac interface voltage", unit="pu")
        self.vshmin = NumParam(default=0.9, info="Minimum ac interface voltage", unit="pu")
        self.Ishmax = NumParam(default=2, info="Maximum ac current", unit="pu")

        # define variables and equations
        self.flags.update({'pflow': True})

        self.ash = Algeb(info='voltage phase behind the transformer',
                         unit='rad',
                         tex_name=r'\theta_{sh}',
                         )
        self.vsh = Algeb(info='voltage magnitude behind transformer',
                         tex_name="V_{sh}",
                         unit='p.u.',
                         )
        self.psh = Algeb(info='active power injection into VSC',
                         tex_name="P_{sh}",
                         unit='p.u.',
                         )
        self.qsh = Algeb(info='reactive power injection into VSC',
                         tex_name="Q_{sh}",
                         )
        self.pdc = Algeb(info='DC power injection',
                         tex_name="P_{dc}",
                         )
        self.Ish = Algeb(info='converter ac injection current',
                         tex_name="I_{sh}",
                         )
