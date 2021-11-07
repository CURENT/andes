"""
Shunt-connected voltage-source converter for power flow.
"""

from andes.core import NumParam, ConstService, Switcher, Algeb
from andes.models.acdc.acdcbase import ACDC2Term


class VSCShunt(ACDC2Term):
    """Data for VSC Shunt in power flow"""

    def __init__(self, system, config):
        ACDC2Term.__init__(self, system, config)
        self.rsh = NumParam(default=0.0025, info="AC interface resistance", unit="ohm", z=True,
                            tex_name='r_{sh}')
        self.xsh = NumParam(default=0.06, info="AC interface reactance", unit="ohm", z=True,
                            tex_name='x_{sh}')

        self.control = NumParam(info="Control method: 0-PQ, 1-PV, 2-vQ or 3-vV", mandatory=True)
        self.v0 = NumParam(default=1.0, info="AC voltage setting (PV or vV) or initial guess (PQ or vQ)")
        self.p0 = NumParam(default=0.0, info="AC active power setting", unit="pu")
        self.q0 = NumParam(default=0.0, info="AC reactive power setting", unit="pu")
        self.vdc0 = NumParam(default=1.0, info="DC voltage setting", unit="pu", tex_name='v_{dc0}')

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
        self.group = 'StaticACDC'

        self.gsh = ConstService(tex_name='g_{sh}',
                                v_str='re(1/(rsh + 1j * xsh))',
                                )
        self.bsh = ConstService(tex_name='b_{sh}',
                                v_str='im(1/(rsh + 1j * xsh))',
                                )

        self.mode = Switcher(u=self.control, options=(0, 1, 2, 3))

        self.ash = Algeb(info='voltage phase behind the transformer',
                         unit='rad',
                         tex_name=r'\theta_{sh}',
                         v_str='a',
                         e_str='u * (gsh * v**2 - gsh * v * vsh * cos(a - ash) - '
                               'bsh * v * vsh * sin(a - ash)) - psh',
                         diag_eps=True,
                         )
        self.vsh = Algeb(info='voltage magnitude behind transformer',
                         tex_name="V_{sh}",
                         unit='p.u.',
                         v_str='v0',
                         e_str='u * (-bsh * v**2 - gsh * v * vsh * sin(a - ash) + '
                               'bsh * v * vsh * cos(a - ash)) - qsh',
                         diag_eps=True,
                         )
        self.psh = Algeb(info='active power injection into VSC',
                         tex_name="P_{sh}",
                         unit='p.u.',
                         v_str='p0 * (mode_s0 + mode_s1)',
                         e_str='u * (mode_s0 + mode_s1) * (p0 - psh) + '
                               'u * (mode_s2 + mode_s3) * (v1 - v2 - vdc0)',
                         diag_eps=True,
                         )
        self.qsh = Algeb(info='reactive power injection into VSC',
                         tex_name="Q_{sh}",
                         v_str='q0 * (mode_s0 + mode_s2)',
                         e_str='u * (mode_s0 + mode_s2) * (q0 - qsh) + '
                               'u * (mode_s1 + mode_s3) * (v0 - v)',
                         diag_eps=True,
                         )
        self.pdc = Algeb(info='DC power injection',
                         tex_name="P_{dc}",
                         v_str='0',
                         e_str='u * (gsh * vsh * vsh - gsh * v * vsh * cos(a - ash) + '
                               'bsh * v * vsh * sin(a - ash)) + pdc',
                         )
        self.a.e_str = '-psh'
        self.v.e_str = '-qsh'
        self.v1.e_str = '-pdc / (v1 - v2)'
        self.v2.e_str = 'pdc / (v1 - v2)'
