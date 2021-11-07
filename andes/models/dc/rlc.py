"""
R, L, C and their combinations.
"""

from andes.core import NumParam, Algeb, State
from andes.models.dc.dcbase import DC2Term


class R(DC2Term):
    """
    Resistive dc line
    """

    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'
        self.R = NumParam(unit='p.u.',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         v_str='u * (v2 - v1) / R',
                         e_str='u * (v2 - v1) / R - Idc',
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class L(DC2Term):
    """
    Inductive dc line
    """

    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.L = NumParam(unit='p.u.',
                          info='DC line inductance',
                          non_zero=True,
                          default=0.001,
                          r=True,
                          )
        self.IL = State(tex_name='I_L',
                        info='Inductance current',
                        unit='p.u.',
                        v_str='0',
                        e_str='-u * (v1 - v2)',
                        t_const=self.L,
                        )
        self.v1.e_str = '-IL'
        self.v2.e_str = '+IL'


class C(DC2Term):
    """
    Capacitive dc branch
    """

    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.C = NumParam(unit='p.u.',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        v_str='0',
                        e_str='-u * Idc',
                        t_const=self.C
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         v_str='0',
                         e_str='u * (vC - (v1 - v2)) + '
                               '(1 - u) * Idc',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RLs(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.L = NumParam(unit='p.u.',
                          tex_name='L',
                          info='DC line inductance',
                          non_zero=True,
                          default=0.001,
                          r=True,
                          )
        self.IL = State(tex_name='I_L',
                        info='Inductance current',
                        unit='p.u.',
                        e_str='u * (v1 - v2 - R * IL)',
                        v_str='(v1 - v2) / R',
                        t_const=self.L,
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='-u * IL - Idc',
                         v_str='-u * (v1 - v2) / R',
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RCp(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.C = NumParam(unit='p.u.',
                          tex_name='C',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        e_str='-u * (Idc - vC/R)',
                        v_str='v1 - v2',
                        t_const=self.C,
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='u * (vC - (v1 - v2)) + '
                               '(1 - u) * Idc',
                         v_str='-(v1 - v2) / R',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RLCp(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.L = NumParam(unit='p.u.',
                          tex_name='L',
                          info='DC line inductance',
                          non_zero=True,
                          default=0.001,
                          r=True,
                          )
        self.C = NumParam(unit='p.u.',
                          tex_name='C',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.IL = State(tex_name='I_L',
                        info='Inductance current',
                        unit='p.u.',
                        v_str='0',
                        e_str='u * vC',
                        t_const=self.L,
                        )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        e_str='-u * (Idc - vC/R - IL)',
                        v_str='v1 - v2',
                        t_const=self.C,
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='u * (vC - (v1 - v2)) + '
                               '(1 - u) * Idc',
                         v_str='-(v1 - v2) / R',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RCs(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.C = NumParam(unit='p.u.',
                          tex_name='C',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        e_str='-u * Idc',
                        v_str='v1 - v2',
                        t_const=self.C,
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='u * (vC - (v1 - v2) - Idc * R) + '
                               '(1 - u) * Idc',
                         v_str='-(v1 - v2) / R',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'


class RLCs(DC2Term):
    def __init__(self, system, config):
        DC2Term.__init__(self, system, config)
        self.flags.pflow = True
        self.group = 'DCLink'

        self.R = NumParam(unit='p.u.',
                          tex_name='R',
                          info='DC line resistance',
                          non_zero=True,
                          default=0.01,
                          r=True,
                          )
        self.L = NumParam(unit='p.u.',
                          tex_name='L',
                          info='DC line inductance',
                          non_zero=True,
                          default=0.001,
                          r=True,
                          )
        self.C = NumParam(unit='p.u.',
                          tex_name='C',
                          info='DC capacitance',
                          non_zero=True,
                          default=0.001,
                          g=True,
                          )
        self.IL = State(tex_name='I_L',
                        info='Inductance current',
                        unit='p.u.',
                        e_str='u * (v1 - v2 - R * IL - vC)',
                        v_str='0',
                        t_const=self.L,
                        )
        self.vC = State(tex_name='v_C',
                        info='Capacitor current',
                        unit='p.u.',
                        e_str='u * IL',
                        v_str='v1 - v2',
                        t_const=self.C,
                        )
        self.Idc = Algeb(tex_name='I_{dc}',
                         info='Current from node 2 to 1',
                         unit='p.u.',
                         e_str='-IL - Idc',
                         v_str='0',
                         diag_eps=True,
                         )
        self.v1.e_str = '-Idc'
        self.v2.e_str = '+Idc'
