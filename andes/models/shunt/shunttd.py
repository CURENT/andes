"""
Three-phase symmetric shunt model.
"""

from andes.core import Algeb
from andes.models.shunt import Shunt


class ShuntTD(Shunt):
    """
    Static shunt model with inverse transformation from phasor to time-domain.
    """

    def __init__(self, system=None, config=None):
        Shunt.__init__(self, system, config)
        self.vta = Algeb(e_str='v / sqrt(3) * cos(2*pi * sys_f * dae_t + a) - vta',
                         v_str='v / sqrt(3) * cos(2*pi * sys_f * dae_t + a)',
                         tex_name='V_{ta}',
                         )
        self.vtb = Algeb(e_str='v / sqrt(3) * cos(2*pi * sys_f * dae_t + a - 2/3*pi) - vtb',
                         v_str='v / sqrt(3) * cos(2*pi * sys_f * dae_t + a - 2/3*pi)',
                         tex_name='V_{tb}',
                         )
        self.vtc = Algeb(e_str='v / sqrt(3) * cos(2*pi * sys_f * dae_t + a + 2/3*pi) - vtc',
                         v_str='v / sqrt(3) * cos(2*pi * sys_f * dae_t + a + 2/3*pi)',
                         tex_name='V_{tc}',
                         )
