"""
Renewable energy generator (converter) model P with PLL support.

This is a customized model in ANDES, modified from the standard REGCA1 model.
"""

from andes.core import Algeb, ExtState, IdxParam
from andes.core.service import DeviceFinder
from andes.models.renewable.regca1 import REGCA1Data, REGCA1Model


class REGCP1Data(REGCA1Data):
    """
    REGCP1 model data.
    """

    def __init__(self):
        REGCA1Data.__init__(self)

        self.pll = IdxParam(info='Phase-lock loop device idx',
                            model='PLL',
                            default=None,
                            )


class REGCP1Model(REGCA1Model):
    """
    REGCP1 implementation.
    """

    def __init__(self, system, config):
        REGCA1Model.__init__(self, system, config)
        self.pllidx = DeviceFinder(self.pll,
                                   link=self.bus,
                                   idx_name='bus',
                                   default_model='PLL1',
                                   )

        self.am = ExtState(model='PLL', src='am', indexer=self.pllidx)

        self.vd = Algeb(v_str='v', info='d-axis voltage', tex_name='V_d',
                        e_str='v*cos(a - am) - vd')

        self.vq = Algeb(v_str='0', info='q-axis voltage', tex_name='V_q',
                        e_str='- v*sin(a - am) - vq')

        self.Pe.e_str = '(vd * Ipout + vq * Iqout_y) - Pe'

        self.Qe.e_str = '(vd * Iqout_y - vq * Ipout) - Qe'

    def v_numeric(self, **kwargs):
        """
        Disable the corresponding `StaticGen`s.
        """
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class REGCP1(REGCP1Data, REGCP1Model):
    """
    Renewable energy generator model type A with PLL.

    A PLL device needs to be specified for estimating the phase angle at the
    coupling bus. If not provided, a PLL1 device will be used, but one should
    carefully tune the PLL parameters to match the desired performance.

    All remarks for ``REGCA1`` apply.
    """

    def __init__(self, system, config):
        REGCP1Data.__init__(self)
        REGCP1Model.__init__(self, system, config)
