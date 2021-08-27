"""Electric vehicle model"""

from andes.core.param import NumParam
from andes.core.var import Algeb
from andes.core.discrete import Limiter

from andes.models.distributed.esd1 import ESD1Data, ESD1Model


class EV1Data(ESD1Data):
    """
    Data for electric vehicle model.
    """

    def __init__(self):
        ESD1Data.__init__(self)
        self.pmn = NumParam(default=-999.0, info='minimum power limit',
                            tex_name='p_{mn}',
                            power=True,
                            unit='pu',
                            )


class EV1Model(ESD1Model):
    """
    Model implementation of EV1.
    """

    def __init__(self, system, config):
        ESD1Model.__init__(self, system, config)

        # Modify limiter for Psum
        # Change PHL enable to 1 from config.plim
        # Change PHL lower to pmn from 0
        self.PHL.enable = 1
        self.PHL.lower = self.pmn
        self.PHL.info = 'limiter for Psum in [pmn, pmx]'

        self.Ipul.v_str = '(Psum * PHL_zi + pmx * PHL_zu + pmn * PHL_zl) / vp'
        self.Ipul.e_str = '(Psum * PHL_zi + pmx * PHL_zu + pmn * PHL_zl) / vp - Ipul'

        self.Ipcmd.lower = '-Ipmax'


class EV1(EV1Data, EV1Model):
    """
    Electric vehicle model type 1.

    Modified from ESD1 model by adding the minumum power limit `pmn`.
    Like `pmx`, `pmn` acts on `Psum`, the sum of the active power references.

    The limiter that uses `pmx` and `pmn` is enabled by default.
    """
    def __init__(self, system, config):
        EV1Data.__init__(self)
        EV1Model.__init__(self, system, config)


class EV2Data(EV1Data):
    """
    Data for electric vehicle model 2.
    """

    def __init__(self):
        EV1Data.__init__(self)
        self.pcap = NumParam(default=0,
                             info='power ratio multiplied to pmx in [-1, 1]',
                             tex_name='p_{cap}',
                             vrange=(-1, 1),
                             )

        # Increase the ddn default,
        # so the output power will be determined by power cap
        self.ddn.default = 1


class EV2Model(EV1Model):
    """
    Model implementation of EV2.
    """

    def __init__(self, system, config):
        EV1Model.__init__(self, system, config)

        # Modify power limit PHL
        self.PHLup = Algeb(info='PHL upper limit',
                           v_str='pcap * pmx',
                           e_str='pcap * pmx - PHLup',
                           tex_name=r'PHL_{upper}',
                           )

        self.PHL2 = Limiter(u=self.Psum, lower=self.pmn, upper=self.PHLup,
                            enable=1,
                            info='limiter for Psum in [pmn, pcap * pmx]',
                            )

        # Modify
        self.Ipul.v_str = '(Psum * PHL2_zi + PHLup * PHL2_zu + pmn * PHL2_zl) / vp'
        self.Ipul.e_str = '(Psum * PHL2_zi + PHLup * PHL2_zu + pmn * PHL2_zl) / vp - Ipul'


class EV2(EV2Data, EV2Model):
    """
    Electric vehicle model type 2.

    Derived from EV1, EV2 introduces `pcap` multiplied to `pmx`.

    `Psum` will be limited to [pmn, pmx * pcap].

    The model does not check the signs or values of `pmn`,
    `pmx`, or `pcap`.
    The input data is required to satisfy `pmn <= pmx * pcap`.
    """

    def __init__(self, system, config):
        EV2Data.__init__(self)
        EV2Model.__init__(self, system, config)
