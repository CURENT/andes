"""
Phase measurement loop model.
"""
from andes.core.model import ModelData, Model
from andes.core.param import NumParam, IdxParam
from andes.core.block import PIController, Lag  # NOQA
from andes.core.var import ExtAlgeb, State


class PLLBaseData:
    """
    Common data for PLL.
    """

    def __init__(self) -> None:

        self.bus = IdxParam(info="bus idx", mandatory=True)

        self.fn = NumParam(default=60.0,
                           info="nominal frequency",
                           unit='Hz',
                           tex_name='f_n',
                           )


class PLLBaseModel:
    """
    Common implementation of PLL.
    """

    def __init__(self) -> None:

        self.flags.tds = True
        self.group = 'PLL'

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle'
                          )


class PLL1Data(ModelData, PLLBaseData):
    """
    Data for PLL.
    """

    def __init__(self):
        ModelData.__init__(self)

        PLLBaseData.__init__(self)

        self.Kp = NumParam(info='proportional gain', default=0.1,
                           tex_name='K_p',
                           )

        self.Ki = NumParam(info='integral gain', default=0.1,
                           tex_name='K_i',
                           )

        self.Tf = NumParam(default=0.05,
                           info="input digital filter time const",
                           unit="sec",
                           tex_name='T_f',
                           )
        self.Tp = NumParam(default=0.05,
                           info='output filter time const.',
                           unit='sec',
                           tex_name='T_p')


class PLL1Model(Model, PLLBaseModel):
    """
    Simple PLL1 implementation.
    """

    def __init__(self, system, config):
        super().__init__(system, config)

        PLLBaseModel.__init__(self)

        self.af = Lag(u=self.a, T=self.Tf, K=1, D=1,
                      info='input angle signal filter',
                      )

        self.PI = PIController(u='u * (af_y - am)', kp=self.Kp, ki=self.Ki,
                               tex_name='PI',
                               info='PI controller',
                               )

        self.ae = State(info='PLL angle output before filter',
                        e_str='2 * pi *fn * PI_y', v_str='a',
                        tex_name=r'\theta_{est}'
                        )

        self.am = State(info='PLL output angle after filtering',
                        e_str='ae - am',
                        t_const=self.Tp,
                        v_str='a',
                        tex_name=r'\theta_{m}'
                        )


class PLL1(PLL1Data, PLL1Model):
    """
    Simple Phasor Lock Loop (PLL) using one PI controller. The PI controller
    minimizes the error between the input and output angle.

    Input bus angle signal -> Lag filter 1 with Tf -> Output angle `af_y`.

    (af_y - am) -> PI Controller (Kp, Ki) -> PI_y

    Estimated angle ae = (2 * pi * fn * PI_y) -> Lag filter 2 with Tp -> am.

    The output signal is ``am``, a state variable.
    """

    def __init__(self, system, config):
        PLL1Data.__init__(self)
        PLL1Model.__init__(self, system, config)


class PLL2Data(ModelData, PLLBaseData):
    """
    Type-2 synchronously-rotating reference frame (SRF) PLL.
    """

    def __init__(self, *args, three_params=True, **kwargs):
        super().__init__(*args, three_params=three_params, **kwargs)

        PLLBaseData.__init__(self)

        self.Kp = NumParam(info='proportional gain', default=0.1,
                           tex_name='K_p',
                           )

        self.Ki = NumParam(info='integral gain', default=0.1,
                           tex_name='K_i',
                           )


class PLL2Model(Model, PLLBaseModel):
    """
    Implementation of PLL2.
    """

    def __init__(self, system=None, config=None):
        super().__init__(system, config)
        PLLBaseModel.__init__(self)

        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name='V',
                          info='Bus voltage magnitude'
                          )

        self.PI = PIController(u='v * sin(a - am)',
                               kp=self.Kp, ki=self.Ki, x0='0',
                               )

        self.am = State(info='PLL angle output',
                        e_str='2 * pi *fn * PI_y', v_str='a',
                        tex_name=r'\theta_m'
                        )


class PLL2(PLL2Data, PLL2Model):
    """
    Synchronously-rotating Reference Frame (SRF) Phasor Lock Loop (PLL).

    The PLL minimizes ``vq = v sin(a - am)`` using a PI controller.

    The output signal is ``am``, a state variable.
    """

    def __init__(self, system, config):
        PLL2Data.__init__(self)
        PLL2Model.__init__(self, system, config)
