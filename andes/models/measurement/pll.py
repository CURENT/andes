"""
Phase measurement loop model.
"""
from andes.core.model import ModelData, Model
from andes.core.param import NumParam, IdxParam
from andes.core.block import PIController, Lag  # NOQA
from andes.core.var import ExtAlgeb, State


class PLL1Data(ModelData):
    """
    Data for PLL.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.bus = IdxParam(info="bus idx", mandatory=True)

        self.Kp = NumParam(info='proportional gain', default=1,
                           tex_name='K_p',
                           )

        self.Ki = NumParam(info='integral gain', default=0.2,
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

        self.fn = NumParam(default=60.0,
                           info="nominal frequency",
                           unit='Hz',
                           tex_name='f_n',
                           )


class PLL1Model(Model):
    """
    Simple PLL1 implementation.
    """

    def __init__(self, system, config):
        super().__init__(system, config)

        self.flags.tds = True
        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle'
                          )

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
                        tex_name=r'\theta_{PLL}'
                        )


class PLL1(PLL1Data, PLL1Model):
    """
    Simple Phasor Lock Loop (PLL) using one PI controller.

    Input bus angle signal -> Lag filter 1 with Tf ->
    PI Controller (Kp, Ki) -> Estimated angle (2 * pi * fn * PI_y) ->
    Lag filter 2 with Tp

    The output signal is ``am``.
    """

    def __init__(self, system, config):
        PLL1Data.__init__(self)
        PLL1Model.__init__(self, system, config)
