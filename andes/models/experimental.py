"""
Experimental Models
"""
from andes.core.model import ModelData, Model
from andes.core.var import Algeb, State
from andes.core.param import NumParam
from andes.core.discrete import HardLimiter, AntiWindup  # NOQA
from andes.core.block import DeadBand1, PIController, PITrackAW, PIFreeze, PITrackAWFreeze  # NOQA


class PI2Data(ModelData):
    """Data for PI2 model with deadlock issue"""
    def __init__(self):
        ModelData.__init__(self)
        self.Kp = NumParam()
        self.Ki = NumParam()
        self.Wmax = NumParam()
        self.Wmin = NumParam()


class PI2Model(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'Experimental'
        self.flags.update({'tds': True})
        self.uin = State(v_str=0,
                         e_str='Piecewise((0, dae_t<= 0), (1, dae_t <= 2), (-1, dae_t <6), (1, True))',
                         tex_name='u_{in}',
                         )
        self.x = State(e_str='uin * Ki * HL_zi',
                       v_str=0.05,
                       )
        self.y = Algeb(e_str='uin * Kp + x - y',
                       v_str=0.05)

        self.HL = HardLimiter(u=self.y, lower=self.Wmin, upper=self.Wmax)
        self.w = Algeb(e_str='HL_zi * y + HL_zl * Wmin + HL_zu * Wmax - w',
                       v_str=0.05)


class PI2(PI2Data, PI2Model):
    def __init__(self, system, config):
        PI2Data.__init__(self)
        PI2Model.__init__(self, system, config)


class TestDB1(ModelData, Model):
    """
    Test model for `DeadBand1`.
    """
    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Experimental'

        self.flags.tds = True

        self.uin = Algeb(v_str='-10',
                         e_str='(dae_t - 10) - uin',
                         tex_name='u_{in}'
                         )
        self.DB = DeadBand1(self.uin, center=0, lower=-5, upper=5)


class TestPI(ModelData, Model):
    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Experimental'
        self.flags.tds = True

        self.uin = Algeb(v_str=0,
                         e_str='sin(dae_t) - uin',
                         tex_name='u_{in}',
                         )

        self.zf = Algeb(v_str=0,
                        e_str='Piecewise((0, dae_t <= 2), (1, dae_t <=6), (0, True)) - zf',
                        tex_name='z_f',
                        )

        self.PI = PIController(u=self.uin, kp=1, ki=0.1)

        self.PIAW = PITrackAW(u=self.uin, kp=0.5, ki=0.5, ks=2,
                              lower=-0.5, upper=0.5, x0=0.0,
                              )

        self.PIF = PIFreeze(u=self.uin, kp=0.5, ki=0.5, x0=0,
                            freeze=self.zf)

        self.PIAWF = PITrackAWFreeze(u=self.uin, kp=0.5, ki=0.5, ks=2, x0=0,
                                     freeze=self.zf, lower=-0.5, upper=0.5)

    def get_times(self):
        return (2.0, 6.0)
