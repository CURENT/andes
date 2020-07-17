"""
Experimental Models
"""
from andes.core.model import ModelData, Model
from andes.core.var import Algeb, State
from andes.core.param import NumParam
from andes.core.discrete import HardLimiter, AntiWindup  # NOQA
from andes.core.block import DeadBand1


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

        self.uin = State(v_str=0,
                         e_str='Piecewise((0, dae_t<=0), (1, dae_t<=2), (-1, dae_t<6), (1, True))',
                         )
        self.DB = DeadBand1(self.uin, center=0, lower=-1, upper=1)
