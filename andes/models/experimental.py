"""
Experimental Models
"""
from andes.core.model import ModelData, Model
from andes.core.var import Algeb, State
from andes.core.param import NumParam
from andes.core.discrete import HardLimiter, AntiWindup  # NOQA
from andes.core.block import DeadBand1, PIController, PITrackAW, PIFreeze  # NOQA


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

        # ==============================================================
        # TODO: BUG FIX: Deadband is causing some issue the mismatch.
        # ==============================================================
        # Max. iter. 15 reached for t=2.000100, h=0.000100, mis=4.091
        # Max. algebraic mismatch associated with uin TestDB1 1 [y_idx=8]
        #
        # y_index    Variable             Derivative
        # 8          uin TestDB1 1        -1
        # Max. correction is for variable uin TestDB1 1 [10]
        # Associated equation value is -4.09074
        #
        # xy_index   Equation             Derivative           Eq. Mismatch
        # 10         uin TestDB1 1        -1                   -4.09074
        # 11         DB_y TestDB1 1       1                    2.04537
        #
        # xy_index   Variable             Derivative           Eq. Mismatch
        # 10         uin TestDB1 1        -1                   -4.09074
        # 20%|██████▍                         | 20/100 [00:00<00:00, 291.18%/s]
        # ==============================================================

        # self.flags.tds = True
        #
        # self.uin = Algeb(v_str=0,
        #                  e_str='sin(dae_t) - uin',
        #                  )
        # self.DB = DeadBand1(self.uin, center=0, lower=-0.5, upper=0.5)


class TestPI(ModelData, Model):
    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Experimental'
        self.flags.tds = True

        self.uin = Algeb(v_str=0,
                         e_str='sin(dae_t) - uin',
                         )

        # `fz` works fine.
        self.fz = Algeb(v_str=0,
                        e_str='Piecewise((0, dae_t <= 2), (0, dae_t <=6), (0, True)) - fz',
                        )

        # `PI` works fine.
        self.PI = PIController(u=self.uin, kp=1, ki=0.1)

        # `PITrackAW` works fine (with Jacobian update on) with out freeze.
        self.PIAW = PITrackAW(u=self.uin, kp=0.5, ki=0.5, ks=2,
                              lower=-0.5, upper=0.5, x0=0.0,
                              )

        # TODO: If `freeze=self.fz`, pointer error will occur
        self.PIFreeze = PIFreeze(u=self.uin, kp=0.5, ki=0.5, x0=0,
                                 freeze=0.0)

    def get_times(self):
        return (2.0,)
