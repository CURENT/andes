
"""
Inertia estimation model
"""

from andes.core import ConstService, NumParam, ModelData, Model, IdxParam, ExtState, State
from andes.core.block import  Gain, Integrator, Lag, PIDController, PIController
from andes.core.discrete import Limiter

class InertiaEstimation(ModelData, Model):
    """
    Estimates inertia of a device. Outputs estimation in pu value.
    """ 

    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
       

        #parameters

        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.syn = IdxParam(model='SynGen',
                            info='Synchronous generator idx',
                            mandatory=True,
                            unique=True,
                            )
        self.epsilon = NumParam(default=0.000001,
                           info="tolerance",
                           unit="pu",
                           tex_name=r'\epsilon',
                           )
        self.Tm = NumParam(default=0.01,
                           info="Time Constant",
                           unit="sec",
                           tex_name='T_m',
                           )

        self.iTm = ConstService(v_str = "1/Tm",
                           tex_name='1/Tm',
                           )

        self.Tf = NumParam(default=0.0001,
                           info="filter time const",
                           unit="sec",
                           tex_name='T_f',
                           )
        self.Kp = NumParam(default=50,
                           info="proportional constant",
                           unit="pu",
                           tex_name='K_p',
                           )
        self.Ki = NumParam(default=1,
                           info="integral constant",
                           unit="pu",
                           tex_name='K_i'
                           )
        #variables

        self.omegadot = ExtState(model='SynGen',
                          src='omega',
                          indexer=self.syn,
                          tex_name = r'\dot \omega'
                          )
        self.pe_dot = ExtState(model='SynGen',
                          src='Pe',
                          indexer=self.syn,
                          tex_name = r'\dot p'
                          )
        #PI controller
        self.omegadotstar = State(info = 'omegadotstar',
                               v_str = 'omegadot * Kp * Ki',
                               e_str = 'omegadot * Kp * Ki - omegadotstar * Kp * Ki')
        
        self.omegadoubledot = State(info = 'omegadoubledot',
                               v_str = 'omegadotstar * Kp',
                               e_str = 'omegadotstar * Kp - omegadot * Kp - 2 * omegadoubledot')
        #main blocks
        self.gamma = Limiter(u=self.omegadoubledot, lower=self.epsilon, upper=self.epsilon,
                                  equal= True, sign_lower= -1, 
                            )
        self.M_star = State(v_str = '(gamma_zl*1 + gamma_zu*-1) * (pe_dot)',
                            e_str = '(gamma_zl*1 + gamma_zu*-1) * (pe_dot + M_star*omegadoubledot)',
                            t_const= self.Tm,
                            info = "Estimated Inertia"
                            )



