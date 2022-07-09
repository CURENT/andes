"""
Inertia estimation model
Used Blocks and Limiter 
"""
from andes.core import ConstService, NumParam, ModelData, Model, IdxParam, ExtState, State, ExtAlgeb, ExtParam, Algeb
from andes.core.block import  Washout, Integrator, Gain, Lag
from andes.core.discrete import Limiter

class ines(ModelData, Model):
    """
    Estimates inertia of a device. Outputs estimation in pu value.
    """ 

    def __init__(self, system, config):
        ModelData.__init__(self)
        Model.__init__(self, system, config)
        self.flags.update({'tds': True})
        #parameters
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
        self.Two = NumParam(default=0.0001,
                           info="washout time const",
                           unit="sec",
                           tex_name='T_wo',
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
        self.damping = ExtParam(src='D',
                                model='SynGen',
                                indexer=self.syn,
                                tex_name = 'damping',
                                export = True
                                )
        self.Mg = ExtParam(src='M',
                            model='SynGen',
                            indexer=self.syn,
                            tex_name = 'generator inertia',
                            export = True
                            )
        self.ug = ExtParam(src='u',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name = 'generator connectivity',
                           export = True
                           )
        #variables
        self.omega = ExtState(src='omega',
                                model='SynGen',
                                indexer=self.syn,
                                tex_name = r'\dot \omega',
                                export = True
                                )
        self.tm = ExtAlgeb(src='tm',
                          model='SynGen',
                          indexer=self.syn,
                          tex_name = 'mechanical torque of generator',
                          export = True
                          )
        self.te = ExtAlgeb(src='te',
                          model='SynGen',
                          indexer=self.syn,
                          tex_name = 'electrical torque of generator',
                          export = True
                          )
                
        self.omega_dot = Algeb(tex_name = r'\dot \omega', info = r'\dot \omega',
                              v_str = 'ug * (-1 * damping * (omega - 1) - te + tm) / Mg',
                              e_str = 'ug * (-1 * damping * (omega - 1) - te + tm) / Mg - omega_dot'
                              )
        
        self.Pe = ExtAlgeb(src='Pe',
                           model='SynGen',
                           indexer=self.syn,
                           tex_name = 'Pe',
                           export = True
                           )
        self.pe_dot = Washout(u = self.Pe, K = 1,
                            T = self.Two)
        #PI controller
        self.k_omega = Gain(u = "omega_dot - omegadot_star_y", 
                            K = self.Kp
                            )        
        self.omegadot_star = Integrator(u = self.k_omega_y, T = 1, K = self.Ki, 
                                        y0 = '0', check_init = False
                                        )
        self.omegadoubledot = Lag(u = "k_omega_y - omegadoubledot_y",
                                  K = 1, T = self.Tf
                                  )        
        #main blocks
        self.gamma = Limiter(u=self.omegadoubledot_y, lower=self.epsilon, upper=self.epsilon,
                                  equal= True, sign_lower= -1, allow_adjust= False 
                            )
        self.M_star = State(v_str = '(gamma_zl*1 + gamma_zu*-1 ) * (pe_dot_y + M_star*omegadoubledot_y)',
                            e_str = '(gamma_zl*1 + gamma_zu*-1 ) * (pe_dot_y + M_star*omegadoubledot_y)',
                            t_const= self.Tm,
                            info = "Estimated Inertia"
                            )
