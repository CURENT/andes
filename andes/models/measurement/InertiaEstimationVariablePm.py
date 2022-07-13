"""
Inertia estimation model based on swing equation using actual Pm data 
"""

from andes.core import ConstService, NumParam, ModelData, Model, IdxParam, ExtState, State, ExtAlgeb, ExtParam, Algeb
from andes.core.block import  Piecewise
from andes.core.discrete import Derivative


class InertiaEstimationVariablePm(ModelData, Model):
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
        self.gov = IdxParam(model='TurbineGov',
                            info='tgov idx',
                            mandatory=True,
                            unique=True,
                            )
        self.epsilon = NumParam(default=0.000001,
                           info="tolerance",
                           unit="p.u.",
                           tex_name=r'\epsilon',
                           )
        self.negepsilon = ConstService(v_str = '-1 * epsilon')
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
                           unit="p.u.",
                           tex_name='K_p',
                           )
        self.Ki = NumParam(default=1,
                           info="integral constant",
                           unit="p.u.",
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
        self.omega_alg = Algeb(v_str = 'omega',
                               e_str = 'omega - omega_alg'
                               )
        self.omega_fd = Derivative(u = self.omega_alg, 
                                  info= r'\dot \omega from finte differentiation'
                                  )
        self.Pm = ExtAlgeb(src='pout',
                           model='TurbineGov',
                           indexer=self.gov,
                           tex_name = 'Pm',
                           export = True
                           )
        
        
        #self.Pm = PostInitService(info='Initial Pe',
        #                     tex_name='P_m', v_str='Pe'
        #                     )
       
        #main blocks
 
        self.piece = Piecewise(u = self.omega_dot, points= ['negepsilon', 'epsilon'], funs= [1, 0, -1], 
                               name = 'piece')    
        self.M_star = State(v_str = 'piece_y * ( M_star * omega_fd_v - (Pm - Pe))',
                            e_str = 'piece_y * ( M_star * omega_fd_v - (Pm - Pe))',
                            t_const= self.Tm,
                            info = "Estimated Inertia"
                            )
