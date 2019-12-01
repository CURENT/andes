from andes.core.model import Model, ModelData
from andes.core.param import NumParam, IdxParam, ExtParam
from andes.core.var import State, Algeb, ExtState, ExtAlgeb
from andes.core.service import ServiceConst, ExtService
from andes.core.discrete import HardLimiter, DeadBand


class TGBaseData(ModelData):
    def __init__(self):
        super().__init__()
        self.syn = IdxParam(model='SynGen', info='Synchronous generator idx', mandatory=True)
        self.R = NumParam(info='Speed regulation gain', tex_name='R', default=0.05)
        self.pmax = NumParam(info='Maximum power output', tex_name='p_{max}', power=True, default=999.0)
        self.pmin = NumParam(info='Minimum power output', tex_name='p_{min}', power=True, default=0.0)
        self.wref0 = NumParam(info='Base speed reference', tex_name=r'\omega_{ref0}', default=1.0)
        self.dbl = NumParam(info='Deadband lower limit', tex_name='dbL', default=0.999667)
        self.dbu = NumParam(info='Deadband upper limit', tex_name='dbU', default=1.000333)
        self.dbc = NumParam(info='Deadband center', tex_name='dbU', default=1.0)


class TGBase(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.update({'tds': True})
        self.config.add({'deadband': 1})

        self.Sn = ExtParam(src='Sn', model='SynGen', indexer=self.syn, tex_name='S_m')
        self.pm0 = ExtService(src='pm', model='SynGen', indexer=self.syn, tex_name='p_{m0}')
        self.omega = ExtState(src='omega', model='SynGen', indexer=self.syn, tex_name=r'\omega')
        self.pm = ExtAlgeb(src='pm', model='SynGen', indexer=self.syn,tex_name='P_m',
                           e_str='u*(pout - pm0)')
        self.pnl = Algeb(info='Power output before hard limiter', tex_name='P_{nl}',
                         v_init='pm0')
        self.pout = Algeb(info='Turbine power output after limiter', tex_name='P_{out}',
                          v_init='pm0')
        self.wref = Algeb(info='Speed referemce variable', tex_name=r'\omega_{ref}',
                          v_init='wref0', e_str='wref0 - wref')


class TG2Data(TGBaseData):
    def __init__(self):
        super().__init__()
        self.T1 = NumParam(info='Transient gain time', default=0.2)
        self.T2 = NumParam(info='Governor time constant', default=10.0)


class TG2(TG2Data, TGBase):
    def __init__(self, system, config):
        TG2Data.__init__(self)
        TGBase.__init__(self, system, config)
        self.tex_names = {'plim_zl': r'z_{P,l}',
                          'plim_zi': r'z_{P,i}',
                          'plim_zu': r'z_{P,u}',
                          'omega_db_zl': 'z_{db,l}',
                          'omega_db_zi': 'z_{db,i}',
                          'omega_db_zu': 'z_{db,u}',
                          }

        self.T12 = ServiceConst(v_str='T1 / T2')
        self.gain = ServiceConst(v_str='u / R', tex_name='G')

        self.xg = State(tex_name='x_g', e_str='(((1 - T12) * gain * (wref0 - omega_m)) - xg) / T2',
                        v_setter=True)

        self.pnl.e_str = 'pm0 + xg + (gain * T12 * (wref0 - omega_m)) - pnl'
        self.pout.e_str = 'pnl * plim_zi + \
                           pmax * plim_zu + \
                           pmin * plim_zl - \
                           pout'

        self.plim = HardLimiter(var=self.pout, origin=self.pnl, lower=self.pmin, upper=self.pmax)

        self.omega_m = Algeb(info='Measured generator speed after deadband', tex_name=r'\omega_{m}',
                             v_init='1', diag_eps=1e-6)

        self.omega_m.e_str = 'omega * (1 - omega_db_zi) + \
                              dbc * omega_db_zi - \
                              omega_m'
        self.omega_m.e_str = 'omega - omega_m'
        self.omega_db = DeadBand(var=self.omega_m, origin=self.omega,
                                 center=self.dbc, lower=self.dbl, upper=self.dbu,
                                 enable=self.config.deadband)

# Developing a model (use TG2 as an example)
# 0) Find the group class or write a new group class in group.py
# 1) Determine and write the class `TG2Data` derived from ModelData
# 2) Write an empty class for the Model by inheriting `TG2Data` and `Model`
# 3) Implement the `__init__` function
#    a) Call parent class `__init__` methods.
#    b) Set `self.flags` for `pflow` and `tds` if applicable.
#    c) Define external service, algeb and states
#    d) Define base class variables and equations
# 4) Implement the TG2 class variables `pout` and add a limiter
