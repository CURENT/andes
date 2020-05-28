from andes.core.param import TimerParam, IdxParam, DataParam, NumParam
from andes.core.model import Model, ModelData
from andes.core.var import ExtAlgeb
from andes.core.service import ConstService
from andes.shared import np, tqdm
import logging
logger = logging.getLogger(__name__)


class TogglerData(ModelData):
    def __init__(self):
        super(TogglerData, self).__init__()
        self.model = DataParam(info='Model or Group of the device to control', mandatory=True)
        self.dev = IdxParam(info='idx of the device to control', mandatory=True)
        self.t = TimerParam(info='switch time for connection status', mandatory=True)


class Toggler(TogglerData, Model):
    """
    Time-based connectivity status toggler.
    """
    def __init__(self, system, config):
        TogglerData.__init__(self)
        Model.__init__(self, system, config)
        self.flags.update({'tds': True})
        self.group = 'TimedEvent'

        self.t.callback = self._u_switch

    def _u_switch(self, is_time: np.ndarray):
        action = False
        for i in range(self.n):
            if is_time[i] and (self.u.v[i] == 1):
                instance = self.system.__dict__[self.model.v[i]]
                u0 = instance.get(src='u', attr='v', idx=self.dev.v[i])
                instance.set(src='u', attr='v', idx=self.dev.v[i], value=1-u0)
                action = True
                tqdm.write(f'<Toggler {self.idx.v[i]}>: '
                           f'{self.model.v[i]}.{self.dev.v[i]} status changed to {1-u0} at t={self.t.v[i]} sec.')
        return action


class Fault(ModelData, Model):
    """
    Three-phase to ground fault.
    """
    def __init__(self, system, config):
        ModelData.__init__(self)
        self.bus = IdxParam(model='Bus',
                            info="linked bus idx",
                            mandatory=True,
                            )
        self.tf = TimerParam(info='Bus fault start time',
                             unit='second',
                             mandatory=True,
                             callback=self.apply_fault,
                             )
        self.tc = TimerParam(info='Bus fault end time',
                             unit='second',
                             callback=self.clear_fault,
                             )
        self.xf = NumParam(info='Fault to ground impedance (positive)',
                           unit='p.u.(sys)',
                           default=1e-4,
                           tex_name='x_f',
                           )
        self.rf = NumParam(info='Fault to ground resistance (positive)',
                           unit='p.u.(sys)',
                           default=0,
                           tex_name='x_f',
                           )

        Model.__init__(self, system, config)
        self.flags.update({'tds': True})
        self.group = 'TimedEvent'
        self.gf = ConstService(tex_name='g_{f}',
                               v_str='re(1/(rf + 1j * xf))',
                               vtype=np.complex,
                               )
        self.bf = ConstService(tex_name='b_{f}',
                               v_str='im(1/(rf + 1j * xf))',
                               vtype=np.complex,
                               )

        # uf: an internal flag of whether the fault is in action (1) or not (0)
        self.uf = ConstService(tex_name='u_f', v_str='0')

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle',
                          unit='p.u.(kV)',
                          e_str='u * uf * (v ** 2 * gf)',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          unit='p.u.(kV)',
                          info='Bus voltage magnitude',
                          e_str='-u * uf * (v ** 2 * bf)',
                          )
        self._vstore = np.array([])

    def apply_fault(self, is_time: np.ndarray):
        """
        Apply fault and store pre-fault algebraic variables (voltages and other algebs) to `self._vstore`.
        """
        action = False
        for i in range(self.n):
            if is_time[i] and (self.u.v[i] == 1):
                self.uf.v[i] = 1
                self._vstore = np.array(self.system.dae.y[self.system.Bus.n:])
                tqdm.write(f'<Fault {self.idx.v[i]}>: '
                           f'Applying fault on Bus (idx={self.bus.v[i]}) at t={self.tf.v[i]}sec.')
                action = True
        return action

    def clear_fault(self, is_time: np.ndarray):
        """
        Clear fault and restore pre-fault bus algebraic variables (voltages and others).
        """
        action = False
        for i in range(self.n):
            if is_time[i] and (self.u.v[i] == 1):
                self.uf.v[i] = 0
                self.system.dae.y[self.system.Bus.n:] = self._vstore
                tqdm.write(f'<Fault {self.idx.v[i]}>: '
                           f'Clearing fault on Bus (idx={self.bus.v[i]}) at t={self.tc.v[i]}sec.')
                action = True
        return action
