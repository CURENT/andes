import logging
import numpy as np

from collections import OrderedDict

from andes.shared import tqdm
from andes.core.param import TimerParam, IdxParam, DataParam, NumParam
from andes.core.model import Model, ModelData
from andes.core.var import ExtAlgeb
from andes.core.service import ConstService
from andes.core.discrete import Switcher

logger = logging.getLogger(__name__)


class TogglerData(ModelData):
    def __init__(self):
        super(TogglerData, self).__init__()
        self.model = DataParam(info='model or group name of the device',
                               mandatory=True,
                               )
        self.dev = IdxParam(info='idx of the device to control',
                            mandatory=True,
                            )
        self.t = TimerParam(info='switch time for connection status',
                            mandatory=True,
                            )


class Toggler(TogglerData, Model):
    """
    Time-based connectivity status toggler.

    Toggler is used to toggle the connection status
    of a device at a predefined time.
    Both the model name (or group name) and the device
    idx need to be provided.
    """

    def __init__(self, system, config):
        TogglerData.__init__(self)
        Model.__init__(self, system, config)
        self.flags.update({'tds': True})
        self.group = 'TimedEvent'

        self.t.callback = self._u_switch
        self._init = False   # very first initialization that stores `u`
        self._u = ConstService('1')

    def v_numeric(self, **kwargs):
        """
        Custom initialization function that stores and restores the connectivity status.
        """
        if not self._init:
            for i in range(self.n):
                instance = self.system.__dict__[self.model.v[i]]
                self._u.v[i] = instance.get(src='u', attr='v', idx=self.dev.v[i])
                self._init = True
        else:
            for i in range(self.n):
                instance = self.system.__dict__[self.model.v[i]]
                instance.set(src='u', attr='v', idx=self.dev.v[i], value=self._u.v[i])

    def _u_switch(self, is_time: np.ndarray):
        action = False
        for i in range(self.n):
            if (is_time[i] == 0) or (self.u.v[i] == 0):
                continue

            instance = self.system.__dict__[self.model.v[i]]
            u0 = instance.get(src='u', attr='v', idx=self.dev.v[i])
            instance.set(src='u', attr='v', idx=self.dev.v[i], value=1-u0)
            action = True
            tqdm.write(f'<Toggler {self.idx.v[i]}>: '
                       f'{self.model.v[i]}.{self.dev.v[i]} status '
                       f'changed to {1-u0:g} at t={self.t.v[i]} sec.')
        return action


class Fault(ModelData, Model):
    """
    Three-phase to ground fault.

    Two times, `tf` and `tc`, can be defined for fault on
    for fault clearance.
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

        self.config.add(OrderedDict((('restore', 1),
                                     ('scale', 1.0)
                                     )))
        self.config.add_extra('_alt',
                              restore=(0, 1),
                              )
        self.config.add_extra('_help',
                              restore='restore algebraic variables to pre-fault values',
                              scale='scaling factor of restored algebraic values',
                              )

        self.gf = ConstService(tex_name='g_{f}',
                               v_str='re(1/(rf + 1j * xf))',
                               vtype=complex,
                               )
        self.bf = ConstService(tex_name='b_{f}',
                               v_str='im(1/(rf + 1j * xf))',
                               vtype=complex,
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
            if (is_time[i] == 0) or (self.u.v[i] == 0):
                continue

            self.uf.v[i] = 1
            self._vstore = np.array(self.system.dae.y[self.system.Bus.n:])
            tqdm.write(f'<Fault {self.idx.v[i]}>: '
                       f'Applying fault on Bus (idx={self.bus.v[i]}) at t={self.tf.v[i]} sec.')

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

                if self.config.restore:
                    self.system.dae.y[self.system.Bus.n:] = self._vstore * self.config.scale
                    logger.debug(f"Voltage restored after fault clearance at t={self.system.dae.t:.6f}")

                tqdm.write(f'<Fault {self.idx.v[i]}>: '
                           f'Clearing fault on Bus (idx={self.bus.v[i]}) at t={self.tc.v[i]} sec.')

                action = True
        return action


class AlterData(ModelData):
    """
    Data for Alter, which altera values of the given device at a certain time.

    Alter can be used in various timed applications, such as applying load changing,
    tap changing, step response, etc.
    """

    def __init__(self):
        ModelData.__init__(self)
        self.t = TimerParam(info='switch time for connection status', mandatory=True)

        self.model = DataParam(info='model or group name of the device', mandatory=True)
        self.dev = IdxParam(info='idx of the device to alter', mandatory=True)
        self.src = IdxParam(info='model source field (param or service)', mandatory=True)
        self.attr = IdxParam(info='attribute (e.g., v) of the source field', default='v')

        self.method = NumParam(info='alteration method in `+`, `-`, `*`, `/`, `=`',
                               mandatory=True, vtype=object)
        self.amount = NumParam(info='the amount to apply', mandatory=True,)

        self.rand = NumParam(info='use uniform ramdom sampling', default=0)
        self.lb = NumParam(info='lower bound of random sampling', default=0)
        self.ub = NumParam(info='upper bound of random sampling', default=0)


class AlterModel(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'TimedEvent'

        self.SW = Switcher(u=self.method, options=('+', '-', '*', '/', '='),
                           info='Switcher for alteration method',
                           )

        self.t.callback = self._alter_field

    def _alter_field(self, is_time):
        """
        Actuation of the alteration.
        """
        action = False

        for ii in range(self.n):
            if (not is_time[ii]) or (self.u.v[ii] == 0):
                continue

            model = self.system.__dict__[self.model.v[ii]]
            idx = self.dev.v[ii]
            src = self.src.v[ii]
            attr = self.attr.v[ii]
            amount = self.amount.v[ii]

            if self.rand.v[ii] == 1:
                amount = np.random.uniform(low=self.lb.v[ii], high=self.ub.v[ii])

            try:
                v0 = model.get(src=src, idx=idx, attr=attr)
            except KeyError as e:
                tqdm.write("\nError: <%s %s> cannot find idx=%s or src=%s in model <%s>. " % (
                    self.class_name, self.idx.v[ii],
                    idx, src, self.model.v[ii],
                ))
                tqdm.write("<%s %s> disabled due to %s.\n" %
                           (self.class_name, self.idx.v[ii], repr(e)))
                self.u.v[ii] = 0
                continue

            vnew = v0
            if self.SW.s0[ii] == 1:
                vnew = v0 + amount
            elif self.SW.s1[ii] == 1:
                vnew = v0 - amount
            elif self.SW.s2[ii] == 1:
                vnew = v0 * amount
            elif self.SW.s3[ii] == 1:
                vnew = v0 / amount
            elif self.SW.s4[ii] == 1:
                vnew = amount
            else:
                tqdm.write('Error: <%s %s>: undefined method "%s". <%s, %s> disabled.' % (
                    self.class_name, self.idx.v[ii], self.method.v[ii],
                    self.class_name, self.idx.v[ii]
                ))
                self.u.v[ii] = 0
                continue

            model.set(src=src, idx=idx, attr=attr, value=vnew)
            tqdm.write('<Alter %s>: set %s.%s.%s.%s=%.6g at t=%.6g. Previous value was %.6g.' % (
                self.idx.v[ii], self.model.v[ii], idx, src, attr, vnew, self.t.v[ii], v0
            ))
            action = True

        return action


class Alter(AlterData, AlterModel):
    """
    Model for altering device internal data (service or param)
    at a given time.
    """

    def __init__(self, system, config):
        AlterData.__init__(self)
        AlterModel.__init__(self, system, config)
