import logging
from collections import OrderedDict

import numpy as np

from andes.core.discrete import Switcher
from andes.core.model import Model, ModelData
from andes.core.param import DataParam, IdxParam, NumParam, TimerParam
from andes.core.service import ConstService
from andes.core.var import ExtAlgeb
from andes.shared import tqdm

logger = logging.getLogger(__name__)


class ToggleData(ModelData):
    def __init__(self):
        super(ToggleData, self).__init__()
        self.model = DataParam(info='model or group name of the device',
                               mandatory=True,
                               )
        self.dev = IdxParam(info='idx of the device to control',
                            mandatory=True,
                            )
        self.t = TimerParam(info='switch time for connection status',
                            mandatory=True,
                            )


class Toggle(ToggleData, Model):
    """
    Time-based connectivity status toggle.

    Toggle is used to toggle the connection status (online/offline) of a device
    at the predefined time. Both the model name (or group name) and the device
    idx need to be specified. It effectively negates the ``u`` field of the
    connected device.

    Toggle can be useful to implement disconnection, connection, and
    reconnection of devices. For example, a line trip can be implemented by
    setting ``Line`` to the ``model`` field and the corresponding line's ``idx``
    to the ``dev`` field.

    Multiple Toggles can be added to the same device at different times. Adding
    two Toggles for an initially connected line with ``t=0.1`` and ``t=0.2``,
    for instance, will disconnect the line at t=0.1 sec and reconnect it at
    t=0.2 sec.
    """

    def __init__(self, system, config):
        ToggleData.__init__(self)
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
            tqdm.write(f'<Toggle {self.idx.v[i]}>: '
                       f'{self.model.v[i]}.{self.dev.v[i]} status '
                       f'changed to {1-u0:g} at t={self.t.v[i]} sec.')
        return action


class Fault(ModelData, Model):
    """
    Three-phase-to-ground fault.

    A Fault device is used to apply and clear three-phase-to-ground fault to the
    given bus. One can set two time parameters, ``tf`` and ``tc``, for the
    fault-on and fault-clearance time, respectively, although only ``tf`` is
    mandatory.

    A fault is implemented by a very small internal shunt impedance to be
    connected at the fault-on time. Its reactance and resistance are specified
    by the parameters ``xf`` and ``rf``.

    To implement a fault and its clearance by tripping a line, one can combine
    ``Fault`` and ``Toggle``. That is, clear a fault in concurrence with a
    Toggle. The user needs to ensure data consistency so that the line trip
    actually clears the fault.

    Non-convergence can occur in the proximity of a fault due to various reasons,
    including network power transfer capability limitation and parameter issues
    of controllers.
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
        self.xf = NumParam(info='Fault to ground reactance (positive)',
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
                                     ('mode', 1),
                                     ('scale', 1.0),
                                     )))
        self.config.add_extra('_alt',
                              restore=(0, 1),
                              mode=(1, 2, 3),
                              )
        self.config.add_extra('_help',
                              restore='restore algebraic variables to pre-fault values',
                              mode='1. restore all algeb variables, 2. fault bus only',
                              scale='scaling factor of restored algebraic values',
                              )

        self.gf = ConstService(tex_name='g_{f}',
                               v_str='re(1/(rf + 1j * xf))',
                               )
        self.bf = ConstService(tex_name='b_{f}',
                               v_str='im(1/(rf + 1j * xf))',
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
                          ename='P',
                          tex_ename='P',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          unit='p.u.(kV)',
                          info='Bus voltage magnitude',
                          e_str='-u * uf * (v ** 2 * bf)',
                          ename='Q',
                          tex_ename='Q',
                          )
        self._vstore = np.array([])

    def apply_fault(self, is_time: np.ndarray):
        """
        Apply fault and store pre-fault algebraic variables (voltages and other
        algebs) to `self._vstore`.
        """
        action = False
        for i in range(self.n):
            if (is_time[i] == 0) or (self.u.v[i] == 0):
                continue

            self.uf.v[i] = 1
            self._vstore = np.array(self.system.dae.y[self.system.Bus.n:])
            logger.debug("Pre-fault algebraic variables:\n" + str(self._vstore))
            tqdm.write(f'<Fault {self.idx.v[i]}>: '
                       f'Applying fault on Bus (idx={self.bus.v[i]}) at t={self.tf.v[i]} sec.')

            action = True
        return action

    def clear_fault(self, is_time: np.ndarray):
        """
        Clear fault and restore pre-fault bus algebraic variables (voltages and
        others).
        """
        action = False
        for i in range(self.n):
            if is_time[i] and (self.u.v[i] == 1):
                self.uf.v[i] = 0

                if self.config.restore:
                    if self.config.mode == 1:
                        self.system.dae.y[self.system.Bus.n:] = self._vstore * self.config.scale
                        logger.debug("All algebraic variables restored after fault clearance at t=%.6f",
                                     self.system.dae.t)

                    # TODO: neither mode 2 or 3 works. Pending further investigation.
                    elif self.config.mode == 2:
                        v_addr = self.system.Bus.get(src='v', idx=self.bus.v[i], attr='a')
                        bus_uid = self.system.Bus.idx2uid(self.bus.v[i])
                        self.system.dae.y[v_addr] = self._vstore[bus_uid] * self.config.scale
                        logger.debug("Voltage on bus %s restored after fault clearance at t=%.6f",
                                     self.bus.v[i], self.system.dae.t)
                    elif self.config.mode == 3:
                        nbus = self.system.Bus.n
                        self.system.dae.y[nbus:2*nbus] = self._vstore[:nbus] * self.config.scale
                        logger.debug("All bus voltages restored after fault clearance at t=%.6f",
                                     self.system.dae.t)
                    else:
                        logger.error("Unsupport fault voltage restoration mode")

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
    """
    Implementation of the Alter model.
    """

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
    Model for altering device internal data at predefined time.

    Alter is useful to apply load changing, tap changing, step response, etc.
    can be applied to parameters and constant services but cannot be used to
    update variables.

    Alter is implemented by applying the given calculation to the ``v`` field of
    the linked parameter or constant. Alter will not affect other parameters or
    constants that depend on the altered variable.

    It is not uncommon for equations to depend on intermediate constants rather
    than the input parameters. Therefore, one will need to inspect model
    equations to determine the parameter/service to be altered.

    Examples
    --------
    To apply a PQ load change, according to :ref:`PQ`, one needs to set the load
    model to constant power and alter ``Ppf`` and ``Qpf``. Altering ``p0`` and
    ``q0`` will have no impact as they are not used in the equations for
    time-domain simulation.

    """

    def __init__(self, system, config):
        AlterData.__init__(self)
        AlterModel.__init__(self, system, config)
