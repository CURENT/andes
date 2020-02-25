from andes.core.param import TimerParam, IdxParam, DataParam
from andes.core.model import Model, ModelData
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
                tqdm.write(f'<Toggle {i}>: Toggling status for {self.model.v[i]} idx={self.dev.v[i]}.')
        return action
