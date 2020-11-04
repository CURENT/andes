from andes.core.param import ExtParam, NumParam, IdxParam
from andes.core.model import Model, ModelData
from andes.core.var import ExtAlgeb, Algeb
from andes.core.service import ConstService
from andes.core.service import BackRef, DeviceFinder
from andes.core.discrete import Sampling
from andes.utils.tab import Tab
from collections import OrderedDict


class AreaData(ModelData):
    def __init__(self):
        super().__init__()


class Area(AreaData, Model):
    """
    Area model.

    Area collects back references from the Bus model and
    the ACTopology group.
    """
    def __init__(self, system, config):
        AreaData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Collection'
        self.flags.pflow = True
        self.flags.tds = True

        self.Bus = BackRef()
        self.ACTopology = BackRef()

    def bus_table(self):
        """
        Return a formatted table with area idx and bus idx correspondence

        Returns
        -------
        str
            Formatted table

        """
        if self.n:
            header = ['Area ID', 'Bus ID']
            rows = [(i, j) for i, j in zip(self.idx.v, self.Bus.v)]
            return Tab(header=header, data=rows).draw()
        else:
            return ''


class ACEData(ModelData):
    """
    Area Control Error data
    """

    def __init__(self):
        ModelData.__init__(self)
        self.bus = IdxParam(model='Bus', info="bus idx for freq. measurement", mandatory=True)
        self.bias = NumParam(default=1.0, info='bias parameter', tex_name=r'\beta',
                             unit='MW/0.1Hz', power=True)

        self.busf = IdxParam(info='Optional BusFreq device idx', model='BusFreq',
                             default=None)


class ACEc(ACEData, Model):
    """
    Area Control Error model.

    Continuous frequency sampling.
    System base frequency from ``system.config.freq`` is used.

    Note: area idx is automatically retrieved from `bus`.
    """

    def __init__(self, system, config):
        ACEData.__init__(self)
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'Calculation'

        self.config.add(OrderedDict([('freq_model', 'BusFreq'),
                                     ]))
        self.config.add_extra('_help',
                              {'freq_model': 'default freq. measurement model',
                               })
        self.config.add_extra('_alt', {'freq_model': ('BusFreq',)})

        self.area = ExtParam(model='Bus', src='area', indexer=self.bus, export=False)

        self.busf.model = self.config.freq_model
        self.busfreq = DeviceFinder(self.busf, link=self.bus, idx_name='bus')

        self.imva = ConstService(v_str='1/sys_mva', info='reciprocal of system mva',
                                 tex_name='1/S_{b, sys}')

        self.f = ExtAlgeb(model='FreqMeasurement',
                          src='f',
                          indexer=self.busfreq,
                          export=False,
                          info='Bus frequency',
                          unit='p.u. (Hz)'
                          )
        self.ace = Algeb(info='area control error',
                         unit='p.u. (MW)',
                         tex_name='ace',
                         e_str='10 * (bias * imva) * sys_f * (f - 1) - ace',
                         )


class ACE(ACEc):
    """
    Area Control Error model.

    Discrete frequency sampling.
    System base frequency from ``system.config.freq`` is used.

    Frequency sampling period (in seconds) can be specified in
    ``ACE.config.interval``. The sampling start time (in seconds)
    can be specified in ``ACE.config.offset``.

    Note: area idx is automatically retrieved from `bus`.
    """

    def __init__(self, system, config):
        ACEc.__init__(self, system, config)

        self.config.add(OrderedDict([('interval', 4.0),
                                     ('offset', 0.0),
                                     ]))
        self.config.add_extra('_help', {'interval': 'sampling time interval',
                                        'offset': 'sampling time offset'})

        self.fs = Sampling(self.f,
                           interval=self.config.interval,
                           offset=self.config.offset,
                           tex_name='f_s',
                           info='Sampled freq.',
                           )

        self.ace.e_str = '10 * (bias * imva) * sys_f * (fs_v - 1) - ace'
