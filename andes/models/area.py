from andes.core.param import ExtParam, NumParam, IdxParam
from andes.core.model import Model, ModelData
from andes.core.var import ExtAlgeb, Algeb
from andes.core.service import NumReduce, NumRepeat, BackRef, DeviceFinder
from andes.core.discrete import Sampling
from andes.shared import np
from andes.utils.tab import Tab
from collections import OrderedDict


class AreaData(ModelData):
    def __init__(self):
        super().__init__()


class Area(AreaData, Model):
    def __init__(self, system, config):
        AreaData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'Collection'
        self.flags.pflow = True
        self.flags.tds = True

        self.Bus = BackRef()
        self.ACTopology = BackRef()

        # --------------------Experiment Zone--------------------
        self.Vn = ExtParam(model='Bus', src='Vn', indexer=self.ACTopology, export=False)
        self.Vn_sum = NumReduce(u=self.Vn, fun=np.sum, ref=self.Bus)
        self.Vn_sum_rep = NumRepeat(u=self.Vn_sum, ref=self.Bus)

        self.a = ExtAlgeb(model='ACTopology', src='a', indexer=self.ACTopology,
                          info='Bus voltage angle')
        self.v = ExtAlgeb(model='ACTopology', src='v', indexer=self.ACTopology,
                          info='Bus voltage magnitude')

        # self.time = Algeb(e_str='time - dae_t', v_setter=True)

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

        self.busf = IdxParam(info='Optional BusFreq idx', model='BusFreq',
                             default=None)


class ACE(ACEData, Model):
    """
    Area Control Error model.

    Note: area idx is automatically retrieved from `bus`.
    """

    def __init__(self, system, config):
        ACEData.__init__(self)
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.group = 'Calculation'

        self.config.add(OrderedDict([('freq_model', 'BusFreq'),
                                     ('interval', 4.0),
                                     ('offset', 0.0),
                                     ]))
        self.config.add_extra('_help', {'freq_model': 'default freq. measurement model',
                                        'interval': 'sampling time interval',
                                        'offset': 'sampling time offset'})

        self.config.add_extra('_alt', {'freq_model': ('BusFreq',)})

        self.area = ExtParam(model='Bus', src='area', indexer=self.bus, export=False)

        self.busf.model = self.config.freq_model
        self.busfreq = DeviceFinder(self.busf, link=self.bus, idx_name='bus')

        self.f = ExtAlgeb(model='FreqMeasurement', src='f', indexer=self.busfreq,
                          export=False, info='Bus frequency',
                          )

        self.fs = Sampling(self.f,
                           interval=self.config.interval,
                           offset=self.config.offset,
                           tex_name='f_s',
                           info='Sampled freq.',
                           )

        self.ace = Algeb(info='area control error', unit='MW (p.u.)',
                         tex_name='ace',
                         e_str='10 * bias * (fs_v - 1) - ace',
                         )
