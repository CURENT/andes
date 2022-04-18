"""
Frequency-dependent load.
"""

from andes.core import (ModelData, IdxParam, NumParam, Model,
                        ExtParam, ExtService, ExtAlgeb, ConstService)
from andes.core.service import DeviceFinder


class FLoadData(ModelData):
    """
    Data for frequency dependent load.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.pq = IdxParam(model='PQ', mandatory=True,
                           info='idx of the PQ to replace',
                           )
        self.busf = IdxParam(model='BusFreq',
                             info='optional idx of the BusFreq device to use',
                             )

        self.kp = NumParam(info='active power percentage',
                           default=100.0,
                           unit='%',
                           )
        self.kq = NumParam(info='active power percentage',
                           default=100.0,
                           unit='%',
                           )

        self.Tf = NumParam(info='filter time constant',
                           unit='s',
                           default=0.02,
                           non_negative=True,
                           )
        self.ap = NumParam(info='active power voltage exponent',
                           default=1.0,
                           )
        self.aq = NumParam(info='reactive power voltage exponent',
                           default=0.0,
                           )
        self.bp = NumParam(info='active power frequency exponent',
                           default=0.0,
                           )
        self.bq = NumParam(info='reactive power frequency exponent',
                           default=0.0,
                           )


class FLoadModel(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.group = 'DynLoad'
        self.flags.tds = True

        self.bus = ExtParam(model='PQ', src='bus', indexer=self.pq)

        self.p0 = ExtService(model='PQ', src='Ppf', indexer=self.pq,
                             tex_name='P_0',
                             )
        self.q0 = ExtService(model='PQ', src='Qpf', indexer=self.pq,
                             tex_name='Q_0',
                             )
        self.v0 = ExtService(model='Bus', src='v', indexer=self.bus,
                             tex_name='V_0',
                             )

        self.busfreq = DeviceFinder(u=self.busf, link=self.bus, idx_name='bus',
                                    info='found idx of BusFreq',
                                    default_model='BusFreq',
                                    )

        self.f = ExtAlgeb(model='FreqMeasurement', src='f', indexer=self.busfreq,
                          tex_name='f',
                          )

        self.pv0 = ConstService(v_str='u * kp/100 * p0 / (v0) ** ap ')
        self.qv0 = ConstService(v_str='u * kq/100 * q0 / (v0) ** aq ')

        self.a = ExtAlgeb(model='Bus', src='a', indexer=self.bus,
                          tex_name=r'\theta',
                          e_str='pv0 * (v ** ap) * (f ** bp)',
                          ename='P',
                          tex_ename='P',
                          )

        self.v = ExtAlgeb(model='Bus', src='v', indexer=self.bus,
                          tex_name='V',
                          e_str='qv0 * (v ** aq) * (f ** bq)',
                          ename='Q',
                          tex_ename='Q',
                          )

    def v_numeric(self, **kwargs):
        """
        Disable the linked PQs.
        """
        self.system.groups['StaticLoad'].set(src='u', idx=self.pq.v, attr='v', value=0)


class FLoad(FLoadData, FLoadModel):
    """
    Voltage and frequency dependent load.
    """

    def __init__(self, system, config):
        FLoadData.__init__(self)
        FLoadModel.__init__(self, system, config)
