"""
DER protection model.
"""
from andes.core.param import IdxParam, NumParam, ExtParam
from andes.core.model import Model, ModelData
from andes.core.var import Algeb, ExtAlgeb
from andes.core.service import ConstService
from andes.core.discrete import Limiter, Delay


class PlockData(ModelData):
    """
    Plock model data.
    """

    def __init__(self):
        super(PlockData, self).__init__()
        self.dev = IdxParam(info='idx of the target device',
                            mandatory=True,
                            )
        self.busfreq = IdxParam(model='BusFreq',
                                info='Target device interface bus measurement idx',
                                )

        # -- protection enable parameters
        self.fena = NumParam(default=1,
                             tex_name='fena',
                             info='Frequency deviation protection enable. \
                                   1 for enable, 0 for disable.',
                             )
        self.Vena = NumParam(default=0,
                             tex_name='Vena',
                             info='Voltage deviation protection enable.\
                                   1 for enable, 0 for disable.',
                             )

        # -- protection parameters
        self.fl = NumParam(default=57.5,
                           tex_name='fl',
                           info='Under frequency shadding point',
                           unit='Hz',
                           )
        self.fu = NumParam(default=61.5,
                           tex_name='fu',
                           info='Over frequency shadding point',
                           unit='Hz',
                           )
        self.ul = NumParam(default=0.88,
                           tex_name='fl',
                           info='Under voltage shadding point',
                           unit='p.u.',
                           )
        self.uu = NumParam(default=1.1,
                           tex_name='fu',
                           info='Over voltage shadding point',
                           unit='p.u.',
                           )
        self.tf = NumParam(default=10,
                           tex_name=r't_{fdev}',
                           info='Stand time under frequency deviation',
                           )
        self.tv = NumParam(default=1,
                           tex_name=r't_{udev}',
                           info='Stand time under voltage deviation',
                           )


class PlockModel(Model):
    """
    Model implementation of Plock.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'DG'

        self.bus = ExtParam(model='DG',
                            src='bus',
                            indexer=self.dev,
                            export=False)

        # -- Voltage protection
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          export=False,
                          info='Bus voltage',
                          unit='p.u.',
                          )
        # Indicatior of voltage deviation
        self.Vcmp = Limiter(u=self.v,
                            lower=self.ul,
                            upper=self.uu,
                            tex_name=r'V_{cmp}',
                            info='Voltage comparator',
                            equal=False,
                            )
        self.Volt_dev = Algeb(v_str='0',
                              e_str='1 - Vcmp_zi - Volt_dev',
                              info='Show Volt_devs',
                              tex_name='zs_{Vdev}',
                              )
        # Delayed frequency deviation indicator
        self.Volt_devd = Delay(u=self.Volt_dev,
                               mode='time',
                               delay=self.tv.v)

        # -- Frequency protection
        self.fn = ExtParam(model='DG',
                           src='fn',
                           indexer=self.dev,
                           export=False)
        # Convert frequency deviation range to p.u.
        self.fln = ConstService(tex_name=r'fl_{n}',
                                v_str='fl/fn',
                                )
        self.fun = ConstService(tex_name=r'fu_{n}',
                                v_str='fu/fn',
                                )
        self.f = ExtAlgeb(model='FreqMeasurement',
                          src='f',
                          indexer=self.busfreq,
                          export=False,
                          info='Bus frequency',
                          unit='p.u.',
                          )

        # Indicatior of drequency deviation
        self.fcmp = Limiter(u=self.f,
                            lower=self.fln,
                            upper=self.fun,
                            tex_name=r'f_{cmp}',
                            info='Frequency comparator',
                            equal=False,
                            )
        self.freq_dev = Algeb(v_str='0',
                              e_str='1 - fcmp_zi - freq_dev',
                              info='Show freq_devs',
                              tex_name='zs_{Fdev}',
                              )
        # Delayed frequency deviation indicator
        self.freq_devd = Delay(u=self.freq_dev,
                               mode='time',
                               delay=self.tf.v)

        # -- Shut PVD1 current command
        # freqyency protection
        self.Ipul_f = ExtAlgeb(model='DG',
                               src='Ipul',
                               indexer=self.dev,
                               export=False,
                               e_str='-1000 * Ipul_f * freq_devd_v * freq_dev * fena',
                               info='Shut PVD1 current based on frequency protection',
                               )
        # voltage protection
        self.Ipul_V = ExtAlgeb(model='DG',
                               src='Ipul',
                               indexer=self.dev,
                               export=False,
                               e_str='-1000 * Ipul_V * Volt_devd_v * Volt_dev * Vena',
                               info='Shut PVD1 current based on voltage protection',
                               )


class Plock(PlockData, PlockModel):
    """
    Plock is distributed energy resource protection model.
    The target device given variable will be shut down under
    protection condition.
    The protection contains frequency deviation and voltage
    deviation.
    """

    def __init__(self, system, config):
        PlockData.__init__(self)
        PlockModel.__init__(self, system, config)
