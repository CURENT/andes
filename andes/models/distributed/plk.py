"""
DER protection model.
"""
from andes.core.param import IdxParam, NumParam, ExtParam
from andes.core.model import Model, ModelData
from andes.core.var import Algeb, ExtAlgeb
from andes.core.service import ConstService
from andes.core.discrete import Limiter, Delay


class PLKData(ModelData):
    """
    Plock model data.
    """

    def __init__(self):
        super(PLKData, self).__init__()
        self.dev = IdxParam(info='idx of the target device',
                            mandatory=True,
                            )
        self.busfreq = IdxParam(model='BusFreq',
                                info='Target device interface bus measurement device idx',
                                )

        # -- protection enable parameters
        self.fena = NumParam(default=1,
                             tex_name='fena',
                             vrange=(0, 1),
                             info='Frequency deviation protection enable. \
                                   1 for enable, 0 for disable.',
                             )
        self.Vena = NumParam(default=0,
                             tex_name='Vena',
                             vrange=(0, 1),
                             info='Voltage deviation protection enable.\
                                   1 for enable, 0 for disable.',
                             )

        # -- protection parameters
        self.fl = NumParam(default=57.5,
                           tex_name='f_l',
                           info='Under frequency shadding point',
                           unit='Hz',
                           )
        self.fu = NumParam(default=61.5,
                           tex_name='f_u',
                           info='Over frequency shadding point',
                           unit='Hz',
                           )
        self.vl = NumParam(default=0.88,
                           tex_name='V_l',
                           info='Under voltage shadding point',
                           unit='p.u.',
                           )
        self.vu = NumParam(default=1.1,
                           tex_name='V_u',
                           info='Over voltage shadding point',
                           unit='p.u.',
                           )
        self.Tf = NumParam(default=10,
                           tex_name=r't_{fdev}',
                           info='Stand time under frequency deviation',
                           non_negative=True,
                           )
        self.Tv = NumParam(default=1,
                           tex_name=r't_{udev}',
                           info='Stand time under voltage deviation',
                           non_negative=True,
                           )


class PLKModel(Model):
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
                            lower=self.vl,
                            upper=self.vu,
                            tex_name=r'V_{cmp}',
                            info='Voltage comparator',
                            equal=False,
                            )
        self.Volt_dev = Algeb(v_str='0',
                              e_str='1 - Vcmp_zi - Volt_dev',
                              info='Voltage deviation indicator',
                              tex_name='zs_{Vdev}',
                              )
        # Delayed voltage deviation indicator
        self.Volt_devd = Delay(u=self.Volt_dev,
                               mode='time',
                               delay=self.Tv.v)

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

        # Indicatior of frequency deviation
        self.fcmp = Limiter(u=self.f,
                            lower=self.fln,
                            upper=self.fun,
                            tex_name=r'f_{cmp}',
                            info='Frequency comparator',
                            equal=False,
                            )
        self.freq_dev = Algeb(v_str='0',
                              e_str='1 - fcmp_zi - freq_dev',
                              info='Frequency deviation indicator',
                              tex_name='zs_{Fdev}',
                              )
        # Delayed frequency deviation indicator
        self.freq_devd = Delay(u=self.freq_dev,
                               mode='time',
                               delay=self.Tf.v)

        # -- Lock PVD1 current command
        # freqyency protection
        self.Ipul_f = ExtAlgeb(model='DG',
                               src='Ipul',
                               indexer=self.dev,
                               export=False,
                               e_str='-1000 * Ipul_f * freq_devd_v * freq_dev * fena',
                               info='Current locker from frequency protection',
                               )
        # voltage protection
        self.Ipul_V = ExtAlgeb(model='DG',
                               src='Ipul',
                               indexer=self.dev,
                               export=False,
                               e_str='-1000 * Ipul_V * Volt_devd_v * Volt_dev * Vena',
                               info='Current locker from voltage protection',
                               )


class PLK(PLKData, PLKModel):
    """
    DER protection model. PLK stands for Power Lock.

    Target device (limited to DG group) ``Ipul`` will drop to zero immediately
    when frequency/voltage protection is triggered.

    Once the lock is released, ``Ipul`` will return to normal immediately.

    ``fena`` and ``Vena`` are protection enabling parameters. 1 is on and 0 is off.
    """

    def __init__(self, system, config):
        PLKData.__init__(self)
        PLKModel.__init__(self, system, config)
