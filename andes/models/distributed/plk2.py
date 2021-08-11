"""
DER protection model.
"""
from andes.core.param import IdxParam, NumParam, ExtParam
from andes.core.model import Model, ModelData
from andes.core.var import Algeb, ExtAlgeb
from andes.core.service import ConstService
from andes.core.discrete import Limiter, Delay, HardLimiter


class PLK2Data(ModelData):
    """
    PLK2 model data.
    """

    def __init__(self):
        super(PLK2Data, self).__init__()
        self.dev = IdxParam(info='idx of the target device',
                            mandatory=True,
                            )
        self.busfreq = IdxParam(model='BusFreq',
                                info='Target device interface bus measurement device idx',
                                )

        # -- protection enable parameters
        self.fena = NumParam(default=1,
                             tex_name='fena',
                             vrange=[0, 1],
                             info='Frequency deviation protection enable. \
                                   1 for enable, 0 for disable.',
                             )
        self.Vena = NumParam(default=0,
                             tex_name='Vena',
                             vrange=[0, 1],
                             info='Voltage deviation protection enable.\
                                   1 for enable, 0 for disable.',
                             )

        # -- protection parameters, frequency
        self.fl1 = NumParam(default=59.2,
                            tex_name='fl1',
                            info='Under frequency shadding point 1',
                            unit='Hz',
                            )
        self.fu1 = NumParam(default=60.5,
                            tex_name='fu1',
                            info='Over frequency shadding point 1',
                            unit='Hz',
                            )
        self.fl2 = NumParam(default=57.5,
                            tex_name='fl2',
                            info='Over frequency shadding point 2',
                            unit='Hz',
                            )
        self.fu2 = NumParam(default=61.5,
                            tex_name='fu2',
                            info='Over frequency shadding point 2',
                            unit='Hz',
                            )
        self.fl3 = NumParam(default=50,
                            tex_name='fl3',
                            info='Under frequency shadding point 3',
                            unit='Hz',
                            )
        self.fu3 = NumParam(default=70,
                            tex_name='fu3',
                            info='Over frequency shadding point 3',
                            unit='Hz',
                            )

        self.Tfl1 = NumParam(default=300,
                             tex_name=r't_{fl1}',
                             info='Stand time for (fl2, fl1)',
                             non_negative=True,
                             )
        self.Tfl2 = NumParam(default=10,
                             tex_name=r't_{fl2}',
                             info='Stand time for (fl3, fl2)',
                             non_negative=True,
                             )
        self.Tfu1 = NumParam(default=300,
                             tex_name=r't_{tu1}',
                             info='Stand time for (fu1, fu2)',
                             non_negative=True,
                             )
        self.Tfu2 = NumParam(default=10,
                             tex_name=r't_{fu2}',
                             info='Stand time for (fu2, fu3)',
                             non_negative=True,
                             )

        # -- protection parameters, voltage
        self.ul1 = NumParam(default=0.88,
                            tex_name='ul1',
                            info='Under voltage shadding point 1',
                            unit='p.u.',
                            )
        self.uu1 = NumParam(default=1.1,
                            tex_name='uu1',
                            info='Over voltage shadding point 1',
                            unit='p.u.',
                            )
        self.ul2 = NumParam(default=0.6,
                            tex_name='ul2',
                            info='Under voltage shadding point 2',
                            unit='p.u.',
                            )
        self.uu2 = NumParam(default=1.2,
                            tex_name='uu2',
                            info='Over voltage shadding point 2',
                            unit='p.u.',
                            )
        self.ul3 = NumParam(default=0.45,
                            tex_name='ul3',
                            info='Under voltage shadding point 3',
                            unit='p.u.',
                            )
        self.ul4 = NumParam(default=0.1,
                            tex_name='ul4',
                            info='Under voltage shadding point 4',
                            unit='p.u.',
                            )
        self.uu3 = NumParam(default=2,
                            tex_name='uu3',
                            info='Over voltage shadding point 3',
                            unit='p.u.',
                            )

        self.Tvl1 = NumParam(default=2,
                             tex_name=r't_{vl1}',
                             info='Stand time for (ul2, ul1)',
                             non_negative=True,
                             )
        self.Tvl2 = NumParam(default=1,
                             tex_name=r't_{vl2}',
                             info='Stand time for (ul3, ul2)',
                             non_negative=True,
                             )
        self.Tvl3 = NumParam(default=0.16,
                             tex_name=r't_{vl3}',
                             info='Stand time for (ul4, ul3)',
                             non_negative=True,
                             )
        self.Tvu1 = NumParam(default=1,
                             tex_name=r't_{vu1}',
                             info='Stand time for (uu1, uu2)',
                             non_negative=True,
                             )
        self.Tvu2 = NumParam(default=0.16,
                             tex_name=r't_{vu2}',
                             info='Stand time for (uu1, uu2)',
                             non_negative=True,
                             )


class PLK2Model(Model):
    """
    Model implementation of PLK2.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.tds = True
        self.group = 'DG'

        self.bus = ExtParam(model='DG',
                            src='bus',
                            indexer=self.dev,
                            export=False)

        # -- Frequency protection
        self.fn = ExtParam(model='DG',
                           src='fn',
                           indexer=self.dev,
                           export=False)
        # Convert frequency deviation range to p.u.
        self.f = ExtAlgeb(model='FreqMeasurement',
                          src='f',
                          indexer=self.busfreq,
                          export=False,
                          info='Bus frequency',
                          unit='p.u.',
                          )

        # Indicatior of frequency deviation
        self.fcl1 = Limiter(u=self.f,
                            lower=self.fl2,
                            upper=self.fl1,
                            tex_name=r'f_{cl1}',
                            info='Frequency comparer for (fl2, fl1)',
                            equal=False,
                            )
        self.fdl1 = Algeb(v_str='0',
                          e_str='fcl1_zi - fdl1',
                          info='Frequency  deviation indicator for (fl2, fl1)',
                          tex_name='zs_{fdl1}',
                          )

        self.fcl2 = Limiter(u=self.f,
                            lower=self.fl3,
                            upper=self.fl2,
                            tex_name=r'f_{cl2}',
                            info='Frequency comparer for (fl2, fl1)',
                            equal=False,
                            )
        self.fdl2 = Algeb(v_str='0',
                          e_str='fcl2_zi - fdl2',
                          info='Frequency  deviation indicator for (fl2, fl1)',
                          tex_name='zs_{fdl2}',
                          )

        self.fcu1 = Limiter(u=self.f,
                            lower=self.fu1,
                            upper=self.fu2,
                            tex_name=r'f_{cu1}',
                            info='Frequency comparer for (fu1, fu2)',
                            equal=False,
                            )
        self.fdu1 = Algeb(v_str='0',
                          e_str='fcu1_zi - fdu1',
                          info='Frequency  deviation indicator for (fu1, fu2)',
                          tex_name='zs_{fdu1}',
                          )

        self.fcu2 = Limiter(u=self.f,
                            lower=self.fu2,
                            upper=self.fu3,
                            tex_name=r'f_{cu2}',
                            info='Frequency comparer for (fu2, fu3)',
                            equal=False,
                            )
        self.fdu2 = Algeb(v_str='0',
                          e_str='fcu2_zi - fdu2',
                          info='Frequency  deviation indicator for (fu2, fu3)',
                          tex_name='zs_{fdu2}',
                          )

        # Delayed frequency deviation indicator
        self.fdl1d = Delay(u=self.fdl1,
                           mode='time',
                           delay=self.Tfl2.v)
        self.fdl2d = Delay(u=self.fdl2,
                           mode='time',
                           delay=self.Tfl1.v)
        self.fdu1d = Delay(u=self.fdu1,
                           mode='time',
                           delay=self.Tfu1.v)
        self.fdu2d = Delay(u=self.fdu2,
                           mode='time',
                           delay=self.Tfu2.v)

        # -- Voltage protection
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          export=False,
                          info='Bus voltage',
                          unit='p.u.',
                          )
        # Indicatior of voltage deviation
        self.Vcl1 = Limiter(u=self.v,
                            lower=self.ul2,
                            upper=self.ul1,
                            tex_name=r'V_{cl1}',
                            info='Voltage comparer for (ul2, ul1)',
                            equal=False,
                            )
        self.Vdl1 = Algeb(v_str='0',
                          e_str='Vcl1_zi - Vdl1',
                          info='Voltage deviation indicator for (ul2, ul1)',
                          tex_name='zs_{Vdl1}',
                          )

        self.Vcl2 = Limiter(u=self.v,
                            lower=self.ul3,
                            upper=self.ul2,
                            tex_name=r'V_{cl1}',
                            info='Voltage comparer for (ul3, ul2)',
                            equal=False,
                            )
        self.Vdl2 = Algeb(v_str='0',
                          e_str='Vcl2_zi - Vdl2',
                          info='Voltage deviation indicator for (ul3, ul2)',
                          tex_name='zs_{Vdl1}',
                          )

        self.Vcl3 = Limiter(u=self.v,
                            lower=self.ul4,
                            upper=self.ul3,
                            tex_name=r'V_{cl3}',
                            info='Voltage comparer for (ul4, ul3)',
                            equal=False,
                            )
        self.Vdl3 = Algeb(v_str='0',
                          e_str='Vcl3_zi - Vdl3',
                          info='Voltage deviation indicator for (ul4, ul3)',
                          tex_name='zs_{Vdl3}',
                          )

        self.Vcu1 = Limiter(u=self.v,
                            lower=self.uu1,
                            upper=self.uu2,
                            tex_name=r'V_{cu1}',
                            info='Voltage comparer for (uu1, uu2)',
                            equal=False,
                            )
        self.Vdu1 = Algeb(v_str='0',
                          e_str='Vcu1_zi - Vdu1',
                          info='Voltage deviation indicator for (uu1, uu2)',
                          tex_name='zs_{Vdl1}',
                          )

        self.Vcu2 = Limiter(u=self.v,
                            lower=self.uu2,
                            upper=self.uu3,
                            tex_name=r'V_{cu2}',
                            info='Voltage comparer for (uu2, uu3)',
                            equal=False,
                            )
        self.Vdu2 = Algeb(v_str='0',
                          e_str='Vcu2_zi - Vdu2',
                          info='Voltage deviation indicator for (uu2, uu3)',
                          tex_name='zs_{Vdu2}',
                          )

        # Delayed voltage deviation indicator
        self.Vdl1d = Delay(u=self.Vdl1,
                           mode='time',
                           delay=self.Tvl1.v)
        self.Vdl2d = Delay(u=self.Vdl2,
                           mode='time',
                           delay=self.Tvl2.v)
        self.Vdl3d = Delay(u=self.Vdl3,
                           mode='time',
                           delay=self.Tvl3.v)
        self.Vdu1d = Delay(u=self.Vdu1,
                           mode='time',
                           delay=self.Tvu1.v)
        self.Vdu2d = Delay(u=self.Vdu2,
                           mode='time',
                           delay=self.Tvu2.v)

        # -- Lock PVD1 output power

        self.ltu = ConstService(v_str='1')
        self.ltl = ConstService(v_str='0')

        # frequency
        self.fp = Algeb(v_str='0',
                        info='Frequency protection signal',
                        tex_name=r'fp',
                        )
        self.fp.e_str = 'fena * (fdl1d_v * fdl1 + fdl2d_v * fdl2 + \
                                 fdu1d_v * fdu1 + fdu2d_v * fdu2) - fp'
        self.fphl = HardLimiter(u=self.fp,
                                lower=self.ltl,
                                upper=self.ltu,
                                tex_name=r'fp_{hl}',
                                )
        # voltage
        self.Vp = Algeb(v_str='0',
                        info='Voltage protection signal',
                        tex_name=r'Vp',
                        )
        self.Vp.e_str = 'Vena * (Vdl1d_v * Vdl1 + Vdl2d_v * Vdl2 + \
                                 Vdl3d_v * Vdl3 + Vdu1d_v * Vdu1 + \
                                 Vdu2d_v * Vdu2) - Vp'
        self.Vphl = HardLimiter(u=self.Vp,
                                lower=self.ltu,
                                upper=self.ltl,
                                tex_name=r'Vp_{hl}',
                                )

        # lock actvie and reactive power output
        self.pplk = ExtAlgeb(model='DG',
                             src='Psum',
                             indexer=self.dev,
                             export=False,
                             e_str='-1000 * pplk * fp * Vp',
                             info='Active power locker',
                             )
        self.qplk = ExtAlgeb(model='DG',
                             src='Qsum',
                             indexer=self.dev,
                             export=False,
                             e_str='-1000 * qplk * fp * Vp',
                             info='Reactive power locker',
                             )


class PLK2(PLK2Data, PLK2Model):
    """
    DER protection model type 2. PLK stands for Power Lock.

    Derived from PLK, followed IEEE-1547.

    Target device (limited to DG group) ``Psum`` and ``Qsum`` will drop to zero
    immediately when frequency/voltage protection is triggered. Once the lock is
    released, ``Psum`` and ``Qsum`` will return to normal immediately.

    ``fena`` and ``Vena`` are protection enabling parameters. 1 is on and 0 is off.

    Frequency (Hz):\n
    (fl3, fl2), Tfl2; [(50.0, 57.5), 10s]\n
    (fl2, fl1), Tfl1; [(57.5, 59.2), 300s]\n
    (fu1, fu2), Tfu1; [(60.5, 61.5), 300s]\n
    (fu2, fu3), Tfu2; [(61.5, 70.0), 10s]\n

    Voltage (p.u.):\n
    (ul4, ul3), Tvl3; [(0.10, 0.45), 0.16s]\n
    (ul3, ul2), Tvl2; [(0.45, 0.60), 1s]\n
    (ul2, ul1), Tvl1; [(0.60, 0.88), 2s]\n
    (uu1, uu2), Tvu1; [(1.10, 1.20), 1s]\n
    (uu2, uu3), Tvu2; [(1.20, 2.00), 0.16s]\n
    """

    def __init__(self, system, config):
        PLK2Data.__init__(self)
        PLK2Model.__init__(self, system, config)

