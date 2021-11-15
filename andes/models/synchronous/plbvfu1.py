"""
V-f playback generator model.
"""

from andes.shared import np

from andes.core import (Model, ModelData, IdxParam, NumParam, DataParam, ExtParam,
                        State, ExtAlgeb, ExtService, ConstService)


class PLBVFU1Data(ModelData):
    """
    Data for PLBVFU1 model.
    """

    def __init__(self):
        ModelData.__init__(self)
        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            model='StaticGen',
                            mandatory=True,
                            )
        self.ra = NumParam(info='armature resistance',
                           default=0.0,
                           )
        self.xs = NumParam(info='generator transient reactance',
                           default=0.2,
                           )
        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f',
                           )
        self.Vflag = NumParam(default=1.0,
                              info='playback voltage signal',
                              vrange=(0, 1),
                              unit='bool')
        self.fflag = NumParam(default=1.0,
                              info='playback frequency signal',
                              vrange=(0, 1),
                              unit='bool')
        self.filename = DataParam(default='',
                                  info='playback file name',
                                  mandatory=True,
                                  unit='string')
        self.Vscale = NumParam(default=1.0,
                               info='playback voltage scale',
                               non_negative=True,
                               unit='pu')
        self.fscale = NumParam(default=1.0,
                               info='playback frequency scale',
                               non_negative=True,
                               unit='pu')
        self.Tv = NumParam(default=0.2,
                           info='filtering time constant for voltage',
                           non_negative=True,
                           unit='s')
        self.Tf = NumParam(default=0.2,
                           info='filtering time constant for frequency',
                           non_negative=True,
                           unit='s',
                           )


class PLBVFU1Model(Model):
    """
    Model implementation of PLBVFU1.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.flags.tds = True
        self.Vn = ExtParam(model='Bus', src='Vn',
                           indexer=self.bus,
                           )

        self.zs = ConstService('ra + 1j * xs', vtype=np.complex,
                               info='impedance',
                               )
        self.zs2n = ConstService('ra * ra - xs * xs',
                                 info='ra^2 - xs^2',
                                 )

        # get power flow solutions

        self.p = ExtService(model='StaticGen', src='p',
                            indexer=self.gen,
                            )
        self.q = ExtService(model='StaticGen', src='p',
                            indexer=self.gen,
                            )
        self.Ec = ConstService('v * exp(1j * a) -'
                               'conj((p + 1j * q) / (v * exp(1j * a))) * (ra + 1j * xs)',
                               vtype=np.complex,
                               )

        self.E0 = ConstService('abs(Ec)')
        self.delta0 = ConstService('arg(Ec)')

        # Note: `Vts` and `fts` are assigned by TimeSeries before initializing this model.
        self.Vts = ConstService()
        self.fts = ConstService()

        self.ifscale = ConstService('1/fscale')
        self.iVscale = ConstService('1/Vscale')

        self.foffs = ConstService('fts * ifscale - 1')
        self.Voffs = ConstService('Vts * iVscale - E0')

        self.Vflt = State(info='filtered voltage',
                          t_const=self.Tv,
                          v_str='(iVscale * Vts - Voffs)',
                          e_str='(iVscale * Vts - Voffs) - Vflt',
                          unit='pu',
                          )

        self.fflt = State(info='filtered frequency',
                          t_const=self.Tf,
                          v_str='fts * ifscale - foffs',
                          e_str='(ifscale * fts - foffs) - fflt',
                          unit='pu',
                          )

        self.delta = State(info='rotor angle',
                           unit='rad',
                           v_str='delta0',
                           tex_name=r'\delta',
                           e_str='u * (2 * pi * fn) * (fflt - 1)')

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          e_str='Vflt*ra*(Vflt - v*cos(a - delta))/(ra**2 + xs**2) - '
                                'Vflt*v*xs*sin(a - delta)/(ra**2 + xs**2)',
                          ename='P',
                          tex_ename='P',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          ename='Q',
                          e_str='Vflt*ra*v*sin(a - delta)/(ra**2 + xs**2) + '
                                'Vflt*xs*(Vflt - v*cos(a - delta))/(ra**2 + xs**2)',
                          tex_ename='Q',
                          )


class PLBVFU1(PLBVFU1Model, PLBVFU1Data):
    """
    PLBVFU1 model: playback of voltage and frequency as a generator.
    """

    def __init__(self, system, config):
        PLBVFU1Data.__init__(self)
        PLBVFU1Model.__init__(self, system, config)

    def v_numeric(self, **kwargs):
        """
        Numeric initialization to disable corresponding ``StaticGen``.
        """

        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)
