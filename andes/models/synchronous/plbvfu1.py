"""
V-f playback generator model.
"""

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
        self.Sn = NumParam(default=100.0,
                           info="Power rating",
                           tex_name='S_n',
                           unit='MVA',
                           )
        self.Vn = NumParam(default=110.0,
                           info="AC voltage rating",
                           tex_name='V_n',
                           )
        self.ra = NumParam(info='armature resistance',
                           default=0.0,
                           tex_name='r_a',
                           z=True,
                           )
        self.xs = NumParam(info='generator transient reactance',
                           default=0.2,
                           non_zero=True,
                           tex_name='x_s',
                           z=True,
                           )
        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f_n',
                           )
        self.Vflag = NumParam(default=1.0,
                              info='playback voltage signal',
                              vrange=(0, 1),
                              unit='bool',
                              )
        self.fflag = NumParam(default=1.0,
                              info='playback frequency signal',
                              vrange=(0, 1),
                              unit='bool',
                              )
        self.filename = DataParam(default='',
                                  info='playback file name',
                                  mandatory=True,
                                  unit='string')
        self.Vscale = NumParam(default=1.0,
                               info='playback voltage scale',
                               non_negative=True,
                               unit='pu',
                               tex_name='V_{scale}',
                               )
        self.fscale = NumParam(default=1.0,
                               info='playback frequency scale',
                               non_negative=True,
                               unit='pu',
                               tex_name='f_{scale}',
                               )
        self.Tv = NumParam(default=0.2,
                           info='filtering time constant for voltage',
                           non_negative=True,
                           unit='s',
                           tex_name='T_v',
                           )
        self.Tf = NumParam(default=0.2,
                           info='filtering time constant for frequency',
                           non_negative=True,
                           unit='s',
                           tex_name='T_f',
                           )


class PLBVFU1Model(Model):
    """
    Model implementation of PLBVFU1.

    The test case of this model require that ``TDS.config.reset_tiny = 1``.
    The cause needs investigating.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)

        self.group = 'SynGen'

        self.group_param_exception = ['Sn', 'M', 'D']
        self.group_var_exception = ['vd', 'vq', 'Id', 'Iq', 'tm', 'te', 'vf', 'XadIfd']
        self.flags.tds = True

        self.subidx = ExtParam(model='StaticGen',
                               src='subidx',
                               indexer=self.gen,
                               export=False,
                               info='Generator idx in plant; only used by PSS/E data'
                               )

        self.zs = ConstService('ra + 1j * xs', vtype=complex,
                               info='impedance',
                               )
        self.zs2n = ConstService('ra * ra - xs * xs',
                                 info='ra^2 - xs^2',
                                 )

        # get power flow solutions

        self.p = ExtService(model='StaticGen', src='p',
                            indexer=self.gen,
                            )
        self.q = ExtService(model='StaticGen', src='q',
                            indexer=self.gen,
                            )
        self.Ec = ConstService('v * exp(1j * a) +'
                               'conj((p + 1j * q) / (v * exp(1j * a))) * (ra + 1j * xs)',
                               vtype=complex,
                               tex_name='E_c',
                               )

        self.E0 = ConstService('abs(Ec)', tex_name='E_0')
        self.delta0 = ConstService('arg(Ec)', tex_name=r'\delta_0')

        # Note: `Vts` and `fts` are assigned by TimeSeries before initializing this model.
        self.Vts = ConstService()
        self.fts = ConstService()

        self.ifscale = ConstService('1/fscale', tex_name='1/f_{scale}')
        self.iVscale = ConstService('1/Vscale', tex_name='1/V_{scale}')

        self.foffs = ConstService('fts * ifscale - 1', tex_name='f_{offs}')
        self.Voffs = ConstService('Vts * iVscale - E0', tex_name='V_{offs}')

        self.Vflt = State(info='filtered voltage',
                          t_const=self.Tv,
                          v_str='(iVscale * Vts - Voffs)',
                          e_str='(iVscale * Vts - Voffs) - Vflt',
                          unit='pu',
                          tex_name='V_{flt}',
                          )

        self.omega = State(info='filtered frequency',
                           t_const=self.Tf,
                           v_str='fts * ifscale - foffs',
                           e_str='(ifscale * fts - foffs) - omega',
                           unit='pu',
                           tex_name=r'\omega',
                           )

        self.delta = State(info='rotor angle',
                           unit='rad',
                           v_str='delta0',
                           tex_name=r'\delta',
                           e_str='u * (2 * pi * fn) * (omega - 1)',
                           )

        # --- Power injections are obtained by sympy ---

        # >>> from sympy import symbols, sin, cos, conjugate
        # >>> Vflt, delta, v, a, ra, xs = symbols('Vflt delta v a ra xs', real=True)

        # >>> S = -v * (cos(a) + 1j*sin(a)) * \
        #         conjugate((Vflt * (cos(delta)+1j*sin(delta)) - v*(cos(a)+1j*sin(a))) / (ra+1j*xs))
        # >>> S.simplify().as_real_imag()

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage phase angle',
                          e_str='Vflt*v*xs*sin(a - delta)/(ra*ra + xs*xs) + '
                                'ra*v*(-Vflt*cos(a - delta) + v)/(ra*ra + xs*xs)',
                          ename='P',
                          tex_ename='P',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name=r'V',
                          info='Bus voltage magnitude',
                          ename='Q',
                          e_str='-Vflt*ra*v*sin(a - delta)/(ra*ra + xs*xs) + '
                                'v*xs*(-Vflt*cos(a - delta) + v)/(ra*ra + xs*xs)',
                          tex_ename='Q',
                          )


class PLBVFU1(PLBVFU1Model, PLBVFU1Data):
    """
    PLBVFU1 model: playback of voltage and frequency as a generator.

    The internal voltage and frequency are named ``Vflt`` and ``omega``.
    Rotor angle is named ``delta``.

    The current implementation relies on a ``TimeSeries`` device
    to provide the voltage and frequency signals.
    See ``ieee14_plbvfu1.xlsx`` and ``plbvf.xlsx`` in
    ``andes/cases/ieee14`` for an example.

    Voltage and frequeny data needs to be specified in per unit.
    Nominal values are not yet supported.
    """

    def __init__(self, system, config):
        PLBVFU1Data.__init__(self)
        PLBVFU1Model.__init__(self, system, config)

    def v_numeric(self, **kwargs):
        """
        Numeric initialization to disable corresponding ``StaticGen``.
        """

        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)
