import logging

from andes.core.model import Model, ModelData  # NOQA
from andes.core.param import IdxParam, DataParam, NumParam  # NOQA
from andes.core.var import Algeb, State, ExtAlgeb  # NOQA
from andes.core.service import ConstService  # NOQA
logger = logging.getLogger(__name__)


class LineData(ModelData):
    def __init__(self):
        super().__init__()

        self.bus1 = IdxParam(model='Bus', info="idx of from bus")
        self.bus2 = IdxParam(model='Bus', info="idx of to bus")

        self.Sn = NumParam(default=100.0,
                           info="Power rating",
                           non_zero=True,
                           tex_name=r'S_n',
                           unit='MW',
                           )
        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f',
                           unit='Hz',
                           )
        self.Vn1 = NumParam(default=110.0,
                            info="AC voltage rating",
                            non_zero=True,
                            tex_name=r'V_{n1}',
                            unit='kV',
                            )
        self.Vn2 = NumParam(default=110.0,
                            info="rated voltage of bus2",
                            non_zero=True,
                            tex_name=r'V_{n2}',
                            unit='kV',
                            )

        self.r = NumParam(default=1e-8,
                          info="line resistance",
                          tex_name='r',
                          z=True,
                          unit='p.u.',
                          )
        self.x = NumParam(default=1e-8,
                          info="line reactance",
                          tex_name='x',
                          z=True,
                          unit='p.u.',
                          )
        self.b = NumParam(default=0.0,
                          info="shared shunt susceptance",
                          y=True,
                          unit='p.u.',
                          )
        self.g = NumParam(default=0.0,
                          info="shared shunt conductance",
                          y=True,
                          unit='p.u.',
                          )
        self.b1 = NumParam(default=0.0,
                           info="from-side susceptance",
                           tex_name='b_1',
                           unit='p.u.',
                           )
        self.g1 = NumParam(default=0.0,
                           info="from-side conductance",
                           tex_name='g_1',
                           unit='p.u.',
                           )
        self.b2 = NumParam(default=0.0,
                           info="to-side susceptance",
                           tex_name='b_2',
                           unit='p.u.',
                           )
        self.g2 = NumParam(default=0.0,
                           info="to-side conductance",
                           tex_name='g_2',
                           unit='p.u.',
                           )

        self.trans = NumParam(default=0,
                              info="transformer branch flag",
                              unit='bool',
                              )
        self.tap = NumParam(default=1.0,
                            info="transformer branch tap ratio",
                            tex_name='t_{ap}',
                            non_negative=True,
                            unit='float',
                            )
        self.phi = NumParam(default=0.0,
                            info="transformer branch phase shift in rad",
                            tex_name=r'\phi',
                            unit='radian',
                            )

        self.owner = IdxParam(model='Owner', info="owner code")

        self.xcoord = DataParam(info="x coordinates")
        self.ycoord = DataParam(info="y coordinates")


class Line(LineData, Model):
    """
    AC transmission line model.

    To reduce the number of variables, line injections are summed at bus equations
    and are not stored. Current injections are not computed.
    """

    def __init__(self, system=None, config=None):
        LineData.__init__(self)
        Model.__init__(self, system, config)
        self.group = 'ACLine'
        self.flags.pflow = True
        self.flags.tds = True

        self.a1 = ExtAlgeb(model='Bus', src='a', indexer=self.bus1, tex_name='a_1',
                           info='phase angle of the from bus')
        self.a2 = ExtAlgeb(model='Bus', src='a', indexer=self.bus2, tex_name='a_2',
                           info='phase angle of the to bus')
        self.v1 = ExtAlgeb(model='Bus', src='v', indexer=self.bus1, tex_name='v_1',
                           info='voltage magnitude of the from bus')
        self.v2 = ExtAlgeb(model='Bus', src='v', indexer=self.bus2, tex_name='v_2',
                           info='voltage magnitude of the to bus')

        self.gh = ConstService(tex_name='g_h')
        self.bh = ConstService(tex_name='b_h')
        self.gk = ConstService(tex_name='g_k')
        self.bk = ConstService(tex_name='b_k')

        self.yh = ConstService(tex_name='y_h', vtype=complex)
        self.yk = ConstService(tex_name='y_k', vtype=complex)
        self.yhk = ConstService(tex_name='y_{hk}', vtype=complex)

        self.ghk = ConstService(tex_name='g_{hk}')
        self.bhk = ConstService(tex_name='b_{hk}')

        self.itap = ConstService(tex_name='1/t_{ap}')
        self.itap2 = ConstService(tex_name='1/t_{ap}^2')

        self.gh.v_str = 'g1 + 0.5 * g'
        self.bh.v_str = 'b1 + 0.5 * b'
        self.gk.v_str = 'g2 + 0.5 * g'
        self.bk.v_str = 'b2 + 0.5 * b'

        self.yh.v_str = 'u * (gh + 1j * bh)'
        self.yk.v_str = 'u * (gk + 1j * bk)'
        self.yhk.v_str = 'u/((r+1e-8) + 1j*(x+1e-8))'

        self.ghk.v_str = 're(yhk)'
        self.bhk.v_str = 'im(yhk)'

        self.itap.v_str = '1/tap'
        self.itap2.v_str = '1/tap/tap'

        self.a1.e_str = 'u * (v1 ** 2 * (gh + ghk) * itap2  - \
                              v1 * v2 * (ghk * cos(a1 - a2 - phi) + \
                                         bhk * sin(a1 - a2 - phi)) * itap)'

        self.v1.e_str = 'u * (-v1 ** 2 * (bh + bhk) * itap2 - \
                              v1 * v2 * (ghk * sin(a1 - a2 - phi) - \
                                         bhk * cos(a1 - a2 - phi)) * itap)'

        self.a2.e_str = 'u * (v2 ** 2 * (gh + ghk) - \
                              v1 * v2 * (ghk * cos(a1 - a2 - phi) - \
                                         bhk * sin(a1 - a2 - phi)) * itap)'

        self.v2.e_str = 'u * (-v2 ** 2 * (bh + bhk) + \
                              v1 * v2 * (ghk * sin(a1 - a2 - phi) + \
                                         bhk * cos(a1 - a2 - phi)) * itap)'


class JumperData(ModelData):
    """
    Data for jumper that merges two buses into one.
    """
    def __init__(self):
        ModelData.__init__(self)

        self.bus1 = IdxParam(model='Bus', info="idx of from bus")
        self.bus2 = IdxParam(model='Bus', info="idx of to bus")


class JumperModel(Model):
    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.pflow = True
        self.flags.tds = True
        self.group = 'ACShort'

        self.a1 = ExtAlgeb(model='Bus', src='a', indexer=self.bus1, tex_name='a_1',
                           info='phase angle of the from bus')
        self.a2 = ExtAlgeb(model='Bus', src='a', indexer=self.bus2, tex_name='a_2',
                           info='phase angle of the to bus')
        self.v1 = ExtAlgeb(model='Bus', src='v', indexer=self.bus1, tex_name='v_1',
                           info='voltage magnitude of the from bus')
        self.v2 = ExtAlgeb(model='Bus', src='v', indexer=self.bus2, tex_name='v_2',
                           info='voltage magnitude of the to bus')

        self.p = Algeb(info='active power (1 to 2)',
                       e_str='u * (a1 - a2)',
                       tex_name='P',
                       diag_eps=True,
                       )

        self.q = Algeb(info='active power (1 to 2)',
                       e_str='u * (v1 - v2)',
                       tex_name='Q',
                       diag_eps=True,
                       )

        self.a1.e_str = 'p'
        self.a2.e_str = '-p'

        self.v1.e_str = 'q'
        self.v2.e_str = '-q'


class Jumper(JumperData, JumperModel):
    """
    Jumper is a device to short two buses (merging two buses into one).

    Jumper can connect two buses satisfying one of the following conditions:

    - neither bus is voltage-controlled
    - either bus is voltage-controlled
    - both buses are voltage-controlled, and the voltages are the same.

    If the buses are controlled in different voltages, power flow will
    not solve (as the power flow through the jumper will be infinite).

    In the solutions, the ``p`` and ``q`` are flowing out of bus1
    and flowing into bus2.
    """

    def __init__(self, system, config):
        JumperData.__init__(self)
        JumperModel.__init__(self, system, config)
