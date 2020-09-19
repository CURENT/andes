import logging
from andes.core.model import Model, ModelData  # NOQA
from andes.core.param import IdxParam, DataParam, NumParam  # NOQA
from andes.core.var import Algeb, State, ExtAlgeb  # NOQA
from andes.core.service import ConstService  # NOQA
import numpy as np
logger = logging.getLogger(__name__)


class LineData(ModelData):
    def __init__(self):
        super().__init__()

        self.bus1 = IdxParam(model='Bus', info="idx of from bus")
        self.bus2 = IdxParam(model='Bus', info="idx of to bus")

        self.Sn = NumParam(default=100.0, info="Power rating", non_zero=True, tex_name=r'S_n')
        self.fn = NumParam(default=60.0, info="rated frequency", tex_name=r'f')
        self.Vn1 = NumParam(default=110.0, info="AC voltage rating", non_zero=True, tex_name=r'V_{n1}')
        self.Vn2 = NumParam(default=110.0, info="rated voltage of bus2", non_zero=True, tex_name=r'V_{n2}')

        self.r = NumParam(default=1e-8, info="line resistance", tex_name='r', z=True)
        self.x = NumParam(default=1e-8, info="line reactance", tex_name='x', z=True)
        self.b = NumParam(default=0.0, info="shared shunt susceptance", y=True)
        self.g = NumParam(default=0.0, info="shared shunt conductance", y=True)
        self.b1 = NumParam(default=0.0, info="from-side susceptance", tex_name='b_1')
        self.g1 = NumParam(default=0.0, info="from-side conductance", tex_name='g_1')
        self.b2 = NumParam(default=0.0, info="to-side susceptance", tex_name='b_2')
        self.g2 = NumParam(default=0.0, info="to-side conductance", tex_name='g_2')

        self.trans = NumParam(default=0, info="transformer branch flag")
        self.tap = NumParam(default=1.0, info="transformer branch tap ratio", tex_name='t_{ap}')
        self.phi = NumParam(default=0.0, info="transformer branch phase shift in rad", tex_name=r'\phi')

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

        self.yh = ConstService(tex_name='y_h', vtype=np.complex)
        self.yk = ConstService(tex_name='y_k', vtype=np.complex)
        self.yhk = ConstService(tex_name='y_{hk}', vtype=np.complex)

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
