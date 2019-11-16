import logging
from andes.core.model import Model, ModelData, ModelConfig  # NOQA
from andes.core.param import DataParam, NumParam, ExtParam  # NOQA
from andes.core.var import Algeb, State, ExtAlgeb  # NOQA
from andes.core.limiter import Comparer, SortedLimiter  # NOQA
from andes.core.service import Service  # NOQa
logger = logging.getLogger(__name__)


class LineData(ModelData):
    def __init__(self):
        super().__init__()

        self.bus1 = DataParam(info="idx of from bus")
        self.bus2 = DataParam(info="idx of to bus")
        self.xcoord = DataParam(info="x coordinates")
        self.ycoord = DataParam(info="y coordinates")
        self.owner = DataParam(info="owner code")

        self.Sn = NumParam(default=100.0, info="Power rating", non_zero=True)
        self.fn = NumParam(default=60, info="rated frequency")
        self.Vn1 = NumParam(default=110.0, info="AC voltage rating", non_zero=True)
        self.Vn2 = NumParam(default=110.0, info="rated voltage of bus2", non_zero=True)

        self.r = NumParam(default=0, info="connection line resistance")
        self.x = NumParam(default=1e-8, info="connection line reactance")
        self.b = NumParam(default=1e-10, info="shared shunt susceptance")
        self.g = NumParam(default=0.0, info="shared shunt conductance")
        self.b1 = NumParam(default=0.0, info="from-side susceptance")
        self.g1 = NumParam(default=0.0, info="from-side conductance")
        self.b2 = NumParam(default=0.0, info="to-side susceptance")
        self.g2 = NumParam(default=0.0, info="to-side conductance")

        self.trans = NumParam(default=0, info="transformer branch flag")
        self.tap = NumParam(default=1.0, info="transformer branch tap ratio")
        self.phi = NumParam(default=0, info="transformer branch phase shift in rad")


class Line(LineData, Model):
    def __init__(self, system=None, name=None, config=None):
        LineData.__init__(self)
        Model.__init__(self, name, system, config)
        self.flags['pflow'] = True

        self.a1 = ExtAlgeb(model='Bus', src='a', indexer=self.bus1)
        self.a2 = ExtAlgeb(model='Bus', src='a', indexer=self.bus2)
        self.v1 = ExtAlgeb(model='Bus', src='v', indexer=self.bus1)
        self.v2 = ExtAlgeb(model='Bus', src='v', indexer=self.bus2)

        self.gh = Service()
        self.bh = Service()
        self.gk = Service()
        self.bk = Service()

        self.yh = Service()
        self.yk = Service()
        self.yhk = Service()

        self.ghk = Service()
        self.bhk = Service()

        self.gh.e_symbolic = 'g1 + 0.5 * g'
        self.bh.e_symbolic = 'b1 + 0.5 * b'
        self.gk.e_symbolic = 'g2 + 0.5 * g'
        self.bk.e_symbolic = 'b2 + 0.5 * b'

        self.yh.e_symbolic = 'u * (gh + 1j * bh)'
        self.yk.e_symbolic = 'u * (gk + 1j * bk)'
        self.yhk.e_symbolic = 'u / (r + 1j * x)'

        self.ghk.e_symbolic = 're(yhk)'
        self.bhk.e_symbolic = 'im(yhk)'

        self.a1.e_symbolic = 'v1 ** 2 * (gh + ghk / tap ** 2)  - \
                              v1 * v2 * (ghk * cos(a1 - a2 - phi) + \
                                         bhk * sin(a1 - a2 - phi)) / tap'

        self.v1.e_symbolic = '-v1 ** 2 * (bh + bhk / tap ** 2) - \
                              v1 * v2 * (ghk * sin(a1 - a2 - phi) - \
                                         bhk * cos(a1 - a2 - phi)) / tap'

        self.a2.e_symbolic = 'v2 ** 2 * ghk - \
                              v1 * v2 * (ghk * cos(a1 - a2 - phi) - \
                                         bhk * sin(a1 - a2 - phi)) / tap'

        self.v2.e_symbolic = '-v2 ** 2 * bhk + \
                              v1 * v2 * (ghk * sin(a1 - a2 - phi) + \
                                         bhk * cos(a1 - a2 - phi)) / tap'
