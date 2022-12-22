"""
Jumper model for connecting two buses without impedance.
"""

from andes.core import ModelData, IdxParam, Model, ExtAlgeb, Algeb


class JumperData(ModelData):
    """
    Data for jumper that merges two buses into one.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.bus1 = IdxParam(model='Bus', info="idx of from bus")
        self.bus2 = IdxParam(model='Bus', info="idx of to bus")


class JumperModel(Model):
    """
    Jumper model implementation.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self.flags.pflow = True
        self.flags.tds = True
        self.group = 'ACShort'

        self.a1 = ExtAlgeb(model='Bus', src='a', indexer=self.bus1, tex_name='a_1',
                           info='phase angle of the from bus',
                           ename='Pij',
                           tex_ename='P_{ij}',
                           )
        self.a2 = ExtAlgeb(model='Bus', src='a', indexer=self.bus2, tex_name='a_2',
                           info='phase angle of the to bus',
                           ename='Pji',
                           tex_ename='P_{ji}',
                           )
        self.v1 = ExtAlgeb(model='Bus', src='v', indexer=self.bus1, tex_name='v_1',
                           info='voltage magnitude of the from bus',
                           ename='Qij',
                           tex_ename='Q_{ij}',
                           )
        self.v2 = ExtAlgeb(model='Bus', src='v', indexer=self.bus2, tex_name='v_2',
                           info='voltage magnitude of the to bus',
                           ename='Qji',
                           tex_ename='Q_{ji}',
                           )

        self.p = Algeb(info='active power (1 to 2)',
                       e_str='u*(a1 - a2) + (1-u) * p',
                       tex_name='P',
                       diag_eps=True,
                       )

        self.q = Algeb(info='active power (1 to 2)',
                       e_str='u*(v1 - v2) + (1-u) * p',
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

    Setting a Jumper's connectivity status ``u`` to zero will disconnect the two
    buses. In the case of a system split, one will need to call
    ``System.connectivity()`` immediately following the split to detect islands.
    """

    def __init__(self, system, config):
        JumperData.__init__(self)
        JumperModel.__init__(self, system, config)
