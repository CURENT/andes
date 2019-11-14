import unittest
from andes.core.var import VarBase, Algeb, State, Calc, ExtVar, ExtState, ExtAlgeb  # NOQA


class TestVar(unittest.TestCase):
    def setUp(self):
        self.v = VarBase(name='v',
                         tex_name='V',
                         info='test_variable',
                         unit='na',
                         v_setter=False,
                         e_setter=False)

    # def test_
