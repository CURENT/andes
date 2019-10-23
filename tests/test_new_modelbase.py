import unittest


from andes.models.base import NewModelBase, VarType, Parameter, Variable  # NOQA
from andes.system import PowerSystem


class TestVariable(unittest.TestCase):
    def setUp(self) -> None:
        self.var = Variable("test", var_type=VarType.y)

    def test_xxx(self):
        pass


class TestNewModelBase(unittest.TestCase):
    def setUp(self) -> None:
        self.system = PowerSystem()
        self.new_model_base = NewModelBase(self.system, "NewModel")

    def test_XXX(self):
        pass
