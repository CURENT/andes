import unittest


from andes.models.base import NewModelBase, VarType, NumParam, VarBase  # NOQA
from andes.system import PowerSystem, SystemNew  # NOQA


class TestVariable(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_xxx(self):
        pass


class TestNewModelBase(unittest.TestCase):
    def setUp(self) -> None:
        self.system = SystemNew()
        self.new_model_base = NewModelBase(self.system, "NewModel")

    def test_XXX(self):
        pass
