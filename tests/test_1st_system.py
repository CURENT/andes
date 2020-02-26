import unittest
from andes.system import System


class TestCodegen(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = System()
        system = self.ss
        system.prepare()

    def test_docs(self):
        out = ''
        for group in self.ss.groups.values():
            out += group.doc_all(export='rest')
