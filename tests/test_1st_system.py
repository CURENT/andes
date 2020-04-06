import unittest
import andes


class TestCodegen(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = andes.main.prepare(quick=True)

    def test_docs(self) -> None:
        out = ''
        for group in self.ss.groups.values():
            out += group.doc_all()
