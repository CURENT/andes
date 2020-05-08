import unittest
import andes


class TestCodegen(unittest.TestCase):
    def test_1_docs(self) -> None:
        self.ss = andes.main.prepare(quick=True)
        out = ''
        for group in self.ss.groups.values():
            out += group.doc_all()

    def test_codegen_inc(self):
        self.ss = andes.main.prepare(quick=True, incremental=True)
