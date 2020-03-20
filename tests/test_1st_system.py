import unittest
import andes


class TestCodegen(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = andes.main.prepare()

    def test_docs(self):
        out = ''
        for group in self.ss.groups.values():
            out += group.doc_all(export='rest')
