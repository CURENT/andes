import unittest

import andes


class TestCodegen(unittest.TestCase):
    """
    Test code generation.
    """

    def test_1_docs(self) -> None:
        ss = andes.main.prepare(quick=True)
        out = ''
        for group in ss.groups.values():
            out += group.doc_all()
