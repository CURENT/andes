import unittest

import numpy as np  # NOQA
from andes.system import SystemNew
import dill

dill.settings['recurse'] = True


class TestSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.ss = SystemNew()
        system = self.ss
        system.prepare()

    def test_docs(self):
        out = ''
        for group in self.ss.groups.values():
            out += group.doc_all(export='rest')
