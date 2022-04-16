"""
Tests for PSS/E parser.
"""

import os
import unittest
import yaml

from andes.system import System
from andes.io import psse


class TestPSSEParser(unittest.TestCase):
    """
    Test PSSE parsers.
    """

    def test_sort_models(self):
        """
        Test sort_models.
        """

        system = System()

        dirname = os.path.dirname(__file__)
        with open(f'{dirname}/../andes/io/psse-dyr.yaml', 'r') as f:
            dyr_yaml = yaml.full_load(f)
        psse.sort_psse_models(dyr_yaml, system)
