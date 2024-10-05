"""
Tests for PSS/E parser.
"""

import os
import unittest
import yaml

from andes.system import System
from andes.io import psse

import andes


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

    def test_3wxfr(self):
        """
        Test three winding transformer parsing.
        """

        ss = andes.load(andes.get_case('wscc9/wscc9_3wxfr.raw'),
                        setup=True)

        hv_bus = 4  # high voltage bus
        area_hv = ss.Bus.get(src='area', attr='v', idx=hv_bus)
        zone_hv = ss.Bus.get(src='zone', attr='v', idx=hv_bus)
        owner_hv = ss.Bus.get(src='owner', attr='v', idx=hv_bus)

        star_bus = 10  # created star bus
        area_star = ss.Bus.get(src='area', attr='v', idx=star_bus)
        zone_hv = ss.Bus.get(src='zone', attr='v', idx=star_bus)
        owner_hv = ss.Bus.get(src='owner', attr='v', idx=star_bus)

        self.assertEqual(area_hv, area_star)
        self.assertEqual(zone_hv, zone_hv)
        self.assertEqual(owner_hv, owner_hv)
