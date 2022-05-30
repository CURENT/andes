"""
Tests for configurations.
"""

import unittest

import andes


class TestConfigOption(unittest.TestCase):
    """
    Tests for config_option.
    """

    def test_config_option(self):
        """
        Test single and multiple config_option passed to `andes.run`.
        """

        path = andes.get_case("5bus/pjm5bus.json")
        self.assertRaises(ValueError, andes.load, path, config_option={"TDS = 1"})
        self.assertRaises(ValueError, andes.load, path, config_option={"System.TDS.any = 1"})
        self.assertRaises(ValueError, andes.load, path, config_option={"TDS.tf == 1"})

        ss = andes.load(path, config_option={"PQ.pq2z = 0"}, default_config=True)
        self.assertEqual(ss.PQ.config.pq2z, 0)

        ss = andes.load(path, config_option={"PQ.pq2z=0"}, default_config=True)
        self.assertEqual(ss.PQ.config.pq2z, 0)

        ss = andes.load(path, config_option=["PQ.pq2z=0", "TDS.tf = 1"], default_config=True)
        self.assertEqual(ss.PQ.config.pq2z, 0)
        self.assertEqual(ss.TDS.config.tf, 1)
