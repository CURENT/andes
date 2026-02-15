"""
Tests for connectivity analysis (island detection, slack coverage).

Status propagation tests are in ``test_status.py``.
"""

import unittest

import andes


class TestConnMan(unittest.TestCase):
    """
    Test connectivity analysis using ieee14_conn case (Bus 15 offline).
    """

    def setUp(self) -> None:
        self.ss = andes.load(andes.get_case("ieee14/ieee14_conn.xlsx"),
                             setup=True, default_config=True, no_output=True)

    def test_offline_bus_detected(self):
        """Bus 15 should be offline in the data."""
        self.assertEqual(self.ss.Bus.get(src='u', attr='v', idx=15), 0)

    def test_dependent_devices_ue_zero(self):
        """Devices on offline bus should have ue=0 after setup."""
        ss = self.ss

        # Line connected to offline bus should have ue=0
        line_uid = ss.Line.idx2uid('Line_21')
        self.assertEqual(ss.Line.ue.v[line_uid], 0)

        # StaticGen on offline bus should have ue=0
        pv_mdl = ss.StaticGen.idx2model(6)
        pv_uid = pv_mdl.idx2uid(6)
        self.assertEqual(pv_mdl.ue.v[pv_uid], 0)

        # StaticLoad on offline bus should have ue=0
        pq_mdl = ss.StaticLoad.idx2model('PQ_12')
        pq_uid = pq_mdl.idx2uid('PQ_12')
        self.assertEqual(pq_mdl.ue.v[pq_uid], 0)

    def test_turn_off_after_pflow(self):
        """Turning off a bus after PFlow should propagate ue to dependents."""
        ss = andes.load(andes.get_case('ieee14/ieee14_conn.xlsx'),
                        setup=False, no_output=True, default_config=True)
        ss.Bus.set(src='u', attr='v', idx=15, value=1)
        ss.setup()

        ss.PFlow.run()
        self.assertTrue(ss.PFlow.converged)

        # turn off a bus — ue should propagate
        ss.Bus.set_status(15, 0)
        self.assertEqual(ss.Bus.ue.v[ss.Bus.idx2uid(15)], 0)

    def test_connectivity_check_runs(self):
        """Connectivity check should run without error."""
        ss = self.ss
        ss.conn.check_connectivity(info=False)

        # Should have island info populated
        self.assertIsInstance(ss.Bus.island_sets, list)
        self.assertIsInstance(ss.Bus.islands, list)
