"""
Tests for topology dirty flag mechanism.

Verifies that ``ConnMan._dirty`` is correctly set/cleared when topological
models (Line, Jumper, Slack) change status, and that non-topological
status changes do not trigger invalidation.
"""

import unittest

import andes


class TestTopoDirtyFlag(unittest.TestCase):
    """Basic dirty flag lifecycle tests using ieee14."""

    def setUp(self):
        self.ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            default_config=True,
            no_output=True,
        )

    def test_clean_after_setup(self):
        """Dirty flag should be False after setup (check_connectivity ran)."""
        self.assertFalse(self.ss.conn._dirty)

    def test_line_offline_sets_dirty(self):
        """Taking a Line offline should set dirty flag."""
        ss = self.ss
        ss.Line.set_status(ss.Line.idx.v[0], 0)
        self.assertTrue(ss.conn._dirty)

    def test_line_online_no_change_stays_clean(self):
        """Setting Line status to same value should NOT set dirty."""
        ss = self.ss
        # Line is already online (u=1, ue=1), setting to 1 is a no-op
        ss.Line.set_status(ss.Line.idx.v[0], 1)
        self.assertFalse(ss.conn._dirty)

    def test_check_connectivity_clears_dirty(self):
        """check_connectivity should clear dirty flag."""
        ss = self.ss
        ss.Line.set_status(ss.Line.idx.v[0], 0)
        self.assertTrue(ss.conn._dirty)

        ss.conn.check_connectivity(info=False)
        self.assertFalse(ss.conn._dirty)

    def test_invalidate_sets_dirty(self):
        """ConnMan.invalidate() should set dirty flag."""
        ss = self.ss
        self.assertFalse(ss.conn._dirty)
        ss.conn.invalidate()
        self.assertTrue(ss.conn._dirty)

    def test_slack_offline_sets_dirty(self):
        """Taking Slack offline should set dirty (affects slack coverage)."""
        ss = self.ss
        ss.Slack.set_status(ss.Slack.idx.v[0], 0)
        self.assertTrue(ss.conn._dirty)


class TestNonTopoNoInvalidation(unittest.TestCase):
    """Non-topological status changes should NOT set dirty."""

    def setUp(self):
        self.ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            default_config=True,
            no_output=True,
        )

    def test_pq_offline_stays_clean(self):
        """PQ load offline should not invalidate topology."""
        ss = self.ss
        ss.PQ.set_status(ss.PQ.idx.v[0], 0)
        self.assertFalse(ss.conn._dirty)

    def test_pv_offline_stays_clean(self):
        """PV generator offline should not invalidate topology."""
        ss = self.ss
        ss.PV.set_status(ss.PV.idx.v[0], 0)
        self.assertFalse(ss.conn._dirty)


class TestBusCascadeDirty(unittest.TestCase):
    """Bus status change should cascade dirty via topo children."""

    def setUp(self):
        self.ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            default_config=True,
            no_output=True,
        )

    def test_bus_offline_sets_dirty_via_line(self):
        """Bus offline should cascade to Line children, setting dirty."""
        ss = self.ss
        ss.Bus.set_status(ss.Bus.idx.v[0], 0)
        self.assertTrue(ss.conn._dirty)

    def test_bus_offline_and_restore(self):
        """Bus off then on should set dirty, check clears, second on is clean."""
        ss = self.ss
        bus_idx = ss.Bus.idx.v[0]

        ss.Bus.set_status(bus_idx, 0)
        self.assertTrue(ss.conn._dirty)

        ss.conn.check_connectivity(info=False)
        self.assertFalse(ss.conn._dirty)

        # Restore bus — Lines re-gain ue, dirty again
        ss.Bus.set_status(bus_idx, 1)
        self.assertTrue(ss.conn._dirty)


class TestTopoFlag(unittest.TestCase):
    """Verify topo flag is set on the right models."""

    def setUp(self):
        self.ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            default_config=True,
            no_output=True,
        )

    def test_line_has_topo(self):
        self.assertTrue(self.ss.Line.flags.topo)

    def test_slack_has_topo(self):
        self.assertTrue(self.ss.Slack.flags.topo)

    def test_pq_no_topo(self):
        self.assertFalse(self.ss.PQ.flags.topo)

    def test_pv_no_topo(self):
        self.assertFalse(self.ss.PV.flags.topo)

    def test_bus_no_topo(self):
        self.assertFalse(self.ss.Bus.flags.topo)


class TestPFlowNoRedundantCheck(unittest.TestCase):
    """PFlow should not re-run connectivity check."""

    def setUp(self):
        self.ss = andes.load(
            andes.get_case('ieee14/ieee14.json'),
            default_config=True,
            no_output=True,
        )

    def test_dirty_false_after_pflow(self):
        """After PFlow.run(), dirty should still be False (no redundant check)."""
        ss = self.ss
        self.assertFalse(ss.conn._dirty)
        ss.PFlow.run()
        self.assertTrue(ss.PFlow.converged)
        self.assertFalse(ss.conn._dirty)


class TestTDSTopoDirty(unittest.TestCase):
    """TDS should only re-check connectivity when topology actually changed."""

    def test_toggle_line_triggers_check(self):
        """Toggle tripping a Line should set dirty, TDS should re-check."""
        ss = andes.load(
            andes.get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()
        self.assertTrue(ss.PFlow.converged)

        ss.TDS.config.tf = 2.1  # Toggle fires at t=2
        ss.TDS.run()

        # After TDS completes with a Line toggle, dirty should be False
        # (connectivity was re-checked after the toggle)
        self.assertFalse(ss.conn._dirty)

    def test_no_event_stays_clean(self):
        """TDS without events should not trigger connectivity check."""
        ss = andes.load(
            andes.get_case('kundur/kundur_full.json'),
            default_config=True,
            no_output=True,
        )
        ss.PFlow.run()

        # Run TDS for a short time BEFORE the toggle event at t=2
        ss.TDS.config.tf = 0.5
        self.assertFalse(ss.conn._dirty)
        ss.TDS.run()
        self.assertFalse(ss.conn._dirty)


if __name__ == '__main__':
    unittest.main()
