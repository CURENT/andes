"""
Tests for device status propagation framework.

Tests the ``set_status`` / ``get_status`` API and recursive propagation
via BackRef from parent models to child controllers.
"""

import unittest

import andes


class TestStatusPropagation(unittest.TestCase):
    """
    Tests using kundur_st2cut case which has the full chain:
    GENROU -> EXDC2 (Exciter) -> ST2CUT (PSS)
    GENROU -> TGOV1 (TurbineGov)
    """

    def setUp(self):
        # Use routine='tds' so TDS-only models (exciters, governors, PSS)
        # get initialized and their ue ConstService is evaluated.
        self.ss = andes.run(
            andes.get_case('kundur/kundur_st2cut.xlsx'),
            default_config=True,
            no_output=True,
            routine='tds',
        )

    def test_status_graph_built(self):
        """Status graph should be populated during setup."""
        ss = self.ss
        self.assertIn('SynGen', ss._status_children)
        self.assertIn('Exciter', ss._status_children)
        self.assertIn('Exciter', ss._status_children['SynGen'])
        self.assertIn('TurbineGov', ss._status_children['SynGen'])
        self.assertIn('PSS', ss._status_children['Exciter'])

    def test_status_parent_map(self):
        """Status parent map should record child -> parent relationships."""
        ss = self.ss
        self.assertIn('EXDC2', ss._status_parent_map)
        self.assertIn('TGOV1', ss._status_parent_map)
        self.assertIn('ST2CUT', ss._status_parent_map)

        self.assertEqual(ss._status_parent_map['EXDC2'], [('syn', 'SynGen')])
        self.assertEqual(ss._status_parent_map['TGOV1'], [('syn', 'SynGen')])
        self.assertEqual(ss._status_parent_map['ST2CUT'], [('avr', 'Exciter')])

    def test_set_status_generator_off(self):
        """Turning off a generator should propagate to exciter, governor, and PSS."""
        ss = self.ss

        # Generator 1 has EXDC2_1 (exciter), TGOV1_1 (governor)
        # EXDC2_1 has ST2CUT_1 (PSS)
        ss.set_status('GENROU', 1, 0)

        # Generator's own u should be 0
        self.assertEqual(ss.GENROU.u.v[0], 0)

        # Exciter's u should be unchanged (own status preserved)
        self.assertEqual(ss.EXDC2.u.v[0], 1)
        # Exciter's ue should be 0 (parent offline)
        self.assertEqual(ss.EXDC2.ue.v[0], 0)

        # Governor's u should be unchanged
        self.assertEqual(ss.TGOV1.u.v[0], 1)
        # Governor's ue should be 0
        self.assertEqual(ss.TGOV1.ue.v[0], 0)

        # PSS's u should be unchanged
        self.assertEqual(ss.ST2CUT.u.v[0], 1)
        # PSS's ue should be 0 (grandparent offline, via exciter)
        self.assertEqual(ss.ST2CUT.ue.v[0], 0)

    def test_set_status_generator_on(self):
        """Turning a generator back on should restore children's ue."""
        ss = self.ss

        ss.set_status('GENROU', 1, 0)
        ss.set_status('GENROU', 1, 1)

        # All ue should be restored to 1 (all devices have u=1)
        self.assertEqual(ss.EXDC2.ue.v[0], 1)
        self.assertEqual(ss.TGOV1.ue.v[0], 1)
        self.assertEqual(ss.ST2CUT.ue.v[0], 1)

    def test_set_status_no_cross_contamination(self):
        """Turning off generator 1 should not affect generator 2's controllers."""
        ss = self.ss

        ss.set_status('GENROU', 1, 0)

        # Generator 2's exciter and governor should be unaffected
        # First set them to known good state
        ss.set_status('GENROU', 2, 1)
        self.assertEqual(ss.EXDC2.ue.v[1], 1)
        self.assertEqual(ss.TGOV1.ue.v[1], 1)

    def test_set_status_exciter_off(self):
        """Turning off an exciter should propagate to PSS but not governor."""
        ss = self.ss

        # First ensure everything is online
        ss.set_status('GENROU', 1, 1)

        # Turn off exciter 1 only
        ss.set_status('EXDC2', 1, 0)

        # Exciter's u should be 0, ue should be 0
        self.assertEqual(ss.EXDC2.u.v[0], 0)
        self.assertEqual(ss.EXDC2.ue.v[0], 0)

        # PSS should be affected (exciter is parent)
        self.assertEqual(ss.ST2CUT.u.v[0], 1)
        self.assertEqual(ss.ST2CUT.ue.v[0], 0)

        # Governor should NOT be affected (different branch)
        self.assertEqual(ss.TGOV1.ue.v[0], 1)

    def test_set_status_child_independently_off(self):
        """
        If a child is independently offline (u=0), turning parent off and
        back on should not change the child's u.
        """
        ss = self.ss

        # First ensure everything is online
        ss.set_status('GENROU', 1, 1)

        # Turn off exciter independently
        ss.set_status('EXDC2', 1, 0)
        self.assertEqual(ss.EXDC2.u.v[0], 0)

        # Turn off generator
        ss.set_status('GENROU', 1, 0)

        # Turn generator back on
        ss.set_status('GENROU', 1, 1)

        # Exciter's u should still be 0 (set_status doesn't modify children's u)
        self.assertEqual(ss.EXDC2.u.v[0], 0)
        # Exciter's ue should be 0 (own u is 0)
        self.assertEqual(ss.EXDC2.ue.v[0], 0)
        # PSS's ue should be 0 (exciter is off)
        self.assertEqual(ss.ST2CUT.ue.v[0], 0)

    def test_get_status_returns_ue(self):
        """get_status should return ue for models with ue, u for models without."""
        ss = self.ss

        # Ensure all online
        ss.set_status('GENROU', 1, 1)

        # GENROU has no ue ConstService -> returns u.v
        self.assertEqual(ss.get_status('GENROU', 1), 1)

        # EXDC2 has ue -> returns ue.v
        self.assertEqual(ss.get_status('EXDC2', 1), 1)

        # Turn off generator
        ss.set_status('GENROU', 1, 0)

        # GENROU get_status returns u.v = 0
        self.assertEqual(ss.get_status('GENROU', 1), 0)
        # EXDC2 get_status returns ue.v = 0 (parent off)
        self.assertEqual(ss.get_status('EXDC2', 1), 0)

    def test_get_status_group_level(self):
        """get_status should work via group name."""
        ss = self.ss

        ss.set_status('GENROU', 1, 1)
        self.assertEqual(ss.get_status('SynGen', 1), 1)
        self.assertEqual(ss.get_status('Exciter', 1), 1)

    def test_model_level_api(self):
        """set_status and get_status should work via model instances."""
        ss = self.ss

        ss.GENROU.set_status(1, 1)
        self.assertEqual(ss.GENROU.get_status(1), 1)
        self.assertEqual(ss.EXDC2.get_status(1), 1)

        ss.GENROU.set_status(1, 0)
        self.assertEqual(ss.GENROU.get_status(1), 0)
        self.assertEqual(ss.EXDC2.get_status(1), 0)

    def test_group_level_api(self):
        """set_status and get_status should work via group instances."""
        ss = self.ss

        ss.SynGen.set_status(1, 1)
        self.assertEqual(ss.SynGen.get_status(1), 1)

        ss.SynGen.set_status(1, 0)
        self.assertEqual(ss.SynGen.get_status(1), 0)
        self.assertEqual(ss.Exciter.get_status(1), 0)


class TestInitPropagation(unittest.TestCase):
    """
    Tests for init-time status propagation via ``propagate_init_status()``.

    Uses the same kundur_st2cut case (GENROU -> EXDC2 -> ST2CUT, GENROU -> TGOV1).
    """

    def setUp(self):
        self.ss = andes.run(
            andes.get_case('kundur/kundur_st2cut.xlsx'),
            default_config=True,
            no_output=True,
            routine='tds',
        )

    def test_all_online_after_init(self):
        """After normal init with all devices online, ue should equal u."""
        ss = self.ss
        for uid in range(ss.GENROU.n):
            self.assertEqual(ss.GENROU.ue.v[uid], ss.GENROU.u.v[uid])
        for uid in range(ss.EXDC2.n):
            self.assertEqual(ss.EXDC2.ue.v[uid], ss.EXDC2.u.v[uid])
        for uid in range(ss.TGOV1.n):
            self.assertEqual(ss.TGOV1.ue.v[uid], ss.TGOV1.u.v[uid])
        for uid in range(ss.ST2CUT.n):
            self.assertEqual(ss.ST2CUT.ue.v[uid], ss.ST2CUT.u.v[uid])

    def test_propagate_offline_generator(self):
        """propagate_init_status should set children ue=0 for an offline generator."""
        ss = self.ss

        # Simulate generator offline at init (u=0 from input data -> ue=0 after service eval)
        ss.GENROU.u.v[0] = 0
        ss.GENROU.ue.v[0] = 0

        ss.propagate_init_status()

        # Children's ue should be 0
        self.assertEqual(ss.EXDC2.ue.v[0], 0)
        self.assertEqual(ss.TGOV1.ue.v[0], 0)
        self.assertEqual(ss.ST2CUT.ue.v[0], 0)

        # Children's u should be unchanged
        self.assertEqual(ss.EXDC2.u.v[0], 1)
        self.assertEqual(ss.TGOV1.u.v[0], 1)
        self.assertEqual(ss.ST2CUT.u.v[0], 1)

    def test_propagate_offline_exciter(self):
        """propagate_init_status with offline exciter should cascade to PSS only."""
        ss = self.ss

        # Simulate exciter offline at init
        ss.EXDC2.u.v[0] = 0
        ss.EXDC2.ue.v[0] = 0

        ss.propagate_init_status()

        # PSS should have ue=0 (exciter is parent)
        self.assertEqual(ss.ST2CUT.ue.v[0], 0)
        # Governor should be unaffected (different branch)
        self.assertEqual(ss.TGOV1.ue.v[0], 1)

    def test_propagate_no_cross_contamination(self):
        """Offline generator 1 should not affect generator 2's children."""
        ss = self.ss

        ss.GENROU.u.v[0] = 0
        ss.GENROU.ue.v[0] = 0

        ss.propagate_init_status()

        # Generator 2's exciter and governor should remain online
        self.assertEqual(ss.EXDC2.ue.v[1], 1)
        self.assertEqual(ss.TGOV1.ue.v[1], 1)

    def test_propagate_child_also_offline(self):
        """If both generator and exciter are offline, PSS ue should be 0."""
        ss = self.ss

        ss.GENROU.u.v[0] = 0
        ss.GENROU.ue.v[0] = 0
        ss.EXDC2.u.v[0] = 0

        ss.propagate_init_status()

        # exciter ue = exciter.u * gen.ue = 0 * 0 = 0
        self.assertEqual(ss.EXDC2.ue.v[0], 0)
        # PSS ue = pss.u * exciter.ue = 1 * 0 = 0
        self.assertEqual(ss.ST2CUT.ue.v[0], 0)


class TestBusStatusPropagation(unittest.TestCase):
    """
    Test bus→device status propagation via the ACNode group.

    Uses kundur_st2cut which has Bus, Line, GENROU, PQ, PV, Shunt, etc.
    """

    def setUp(self):
        self.ss = andes.run(
            andes.get_case('kundur/kundur_st2cut.xlsx'),
            default_config=True,
            no_output=True,
            routine='tds',
        )

    def test_bus_offline_propagates_to_generators(self):
        """Turning off a bus should set ue=0 on generators at that bus."""
        ss = self.ss
        gen_uid = 0
        bus_idx = ss.GENROU.bus.v[gen_uid]

        ss.set_status('ACNode', bus_idx, 0)

        self.assertEqual(ss.GENROU.ue.v[gen_uid], 0)
        # u stays 1 (device's own status preserved)
        self.assertEqual(ss.GENROU.u.v[gen_uid], 1)

    def test_bus_offline_cascades_full_chain(self):
        """Bus offline should cascade: Bus → GENROU → EXDC2 → ST2CUT."""
        ss = self.ss
        gen_uid = 0
        bus_idx = ss.GENROU.bus.v[gen_uid]

        ss.set_status('ACNode', bus_idx, 0)

        self.assertEqual(ss.GENROU.ue.v[gen_uid], 0)
        self.assertEqual(ss.EXDC2.ue.v[0], 0)
        self.assertEqual(ss.TGOV1.ue.v[0], 0)
        self.assertEqual(ss.ST2CUT.ue.v[0], 0)

    def test_bus_online_restores_chain(self):
        """Turning bus back on should restore the full chain."""
        ss = self.ss
        gen_uid = 0
        bus_idx = ss.GENROU.bus.v[gen_uid]

        ss.set_status('ACNode', bus_idx, 0)
        ss.set_status('ACNode', bus_idx, 1)

        self.assertEqual(ss.GENROU.ue.v[gen_uid], 1)
        self.assertEqual(ss.EXDC2.ue.v[0], 1)
        self.assertEqual(ss.TGOV1.ue.v[0], 1)
        self.assertEqual(ss.ST2CUT.ue.v[0], 1)

    def test_bus_offline_child_independently_off(self):
        """If exciter has u=0, restoring bus should NOT restore exciter ue."""
        ss = self.ss
        gen_uid = 0
        bus_idx = ss.GENROU.bus.v[gen_uid]

        # Turn off exciter independently
        ss.set_status('EXDC2', 1, 0)
        self.assertEqual(ss.EXDC2.u.v[0], 0)

        # Turn off bus, then back on
        ss.set_status('ACNode', bus_idx, 0)
        ss.set_status('ACNode', bus_idx, 1)

        # Exciter should still be off (own u=0)
        self.assertEqual(ss.EXDC2.ue.v[0], 0)
        # PSS should also be off (parent exciter is off)
        self.assertEqual(ss.ST2CUT.ue.v[0], 0)
        # But governor should be restored
        self.assertEqual(ss.TGOV1.ue.v[0], 1)


class TestBusSetIntegration(unittest.TestCase):
    """
    Test that Bus.set(src='u') triggers set_status propagation.
    """

    def setUp(self):
        self.ss = andes.run(
            andes.get_case('kundur/kundur_st2cut.xlsx'),
            default_config=True,
            no_output=True,
            routine='tds',
        )

    def test_bus_set_propagates_ue(self):
        """Bus.set('u', ...) should warn and propagate ue to dependents."""
        ss = self.ss
        gen_uid = 0
        bus_idx = ss.GENROU.bus.v[gen_uid]

        with self.assertWarns(FutureWarning):
            ss.Bus.set('u', bus_idx, 0)

        self.assertEqual(ss.GENROU.ue.v[gen_uid], 0)
        self.assertEqual(ss.EXDC2.ue.v[0], 0)


class TestMultiParentLine(unittest.TestCase):
    """
    Test Line with two parent buses (bus1 and bus2).

    Line.ue = Line.u * bus1.ue * bus2.ue  (AND logic).
    """

    def setUp(self):
        self.ss = andes.run(
            andes.get_case('kundur/kundur_st2cut.xlsx'),
            default_config=True,
            no_output=True,
            routine='tds',
        )

    def test_line_bus1_offline(self):
        """Line with bus1 offline should have ue=0."""
        ss = self.ss
        line_uid = 0
        bus1_idx = ss.Line.bus1.v[line_uid]

        ss.set_status('ACNode', bus1_idx, 0)

        self.assertEqual(ss.Line.ue.v[line_uid], 0)
        self.assertEqual(ss.Line.u.v[line_uid], 1)

    def test_line_bus2_offline(self):
        """Line with bus2 offline should have ue=0."""
        ss = self.ss
        line_uid = 0
        bus2_idx = ss.Line.bus2.v[line_uid]

        ss.set_status('ACNode', bus2_idx, 0)

        self.assertEqual(ss.Line.ue.v[line_uid], 0)

    def test_line_both_buses_online(self):
        """Line ue=1 only when both buses are online."""
        ss = self.ss
        line_uid = 0
        bus1_idx = ss.Line.bus1.v[line_uid]
        bus2_idx = ss.Line.bus2.v[line_uid]

        ss.set_status('ACNode', bus1_idx, 0)
        ss.set_status('ACNode', bus2_idx, 0)
        self.assertEqual(ss.Line.ue.v[line_uid], 0)

        # Restore bus1 only — line should stay off
        ss.set_status('ACNode', bus1_idx, 1)
        self.assertEqual(ss.Line.ue.v[line_uid], 0)

        # Restore bus2 — now line should be on
        ss.set_status('ACNode', bus2_idx, 1)
        self.assertEqual(ss.Line.ue.v[line_uid], 1)


class TestToggleSetStatus(unittest.TestCase):
    """
    Test that Toggle uses set_status at runtime (not model.set).
    """

    def test_toggle_trips_line(self):
        """Toggle tripping a line should update both u and ue."""
        ss = andes.load(
            andes.get_case('kundur/kundur_full.xlsx'),
            default_config=True,
            no_output=True,
        )
        ss.setup()

        line_idx = ss.Line.idx.v[0]
        line_uid = 0

        # Simulate what Toggle._u_switch does
        ss.set_status('Line', line_idx, 0)

        self.assertEqual(ss.Line.u.v[line_uid], 0)
        self.assertEqual(ss.Line.ue.v[line_uid], 0)

    def test_toggle_restores_line(self):
        """Toggle reconnecting a line should restore u and ue."""
        ss = andes.load(
            andes.get_case('kundur/kundur_full.xlsx'),
            default_config=True,
            no_output=True,
        )
        ss.setup()

        line_idx = ss.Line.idx.v[0]
        line_uid = 0

        ss.set_status('Line', line_idx, 0)
        ss.set_status('Line', line_idx, 1)

        self.assertEqual(ss.Line.u.v[line_uid], 1)
        self.assertEqual(ss.Line.ue.v[line_uid], 1)


if __name__ == '__main__':
    unittest.main()
