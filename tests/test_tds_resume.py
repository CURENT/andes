import unittest
import andes


class TestTDSResume(unittest.TestCase):
    def test_tds_resume(self):
        case = andes.get_case('kundur/kundur_full.xlsx')

        ss = andes.run(case, routine='tds', tf=0.1, no_output=True,
                       default_config=True)

        ss.TDS.config.tf = 0.5
        ss.TDS.run()

        self.assertEqual(ss.exit_code, 0.0)

        # check if time stamps are correctly stored
        self.assertIn(0.1, ss.dae.ts._ys)
        self.assertIn(0.5, ss.dae.ts._ys)

        # check if values are correctly stored
        self.assertTrue(len(ss.dae.ts._ys[0.1]) > 0)
        self.assertTrue(len(ss.dae.ts._ys[0.5]) > 0)

        # check if `get_data` works
        data1 = ss.dae.ts.get_data((ss.GENROU.omega, ss.GENROU.v))
        data2 = ss.dae.ts.get_data((ss.GENROU.omega, ss.GENROU.v), a=[2])
        nt = len(ss.dae.ts.t)

        self.assertEqual(data1.shape[1], 8)
        self.assertEqual(data1.shape[0], nt)

        self.assertEqual(data2.shape[1], 2)
        self.assertEqual(data2.shape[0], nt)
