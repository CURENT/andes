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
