import json
import os
import tempfile
import unittest

import numpy as np

import andes
from andes.utils.paths import get_case


class TestREGFMC1REPCGFMC1Measurements(unittest.TestCase):
    """
    Tests for REGFMC1/REPCGFMC1 measurement wiring.
    """

    @staticmethod
    def _write_case():
        with open(get_case('smib/SMIB.json')) as f:
            data = json.load(f)

        data['GENCLS'] = [row for row in data['GENCLS']
                          if row['idx'] != 'GENCLS_1']

        data['REGFMC1'] = [{
            'idx': 'REGFMC1_1',
            'u': 1.0,
            'name': 'REGFMC1_1',
            'bus': 1,
            'gen': 'PV_1',
            'FFFlag': 0.0,
        }]

        data['REPCGFMC1'] = [{
            'idx': 'REPCGFMC1_1',
            'u': 1.0,
            'name': 'REPCGFMC1_1',
            'reg': 'REGFMC1_1',
            'busr': 3,
            'line': 'Line_1',
            'Rloss': 0.0,
            'Xloss': 0.0,
            'FFRFlag': 0.0,
        }]

        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    def test_measurements_and_short_tds(self):
        path = self._write_case()

        try:
            ss = andes.load(path, no_output=True, default_config=True)
            ss.PFlow.run()
            self.assertTrue(ss.PFlow.converged)

            ss.TDS.config.tf = 0.02
            ss.TDS.config.tstep = 0.01
            ss.TDS.init()
            self.assertTrue(ss.TDS.initialized)

            reg = ss.REGFMC1
            rep = ss.REPCGFMC1

            inv_bus = reg.bus.v[0]
            site_bus = rep.busr.v[0]

            np.testing.assert_equal(
                rep.vinv.a[0],
                ss.Bus.v.a[ss.Bus.idx2uid(inv_bus)],
                err_msg='vinv must measure REGFMC1.bus')
            np.testing.assert_equal(
                rep.vsite.a[0],
                ss.Bus.v.a[ss.Bus.idx2uid(site_bus)],
                err_msg='vsite must measure REPCGFMC1.busr')
            np.testing.assert_equal(
                rep.v.a[0],
                ss.Bus.v.a[ss.Bus.idx2uid(site_bus)],
                err_msg='v must measure REPCGFMC1.busr')

            busfreq_uid = ss.BusFreq.idx2uid(rep.busfreq.v[0])
            self.assertEqual(ss.BusFreq.bus.v[busfreq_uid], site_bus)

            np.testing.assert_allclose(rep.Psite_raw.v[0],
                                       -rep.Pline_bus2.v[0],
                                       rtol=0.0, atol=1e-8)
            np.testing.assert_allclose(rep.Qsite_raw.v[0],
                                       -rep.Qline_bus2.v[0],
                                       rtol=0.0, atol=1e-8)
            self.assertGreater(rep.Psite_raw.v[0], 0.0)

            np.testing.assert_allclose(rep.Psite_y.v[0], rep.Psite_raw.v[0],
                                       rtol=0.0, atol=1e-8)
            np.testing.assert_allclose(rep.Qsite_y.v[0], rep.Qsite_raw.v[0],
                                       rtol=0.0, atol=1e-8)

            self.assertEqual(rep.Rloss.v[0], 0.0)
            self.assertEqual(rep.Xloss.v[0], 0.0)

            ss.TDS.run(no_summary=True)
            self.assertTrue(ss.TDS.converged)
            self.assertEqual(ss.exit_code, 0)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
