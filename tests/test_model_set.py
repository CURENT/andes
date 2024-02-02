import unittest
import numpy as np

import andes


class TestModelMethods(unittest.TestCase):
    """
    Test methods of Model.
    """

    def test_model_set(self):
        """
        Test `Model.set()` method.
        """

        ss = andes.run(
            andes.get_case("ieee14/ieee14.json"),
            default_config=True,
            no_output=True,
        )
        ss.TDS.init()

        omega_addr = ss.GENROU.omega.a.tolist()

        # set a single value
        ss.GENROU.set("M", "GENROU_1", "v", 2.0)
        self.assertEqual(ss.GENROU.M.v[0], 2.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[0], omega_addr[0]], 2.0)

        # set a list of values
        ss.GENROU.set("M", ["GENROU_1", "GENROU_2"], "v", [2.0, 3.5])
        np.testing.assert_equal(ss.GENROU.M.v[[0, 1]], [2.0, 3.5])
        self.assertEqual(ss.TDS.Teye[omega_addr[0], omega_addr[0]], 2.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[1], omega_addr[1]], 3.5)

        # set a list of values
        ss.GENROU.set("M", ["GENROU_3", "GENROU_5"], "v", [2.0, 3.5])
        np.testing.assert_equal(ss.GENROU.M.v[[2, 4]], [2.0, 3.5])
        self.assertEqual(ss.TDS.Teye[omega_addr[2], omega_addr[2]], 2.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[4], omega_addr[4]], 3.5)

        # set a list of idxes with a single element to an array of values
        ss.GENROU.set("M", ["GENROU_4"], "v", np.array([4.0]))
        np.testing.assert_equal(ss.GENROU.M.v[3], 4.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[3], omega_addr[3]], 4.0)

        # set an array of idxes with a single element to an array of values
        ss.GENROU.set("M", np.array(["GENROU_4"]), "v", np.array([5.0]))
        np.testing.assert_equal(ss.GENROU.M.v[3], 5.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[3], omega_addr[3]], 5.0)

        # set an array of idxes with a list of single value
        ss.GENROU.set("M", np.array(["GENROU_4"]), "v", 6.0)
        np.testing.assert_equal(ss.GENROU.M.v[3], 6.0)
        self.assertEqual(ss.TDS.Teye[omega_addr[3], omega_addr[3]], 6.0)
