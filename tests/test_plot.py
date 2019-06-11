import unittest
import andes.plot


class TestPlotParseYNew(unittest.TestCase):

    def runTest(self):
        self.assertListEqual(andes.plot.parse_y(['1'], 5),
                             [1])

        self.assertListEqual(andes.plot.parse_y(['1:3'], 5),
                             [1, 2])
        self.assertListEqual(andes.plot.parse_y(['1:3:'], 5),
                             [1, 2])
        self.assertListEqual(andes.plot.parse_y(['1:10'], 5),
                             [1, 2, 3, 4])
        self.assertListEqual(andes.plot.parse_y(['1:10:2'], 5),
                             [1, 3])

        self.assertListEqual(andes.plot.parse_y(['-2:3'], 5),
                             [0, 1, 2])
        self.assertListEqual(andes.plot.parse_y(['-3:10'], 5),
                             [0, 1, 2, 3, 4])
        self.assertListEqual(andes.plot.parse_y(['-5:10:2'], 5),
                             [1, 3])

        self.assertListEqual(andes.plot.parse_y(['5:0:-1'], 5),
                             [4, 3, 2, 1])
        self.assertListEqual(andes.plot.parse_y(['10:1:'], 5),
                             [])

        self.assertListEqual(andes.plot.parse_y(['-1:999:'], 5),
                             [0, 1, 2, 3, 4])
        self.assertListEqual(andes.plot.parse_y(['-999:-1:'], 5),
                             [])
        self.assertListEqual(andes.plot.parse_y(['-999:-1:2'], 5),
                             [])

        self.assertListEqual(andes.plot.parse_y(['-99s9:-1:2'], 5),
                             [])

        self.assertListEqual(andes.plot.parse_y(['1', '4'], 5),
                             [1, 4])

        self.assertListEqual(andes.plot.parse_y(['1', '4', '5', '10'], 5),
                             [1, 4])

        self.assertListEqual(andes.plot.parse_y(['1', '4', '5', '10'], 5, lower=2),
                             [4])
