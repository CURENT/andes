import unittest
import os
from andes.variables import fileman


class TestVariablesFileMan(unittest.TestCase):

    def setUp(self):
        case_file_path = '../cases/ieee14/ieee14_syn.dm'
        self.fileman = fileman.FileMan(case_file_path,
                                       input_format=None, addfile=None,
                                       config=os.path.join(os.path.expanduser('~'), '.andes', 'andes.conf'),
                                       no_output=False, dynfile='ieee14_syn2.dm', dump_raw=None,
                                       output_format=None, output_path='', output=None,
                                       pert='',
                                       )

    def runTest(self):
        test_abs_path = os.path.abspath('.')
        case_abs_path = os.path.abspath('../cases/ieee14/ieee14_syn.dm')

        path_to_case_folder, full_file_name = os.path.split(case_abs_path)
        file_name, file_ext = os.path.splitext(full_file_name)

        # test input file paths
        self.assertEqual(os.path.abspath(self.fileman.case),
                         case_abs_path)

        self.assertEqual(self.fileman.addfile, None)

        self.assertEqual(os.path.abspath(self.fileman.dynfile),
                         os.path.join(path_to_case_folder, 'ieee14_syn2.dm'))

        self.assertEqual(self.fileman.pert, None)
        self.assertEqual(self.fileman.config,
                         os.path.join(os.path.expanduser('~'), '.andes', 'andes.conf'))

        # test output file paths

        self.assertEqual(self.fileman.no_output, False)
        self.assertEqual(self.fileman.dat, os.path.join(test_abs_path, file_name + '_out.dat'))
        self.assertEqual(self.fileman.lst, os.path.join(test_abs_path, file_name + '_out.lst'))
        self.assertEqual(self.fileman.eig, os.path.join(test_abs_path, file_name + '_eig.txt'))
        self.assertEqual(self.fileman.dump_raw, os.path.join(test_abs_path, file_name + '_dm.dm'))
        self.assertEqual(self.fileman.prof, os.path.join(test_abs_path, file_name + '_prof.txt'))


class TestVariablesFileManNoOutput(unittest.TestCase):

    """Test fileman with `no_output == True`.
    """
    def setUp(self):
        case_file_path = '../cases/ieee14/ieee14_syn.dm'
        self.fileman = fileman.FileMan(case_file_path,
                                       input_format=None, addfile=None, config=None,
                                       no_output=True, dynfile=None, dump_raw='dump_raw.dm',
                                       output_format=None, output_path=os.getcwd(), output='test_out',
                                       pert='',
                                       )

    def runTest(self):
        # test output file paths

        self.assertEqual(self.fileman.no_output, True)
        self.assertEqual(self.fileman.dat, None)
        self.assertEqual(self.fileman.lst, None)
        self.assertEqual(self.fileman.eig, None)
        self.assertEqual(self.fileman.dump_raw, None)
        self.assertEqual(self.fileman.prof, None)


class TestVariablesFileManWithInputPath(unittest.TestCase):

    def setUp(self):
        case_file_path = 'ieee14_syn.dm'
        self.fileman = fileman.FileMan(case_file_path,
                                       input_format=None, addfile=None,
                                       input_path=os.path.abspath('../cases/ieee14'),
                                       config=os.path.join(os.path.expanduser('~'), '.andes', 'andes.conf'),
                                       no_output=False,
                                       dynfile='./ieee14_syn2.dm',
                                       dump_raw=None,
                                       output_format=None, output_path='', output=None,
                                       pert='',
                                       )

    def runTest(self):
        test_abs_path = os.path.abspath('.')
        case_abs_path = os.path.abspath('../cases/ieee14/ieee14_syn.dm')

        path_to_case_folder, full_file_name = os.path.split(case_abs_path)
        file_name, file_ext = os.path.splitext(full_file_name)

        # test input file paths
        self.assertEqual(self.fileman.case, case_abs_path)
        self.assertEqual(self.fileman.addfile, None)
        self.assertEqual(os.path.abspath(self.fileman.dynfile),
                         os.path.join(path_to_case_folder, 'ieee14_syn2.dm'))
        self.assertEqual(self.fileman.pert, None)
        self.assertEqual(self.fileman.config, os.path.join(os.path.expanduser('~'), '.andes', 'andes.conf'))

        # test output file paths

        self.assertEqual(self.fileman.no_output, False)
        self.assertEqual(self.fileman.dat, os.path.join(test_abs_path, file_name + '_out.dat'))
        self.assertEqual(self.fileman.lst, os.path.join(test_abs_path, file_name + '_out.lst'))
        self.assertEqual(self.fileman.eig, os.path.join(test_abs_path, file_name + '_eig.txt'))
        self.assertEqual(self.fileman.dump_raw, os.path.join(test_abs_path, file_name + '_dm.dm'))
        self.assertEqual(self.fileman.prof, os.path.join(test_abs_path, file_name + '_prof.txt'))


if __name__ == '__main__':
    unittest.main()
