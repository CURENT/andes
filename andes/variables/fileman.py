import io
import logging
import os

logger = logging.getLogger(__name__)


class FileMan:
    """
    Define a File Manager class for System
    """

    def __init__(self, case=None, **kwargs):
        """
        Initialize the output file names.
        For inputs, all absolute paths will be respected.
        All relative paths are relative to `input_path`.

        case: must be full path to case

        output: desired name for format conversion output

        input_path: default path for input files that only contains file name. If `input_path` is not provided,
                    it will be derived from the path of `case`.

        output_path: path for output files. Default to current working directory where `andes` is invoked.
        """
        self.input_format = None
        self.output_format = None
        self.add_format = None

        self.case = None
        self.input_path = ''
        self.case_path = ''  # absolute path containinng the case file
        self.fullname = None
        self.name = None
        self.ext = None
        self.addfile = None
        self.pert = None
        self.output_path = None
        self.no_output = True

        self._out_fields = ['txt', 'dump', 'lst', 'eig', 'npy', 'npz', 'csv', 'mat', 'prof', 'prof_raw']
        self.set(case, **kwargs)

    def set(self, case=None, **kwargs):
        """
        Perform the input and output set up.
        """
        self.case = case
        input_format = kwargs.get('input_format')
        add_format = kwargs.get('add_format')
        input_path = kwargs.get('input_path')
        addfile = kwargs.get('addfile')
        no_output = kwargs.get('no_output')
        output_path = kwargs.get('output_path')
        output = kwargs.get('output')  # base file name for the output
        pert = kwargs.get('pert')
        dump = kwargs.get('dump')

        # set output fields anyway
        for item in self._out_fields:
            self.__dict__[item] = None

        if case is None:
            return

        self.input_format = input_format
        self.add_format = add_format
        self.input_path = input_path if input_path is not None else ''
        self.output_path = output_path if output_path is not None else ''

        self.case = case
        if os.path.isfile(case):
            self.case_path, self.fullname = os.path.split(self.case)
            self.case_path = os.path.abspath(self.case_path)
        else:
            self.fullname = ''

        # `self.name` is the name part without extension
        self.name, self.ext = os.path.splitext(self.fullname)

        self.addfile = self.get_fullpath(addfile)
        self.pert = self.get_fullpath(pert)
        if dump is None:
            dump = self.name

        # set the output file names anyway
        if not output:
            output = add_suffix(self.name, 'out')
        prof = add_suffix(self.name, 'prof')
        eig = add_suffix(self.name, 'eig')
        mat = add_suffix(self.name, 'As')

        self.lst = os.path.join(self.output_path, output + '.lst')
        self.npy = os.path.join(self.output_path, output + '.npy')
        self.npz = os.path.join(self.output_path, output + '.npz')
        self.csv = os.path.join(self.output_path, output + '.csv')
        self.txt = os.path.join(self.output_path, output + '.txt')

        self.eig = os.path.join(self.output_path, eig + '.txt')
        self.prof = os.path.join(self.output_path, prof + '.txt')
        self.mat = os.path.join(self.output_path, mat + '.mat')
        self.prof_raw = os.path.join(self.output_path, prof + '.prof')
        self.dump = os.path.join(self.output_path, dump + '.xlsx')

        # use `self.no_output` to control calls to saving functions
        if no_output:
            self.no_output = True
        else:
            self.no_output = False

    def get_fullpath(self, fullname=None):
        """
        Return the original full path if full path is specified, otherwise
        search in the case file path.

        Parameters
        ----------
        fullname : str, optional
            Full name of the file. If relative, prepend `input_path`.
            Otherwise, leave it as is.
        """
        # if is an empty path
        if not fullname:
            return fullname
        if isinstance(fullname, io.IOBase):
            return ''

        isabs = os.path.isabs(fullname)

        path, name = os.path.split(fullname)

        if not name:  # path to a folder
            return ''
        else:  # path to a file
            if isabs:
                return fullname
            else:
                return os.path.join(self.input_path, path, name)


def add_suffix(fullname, suffix):
    """
    Add suffix to a full file name.
    """

    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext
