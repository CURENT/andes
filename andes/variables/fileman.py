import logging
import os

logger = logging.getLogger(__name__)


class FileMan(object):
    """Define a File Manager class for PowerSystem"""

    def __init__(self,
                 case,
                 input_format=None,
                 input_path=None,
                 addfile=None,
                 config=None,
                 no_output=False,
                 dynfile=None,
                 dump_raw=None,
                 output_format=None,
                 output_path='',
                 output=None,  # base file name for the output
                 pert=None,
                 **kwargs):
        """
        Initialize the output file names.
        For inputs, all absolute paths will be respected; and all relative paths are relative to `input_path`.

        case: must be full path to case

        dump_raw: desired simulation result file name

        output: desired name for format conversion output

        input_path: default path for input files that only contains file name. If `input_path` is not provided,
                    it will be derived from the path of `case`.

        output_path: path for output files. Default to current working directory where `andes` is invoked.
        """
        if isinstance(input_format, str):
            self.input_format = input_format.lower()
        else:
            self.input_format = None
        if isinstance(output_format, str):
            self.output_format = output_format.lower()
        else:
            self.output_format = None

        if input_path is not None:
            logger.debug('input_path provided as {}.'.format(input_path))
            self.case_path = input_path
        else:
            self.case_path = os.getcwd()  # default to current directory

        if os.path.isabs(case):
            self.case = case
        else:
            self.case = self.get_fullpath(case)
            logger.debug(self.case)

        # update `self.case_path` if `case` contains a path
        self.case_path, self.fullname = os.path.split(self.case)

        # `self.name` is the name part without extension
        self.name, self.ext = os.path.splitext(self.fullname)

        self.addfile = self.get_fullpath(addfile)
        self.pert = self.get_fullpath(pert)
        self.dynfile = self.get_fullpath(dynfile)
        self.config = self.get_fullpath(config)
        self.add_format = None

        # use the path where andes is executed as the default output path
        self.output_path = os.getcwd() if not output_path else output_path
        if no_output:
            self.no_output = True
            self.output = None
            self.lst = None
            self.eig = None
            self.dat = None
            self.dump_raw = None
            self.prof = None
        else:
            self.no_output = False
            if not output:
                output = add_suffix(self.name, 'out')
            if not dump_raw:
                dump_raw = add_suffix(self.name, 'dm')
            prof = add_suffix(self.name, 'prof')
            eig = add_suffix(self.name, 'eig')

            self.lst = os.path.join(self.output_path, output + '.lst')
            self.dat = os.path.join(self.output_path, output + '.dat')
            self.output = os.path.join(self.output_path, output + '.txt')

            self.eig = os.path.join(self.output_path, eig + '.txt')
            self.dump_raw = os.path.join(self.output_path, dump_raw + '.dm')
            self.prof = os.path.join(self.output_path, prof + '.txt')

    def get_fullpath(self, fullname=None, relative_to=None):
        """
        Return the original full path if full path is specified, otherwise
        search in the case file path
        """
        # if is an empty path
        if not fullname:
            return fullname

        isabs = os.path.isabs(fullname)

        path, name = os.path.split(fullname)

        if not name:  # path to a folder
            return None
        else:  # path to a file
            if isabs:
                return fullname
            else:
                return os.path.join(self.case_path, path, name)


def add_suffix(fullname, suffix):
    """ Add suffix to a full file name"""
    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext
