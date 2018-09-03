import os


class FileMan(object):
    """Define a File Manager class for PowerSystem"""

    def __init__(self,
                 case,
                 input_format=None,
                 addfile=None,
                 config=None,
                 no_output=False,
                 dynfile=None,
                 dump_raw=None,
                 output_format=None,
                 output=None,
                 pert=None,
                 **kwargs):
        """initialize the output file names

        case: must be full path to case
        fullname: full name of case only
        name: name of case WITHOUT extension

        dump: desired simulation result file name
        output: desired name for format conversion output
        """
        if isinstance(input_format, str):
            self.input_format = input_format.lower()
        else:
            self.input_format = None
        if isinstance(output_format, str):
            self.output_format = output_format.lower()
        else:
            self.output_format = None

        self.case = case
        path, fullname = os.path.split(case)
        self.fullname = fullname
        self.name, self.ext = os.path.splitext(fullname)
        if not path:
            self.path = os.getcwd()
        else:
            self.path = path

        self.addfile = self.get_fullpath(addfile)
        self.pert = pert
        self.dynfile = self.get_fullpath(dynfile)
        self.config = self.get_fullpath(config)
        self.add_format = None

        if no_output:
            self.no_output = True
            self.log = None
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

            self.lst = add_ext(output, 'lst')
            self.eig = add_ext(eig, 'txt')
            self.dat = add_ext(output, 'dat')
            self.output = add_ext(output, 'txt')
            self.dump_raw = add_ext(dump_raw, 'dm')
            self.prof = add_ext(prof, 'txt')

    def get_fullpath(self, fullname=None):
        """
        Return the original full path if full path is specified, otherwise
        search in the case file path
        """
        if not fullname:
            return None

        path, name = os.path.split(fullname)
        if not name:
            return None
        else:
            if not path:
                return os.path.join(self.path, name)
            else:
                return os.path.join(path, name)


def add_ext(name, extension):
    """Add extension to a name, discard the previos extension if exists"""
    name, ext = os.path.splitext(name)
    return name + '.' + extension


def add_suffix(fullname, suffix):
    """ Add suffix to a full file name"""
    name, ext = os.path.splitext(fullname)
    return name + '_' + suffix + ext
