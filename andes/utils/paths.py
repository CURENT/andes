"""
Utility functions for loading andes stock test cases
"""
import os
import platform
import tempfile
import pathlib
import logging

logger = logging.getLogger(__name__)


class DisplayablePath:
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = pathlib.Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = pathlib.Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


def cases_root():
    """Return the root path to the stock cases"""
    dir_name = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir_name, '..', 'cases')


def tests_root():
    """Return the root path to the stock cases"""
    dir_name = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(dir_name, '..', '..', 'tests'))


def get_case(rpath, check=True):
    """
    Return the path to a stock case for a given path relative to ``andes/cases``.

    To list all cases, use ``andes.list_cases()``.

    Parameters
    ----------
    check : bool
        True to check if file exists

    Examples
    --------
    To get the path to the case `kundur_full.xlsx` under folder `kundur`, do ::

        andes.get_case('kundur/kundur_full.xlsx')

    """
    case_path = os.path.join(cases_root(), rpath)
    case_path = os.path.normpath(case_path)

    if check is True and (not os.path.isfile(case_path)):
        raise FileNotFoundError(f'"{rpath}" is not a valid relative path to a stock case.')
    return case_path


def list_cases(rpath='.', no_print=False):
    """
    List stock cases under a given folder relative to ``andes/cases``
    """
    case_path = os.path.join(cases_root(), rpath)
    case_path = os.path.normpath(case_path)

    tree = DisplayablePath.make_tree(pathlib.Path(case_path))
    out = []
    for path in tree:
        out.append(path.displayable())

    out = '\n'.join(out)
    if no_print is False:
        print(out)
    else:
        return out


def get_config_path(file_name='andes.rc'):
    """
    Return the path of the config file to be loaded.

    Search Priority: 1. current directory; 2. home directory.

    Parameters
    ----------
    file_name : str, optional
        Config file name with the default as ``andes.rc``.

    Returns
    -------
    Config path in string if found; None otherwise.
    """

    conf_path = None
    home_dir = os.path.expanduser('~')

    # test ./andes.conf
    if os.path.isfile(file_name):
        conf_path = file_name
    # test ~/andes.conf
    elif os.path.isfile(os.path.join(home_dir, '.andes', file_name)):
        conf_path = os.path.join(home_dir, '.andes', file_name)

    return conf_path


def get_pycode_path(pycode_path=None, mkdir=False):
    """
    Get the path to the ``pycode`` folder.
    """

    if pycode_path is None:
        pycode_path = os.path.join(get_dot_andes_path(), 'pycode')

    if mkdir is True:
        os.makedirs(pycode_path, exist_ok=True)

    return pycode_path


def get_pkl_path():
    """
    Get the path to the picked/dilled function calls.

    Returns
    -------
    str
        Path to the calls.pkl file

    """
    pkl_name = 'calls.pkl'
    andes_path = get_dot_andes_path()

    if not os.path.exists(andes_path):
        os.makedirs(andes_path)

    pkl_path = os.path.join(andes_path, pkl_name)

    return pkl_path


def get_dot_andes_path():
    """
    Return the path to ``<HomeDir>/.andes``
    """
    return os.path.join(str(pathlib.Path.home()), '.andes')


def get_log_dir():
    """
    Get the directory for log file.

    On Linux or macOS, ``/tmp/andes`` is the default. On Windows, ``%APPDATA%/andes`` is the default.

    Returns
    -------
    str
        The path to the temporary logging directory
    """
    path = ''
    if platform.system() in ('Linux', 'Darwin'):
        path = tempfile.mkdtemp(prefix='andes-')

    elif platform.system() == 'Windows':
        appdata = os.getenv('APPDATA')
        path = os.path.join(appdata, 'andes')

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def confirm_overwrite(outfile, overwrite=None):
    try:
        if os.path.isfile(outfile):
            if overwrite is None:
                choice = input(f'File "{outfile}" already exist. Overwrite? [y/N]').lower()
                if len(choice) == 0 or choice[0] != 'y':
                    logger.warning(f'File "{outfile}" not overwritten.')
                    return False
            elif overwrite is False:
                return False
    except TypeError:
        pass

    return True
