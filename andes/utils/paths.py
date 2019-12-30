"""
Utility functions for loading andes stock test cases
"""
import os
import platform
import tempfile


def cases_root():
    """Return the root path to the stock cases"""
    dir_name = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dir_name, '..', '..', 'cases')


def tests_root():
    """Return the root path to the stock cases"""
    dir_name = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(dir_name, '..', '..', 'tests'))


def get_case(rpath):
    """Return the path to the stock cases"""
    case_path = os.path.join(cases_root(), rpath)
    case_path = os.path.normpath(case_path)

    if not os.path.isfile(case_path):
        raise FileNotFoundError(f"Path {rpath} is not a valid case file.")

    return case_path


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
