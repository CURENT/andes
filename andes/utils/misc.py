import logging
import os
import platform
import tempfile
from _pydecimal import Decimal, ROUND_DOWN
from time import time

logger = logging.getLogger(__name__)


def elapsed(t0=0.0):
    """
    Get the elapsed time from the give time.
    If the start time is not given, returns the unix-time.

    Returns
    -------
    t : float
        Elapsed time from the given time; Otherwise the epoch time.
    s : str
        The elapsed time in seconds in a string
    """
    t = time()
    dt = t - t0
    dt_sec = Decimal(str(dt)).quantize(Decimal('.0001'), rounding=ROUND_DOWN)
    if dt_sec <= 1:
        s = str(dt_sec) + ' second'
    else:
        s = str(dt_sec) + ' seconds'
    return t, s


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


def to_number(s):
    """
    Convert a string to a number. If unsuccessful, return the de-blanked string.
    """
    ret = s
    # try converting to float
    try:
        ret = float(s)
    except ValueError:
        ret = ret.strip('\'').strip()

    # try converting to uid
    try:
        ret = int(s)
    except ValueError:
        pass

    # try converting to boolean
    if ret == 'True':
        ret = True
    elif ret == 'False':
        ret = False
    elif ret == 'None':
        ret = None
    return ret


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
