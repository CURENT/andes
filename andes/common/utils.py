import functools
import operator
import logging
import os
import platform
import tempfile
from _pydecimal import Decimal, ROUND_DOWN
from time import time
import numpy as np


logger = logging.getLogger(__name__)


def list_flatten(idx):
    if len(idx) > 0 and isinstance(idx[0], (list, np.ndarray)):
        return functools.reduce(operator.iconcat, idx, [])
    else:
        return idx


def elapsed(t0=0.0):
    """get elapsed time from the give time

    Returns:
        now: the absolute time now
        dt_str: elapsed time in string
        """
    now = time()
    dt = now - t0
    dt_sec = Decimal(str(dt)).quantize(Decimal('.0001'), rounding=ROUND_DOWN)
    if dt_sec <= 1:
        dt_str = str(dt_sec) + ' second'
    else:
        dt_str = str(dt_sec) + ' seconds'
    return now, dt_str


def get_config_load_path(conf_path=None, file_name='andes.rc'):
    """
    Return config file load path

    Priority:
        1. conf_path
        2. current directory
        3. home directory

    Parameters
    ----------
    conf_path

    file_name

    Returns
    -------

    """

    if conf_path is None:
        # test ./andes.conf
        if os.path.isfile(file_name):
            conf_path = file_name
        # test ~/andes.conf
        home_dir = os.path.expanduser('~')
        if os.path.isfile(os.path.join(home_dir, '.andes', file_name)):
            conf_path = os.path.join(home_dir, '.andes', file_name)

    else:
        logger.debug('Found config file at {}.'.format(conf_path))

    return conf_path


def get_log_dir():
    """
    Get a directory for logging.

    On Linux or macOS, '/tmp/andes' is the default. On Windows,
    '%APPDATA%/andes' is the default.

    Returns
    -------
    str
        Path to the logging directory
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
