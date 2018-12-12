import os
import logging
import platform
import tempfile

logger = logging.getLogger(__name__)


def get_config_load_path(conf_path=None):
    """
    Return config file load path

    Priority:
        1. conf_path
        2. current directory
        3. home directory

    Parameters
    ----------
    conf_path

    Returns
    -------

    """

    if conf_path is None:
        # test ./andes.conf
        if os.path.isfile('andes.conf'):
            conf_path = 'andes.conf'
        # test ~/andes.conf
        home_dir = os.path.expanduser('~')
        if os.path.isfile(os.path.join(home_dir, '.andes', 'andes.conf')):
            conf_path = os.path.join(home_dir, '.andes', 'andes.conf')

    if conf_path is not None:
        logger.debug('Found config file at {}.'.format(conf_path))

    return conf_path


def get_log_dir():
    """
    Get a directory for logging

    On Linux or macOS, '/tmp/andes' is the default. On Windows,
    '%APPDATA%/andes' is the default.

    Returns
    -------
    str
        Path to the logging directory
    """
    PATH = ''
    if platform.system() in ('Linux', 'Darwin'):
        PATH = tempfile.mkdtemp(prefix='andes-')

    elif platform.system() == 'Windows':
        APPDATA = os.getenv('APPDATA')
        PATH = os.path.join(APPDATA, 'andes')

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    return PATH
