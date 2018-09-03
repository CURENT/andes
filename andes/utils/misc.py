import os
import pathlib
import logging

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
        if os.path.isfile(os.path.join(str(pathlib.Path.home()), '.andes', 'andes.conf')):
            conf_path = os.path.join(str(pathlib.Path.home()), '.andes', 'andes.conf')

    if conf_path is not None:
        logger.debug('Found config file at {}.'.format(conf_path))

    return conf_path
