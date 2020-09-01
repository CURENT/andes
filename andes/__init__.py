from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from andes import io         # NOQA
from andes import core       # NOQA
from andes import models     # NOQA
from andes import routines   # NOQA
from andes import utils      # NOQA
from andes import variables  # NOQA

from andes.main import config_logger, run, load, prepare  # NOQA
from andes.system import System                    # NOQA
from andes.utils.paths import get_case, list_cases # NOQA


__author__ = 'Hantao Cui'

__all__ = ['main', 'plot', 'system', 'cli',
           'utils', 'core', 'models', 'io', 'routines', 'variables',
           '__version__']
