from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from andes import io  # NOQA
from andes.main import run, load  # NOQA
from andes import system  # NOQA
from andes.system import System  # NOQA


__author__ = 'Hantao Cui'

__all__ = ['main', 'plot', 'system', 'cli',
           'utils', 'core', 'models', 'io', 'routines', 'variables']
