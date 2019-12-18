from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from andes import main  # NOQA
from andes import io  # NOQA
from andes.main import run  # NOQA
from andes import system  # NOQA
from andes.system import System  # NOQA


__author__ = 'Hantao Cui'

__all__ = ['main', 'plot', 'system',
           'common', 'core', 'devices', 'io', 'programs', 'variables']
