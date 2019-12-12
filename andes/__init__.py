from andes import main  # NOQA
from andes.main import run, run_stock  # NOQA
from andes import system

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


__author__ = 'Hantao Cui'

__all__ = ['main', 'plot', 'system',
           'common', 'core', 'devices', 'io', 'programs', 'variables']
