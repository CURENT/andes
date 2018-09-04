from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import main  # NOQA
from . import plot  # NOQA

__author__ = 'Hantao Cui'

__all__ = ['main', 'consts', 'plot', 'system',
           'config', 'filters', 'models', 'routines', 'utils', 'variables']
