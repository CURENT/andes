from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from andes.main import main  # NOQA
from andes.plot import main as plot  # NOQA

__author__ = 'Hantao Cui'

__all__ = ['main', 'consts', 'plot', 'system', 'config', 'routines', 'filters', 'utils']
