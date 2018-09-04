import importlib
from .eig import EIG
from .pflow import PFLOW
from .tds import TDS


def get_command(all_pkg, hook):
    """
    Collect the command-line interface names by querying ``hook`` in ``all_pkg``

    Parameters
    ----------
    all_pkg: list
        list of package files
    hook: str
        A variable where the command is stored. ``__cli__`` by default.
    Returns
    -------
    list
    """
    ret = []
    for r in all_pkg:
        module = importlib.import_module(__name__ + '.' + r.lower())
        ret.append(getattr(module, hook))
    return ret


__all__ = ['PFLOW', 'TDS', 'EIG']
__cli__ = get_command(__all__, '__cli__')
