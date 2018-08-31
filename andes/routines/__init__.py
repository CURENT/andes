import importlib


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
        module = importlib.import_module(__name__ + '.' + r)
        ret.append(getattr(module, hook))
    return ret


__all__ = ['pflow', 'tds', 'eigenanalysis']
__cli__ = get_command(__all__, '__cli__')
