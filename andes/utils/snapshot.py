"""
Utility functions for saving and loading snapshots.

Code Examples:

1. Setup base case and save the snapshot for once:

.. code:: python

    import andes

    ss = andes.run(andes.get_case("ieee14/ieee14_linetrip.xlsx"))
    ss.Toggle.u.v[:] = 0  # turn off line trips for the base case
    xy = ss.TDS.init()

    andes.utils.snapshot.save_ss("ieee14_snapshot.pkl", ss)

2.  For every scenario afterwards, load the snapshot and apply
disturbances:

.. code:: python

    import andes

    ss = andes.utils.snapshot.load_ss("ieee14_snapshot.pkl")

    # apply specific disturbances
    ss.GENROU.omega.v[0] = 1.02

    ss.TDS.run()

"""

import dill

from andes.system import fix_view_arrays, import_pycode


def save_ss(path, system):
    """
    Save a system with all internal states as a snapshot.

    Returns
    -------
    Path to the saved snapshot.

    Warnings
    --------
    One limitation of the current implementation is version dependency.
    The snapshots only work with the specific ANDES version that created it.
    """

    system.remove_pycapsule()

    if hasattr(path, 'write'):
        dill.dump(system, path, recurse=True)
    else:
        with open(path, 'wb') as file:
            dill.dump(system, file, recurse=True)

    return path


def load_ss(path):
    """
    Load an ANDES snapshot and return a System object.

    Parameters
    ----------
    path : str
        Path to the snapshot file.

    Returns
    -------
    andes.system.System
        The loaded system object
    """

    # the line below is needed to properly import `pycode`.
    import_pycode()

    if hasattr(path, 'read'):
        system = dill.load(path)
    else:
        with open(path, 'rb') as file:
            system = dill.load(file)

    # point the "view arrays" to the correct memory
    fix_view_arrays(system)

    return system
