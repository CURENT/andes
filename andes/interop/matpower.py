"""
Interoperability with MATPOWER through ``matpower-pip``.

Please install the Python package ``matpower`` and configure MATLAB or Octave,
following the instructions at matpower-pip_.

To create a MATLAB/Octave instance, do:

.. code:: python

    from andes.interop.matpower import start_instance

    m = start_instance()

.. _matpower-pip: https://github.com/yasirroni/matpower-pip

"""


import logging
from functools import wraps

import andes

from andes.shared import Oct2PyError

logger = logging.getLogger(__name__)


def require_matpower(f):
    """
    Decorator for functions that require matpower.
    """

    @wraps(f)
    def wrapper(*args, **kwds):
        try:
            from matpower import start_instance   # NOQA
        except ImportError:
            raise ModuleNotFoundError("Package `matpower` needs to be manually installed.")

        return f(*args, **kwds)

    return wrapper


@require_matpower
def from_matpower(m, varname, system=None):
    """
    Retrieve a MATPOWER mpc case from a MATLAB/Octave instance.

    Parameters
    ----------
    m : MATLAB/Octave instance
        Instance from which to retrieve the MATPOWER case.
    varname : str
        Name of the variable in the MATPOWER MPC format to retrieve.

    Returns
    -------
    :class:`~andes.system.System`
        System from the mpc case. The system will not have been set up.

    Examples
    --------
    To retrieve a case from MATPOWER from instance ``m``, do the following:

    .. code:: python


        from andes.interop.matpower import start_instance, to_matpower, from_matpower

        system = from_matpower(m, 'mpc')

    One can create an Excel file with dynamic data only and use the ``xlsx``
    parser to load data into ``system``:

    .. code:: python

        from andes.io import xlsx

        xlsx.read(system, andes.get_case('ieee14/ieee14_dyn_only.xlsx'))

    The ``ieee14_dyn_only.xlsx`` is an example spreadsheet that only contains
    dynamic components. One will need to create the ``idx`` correctly for
    dynamic components to match these from the MATPOWER case. If not sure about
    the indices, one can save the ANDES system to an Excel file, using:

    .. code:: python

        xlsx.write(system, 'system_static.xlsx')

    """

    if m is None:
        raise ImportError("MATPOWER is not installed or not properly configured.")

    mpc = {}
    mpc['baseMVA'] = m.eval(f'{varname}.baseMVA;')
    mpc['version'] = m.eval(f'{varname}.version;')
    mpc['bus'] = m.eval(f'{varname}.bus;')
    mpc['gen'] = m.eval(f'{varname}.gen;')
    mpc['branch'] = m.eval(f'{varname}.branch;')

    try:
        mpc['gencost'] = m.eval(f'{varname}.gencost;')
    except Oct2PyError:
        logger.debug("No gencost in mpc")
    try:
        mpc['bus_names'] = m.eval(f'{varname}.bus_names;')
    except Oct2PyError:
        logger.debug("No bus_name in mpc")

    if system is None:
        system = andes.System()

    andes.io.matpower.mpc2system(mpc, system)

    return system


@require_matpower
def to_matpower(m, varname, system):
    """
    Send an ANDES case to a running MATLAB instance.

    Parameters
    ----------
    m : MATLAB/Octave instance
        Instance to which to send the MATPOWER case.
    varname : str
        Name of the variable to store the mpc case in MATLAB/Octave.
    system : :class:`~andes.system.System`
        System whose power flow data to send to MATPOWER.

    Examples
    --------

    The code below will create an IEEE 14-bus example system in ANDES, convert it to
    MATPOWER's case, and send to the MATLAB/Octave instance.

    .. code:: python

        import andes

        from andes.interop.matpower import (start_instance,
            to_matpower, from_matpower)

        m = start_instance()

        ss = andes.system.example()
        mpc = to_matpower(m, 'mpc', ss)

        m.eval("runpf(mpc)")

        mpc_out = m.pull("mpc")  # retrieve the mpc case from MATLAB/Octave

    """

    if m is None:
        raise ImportError("MATPOWER is not installed or not properly configured.")

    mpc = andes.io.matpower.system2mpc(system)
    m.push(varname, mpc)

    return mpc
