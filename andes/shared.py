"""
Shared constants and delayed imports.

Known issues ::

    1) The delayed import of pandas and newton_krylov will cause a ``RuntimeWarning``

    RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header,
    got 216 from PyObject
        return f(*args, **kwds)
"""
import importlib

pi = 3.14159265358973
jpi2 = 1.5707963267948966j
rad2deg = 57.295779513082323
deg2rad = 0.017453292519943

pd = None
newton_krylov = None


def load_pandas():
    """
    Import pandas to globals() if not exist.
    """
    if globals()['pd'] is None:
        globals()['pd'] = importlib.import_module('pandas')

    return True


def load_newton_krylov():
    if globals()['newton_krylov'] is None:
        optimize = importlib.import_module('scipy.optimize')
        globals()['newton_krylov'] = getattr(optimize, 'newton_krylov')

    return True
