"""
Shared constants and delayed imports.

Known issues ::

    1) The delayed import of pandas and newton_krylov will cause a ``RuntimeWarning``

    RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header,
    got 216 from PyObject
        return f(*args, **kwds)

    2) High overhead when called hundreds of thousands times. For example, NumPy must not be imported with
    LazyImport.
"""
from andes.utils.lazyimport import LazyImport

# Packages
pd = LazyImport('import pandas')
plt = LazyImport('from matplotlib import pyplot')
mpl = LazyImport('import matplotlib')
Process = LazyImport('from multiprocessing import Process')
unittest = LazyImport('import unittest')
yaml = LazyImport('import yaml')

# function calls
newton_krylov = LazyImport('from scipy.optimize import newton_krylov')
fsolve = LazyImport('from scipy.optimize import fsolve')
solve_ivp = LazyImport('from scipy.integrate import solve_ivp')
odeint = LazyImport('from scipy.integrate import odeint')

import coloredlogs  # NOQA
import numpy as np  # NOQA
from tqdm import tqdm  # NOQA

import cvxopt  # NOQA
from cvxopt import umfpack  # NOQA
from cvxopt import spmatrix, matrix, sparse, spdiag  # NOQA
from numpy import ndarray  # NOQA

from andes.utils.texttable import Texttable  # NOQA

try:
    from cvxoptklu import klu
except ImportError:
    klu = None


jpi2 = 1.5707963267948966j
rad2deg = 57.295779513082323
deg2rad = 0.017453292519943
