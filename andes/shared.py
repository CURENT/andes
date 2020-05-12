"""
Shared constants and delayed imports.

PAST NOTES
**********
Known issues of LazyImport ::

    1) The delayed import of pandas and newton_krylov will cause a ``RuntimeWarning``

    RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header,
    got 216 from PyObject
        return f(*args, **kwds)

    2) High overhead when called hundreds of thousands times. For example, NumPy must not be imported with
    LazyImport.

    3) Prevents from serialization due to recursion depth.
"""

import pandas as pd  # NOQA
import matplotlib as mpl  # NOQA
from matplotlib import pyplot as plt  # NOQA
from pathos.multiprocessing import Pool  # NOQA
from multiprocessing import Process  # NOQA
import unittest  # NOQA
import yaml  # NOQA

from scipy.optimize import newton_krylov  # NOQA
from scipy.optimize import fsolve  # NOQA
from scipy.integrate import solve_ivp  # NOQA
from scipy.integrate import odeint  # NOQA

import math
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


deg2rad = math.pi/180
jac_names = ('fx', 'fy', 'gx', 'gy')
jac_types = ('c', '')

jac_full_names = list()
for jname in jac_names:
    for jtype in jac_types:
        jac_full_names.append(jname + jtype)

# *** Deprecated  ***
#
# # Packages
# pd = LazyImport('import pandas')
# plt = LazyImport('from matplotlib import pyplot')
# mpl = LazyImport('import matplotlib')
# Pool = LazyImport('from pathos.multiprocessing import Pool')
# unittest = LazyImport('import unittest')
# yaml = LazyImport('import yaml')
#
# # function calls
# newton_krylov = LazyImport('from scipy.optimize import newton_krylov')
# fsolve = LazyImport('from scipy.optimize import fsolve')
# solve_ivp = LazyImport('from scipy.integrate import solve_ivp')
# odeint = LazyImport('from scipy.integrate import odeint')
#
# *** Deprecated  ***
