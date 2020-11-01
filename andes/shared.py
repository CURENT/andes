"""
Shared constants and delayed imports.
"""
# Known issues of LazyImport ::
#
#     1) High overhead when called hundreds of thousands times.
#     For example, NumPy must not be imported with LazyImport.

from andes.utils.lazyimport import LazyImport

import math
import coloredlogs         # NOQA
import numpy as np         # NOQA
from numpy import ndarray  # NOQA
from tqdm import tqdm      # NOQA

# Library preference:
# KVXOPT + ipadd > CVXOPT + ipadd > KXVOPT > CVXOPT (+ KLU or not)

try:
    import kvxopt
    from kvxopt import umfpack   # test if shared libs can be found
    from kvxopt import spmatrix as kspmatrix
    KIP_ADD = False
    if hasattr(kspmatrix, 'ipadd'):
        KIP_ADD = True
except ImportError:
    kvxopt = None
    kspmatrix = None
    KIP_ADD = False


from cvxopt import spmatrix as cspmatrix
if hasattr(cspmatrix, 'ipadd'):
    CIP_ADD = True
else:
    CIP_ADD = False


if kvxopt is None or (KIP_ADD is False and CIP_ADD is True):
    from cvxopt import umfpack                           # NOQA
    from cvxopt import spmatrix, matrix, sparse, spdiag  # NOQA
    from cvxopt import mul, div                          # NOQA
    from cvxopt.lapack import gesv                       # NOQA
    try:
        from cvxoptklu import klu  # NOQA
    except ImportError:
        klu = None
    IP_ADD = CIP_ADD
else:
    from kvxopt import umfpack, klu                      # NOQA
    from kvxopt import spmatrix, matrix, sparse, spdiag  # NOQA
    from kvxopt import mul, div                          # NOQA
    from kvxopt.lapack import gesv                       # NOQA
    IP_ADD = KIP_ADD

from andes.utils.texttable import Texttable              # NOQA
from andes.utils.paths import get_dot_andes_path         # NOQA

# --- constants ---

deg2rad = math.pi/180
jac_names = ('fx', 'fy', 'gx', 'gy')
jac_types = ('c', '')

jac_full_names = list()
for jname in jac_names:
    for jtype in jac_types:
        jac_full_names.append(jname + jtype)

# --- lazy import packages ---

pd = LazyImport('import pandas')
cupy = LazyImport('import cupy')
mpl = LazyImport('import matplotlib')
unittest = LazyImport('import unittest')
yaml = LazyImport('import yaml')

plt = LazyImport('from matplotlib import pyplot')
Pool = LazyImport('from pathos.multiprocessing import Pool')
Process = LazyImport('from multiprocess import Process')

newton_krylov = LazyImport('from scipy.optimize import newton_krylov')
fsolve = LazyImport('from scipy.optimize import fsolve')
solve_ivp = LazyImport('from scipy.integrate import solve_ivp')
odeint = LazyImport('from scipy.integrate import odeint')
