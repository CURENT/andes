"""
Shared constants and delayed imports.

This module imports shared libraries either directly or with `LazyImport`.

`LazyImport` shall only be used to imported
"""
# Known issues of LazyImport ::
#
#     1) High overhead when called hundreds of thousands times.
#     For example, NumPy must not be imported with LazyImport.

import math
import os
import coloredlogs         # NOQA
import numpy as np         # NOQA

from andes.utils.lazyimport import LazyImport
from distutils.spawn import find_executable

# Library preference:
# KVXOPT + ipadd > CVXOPT + ipadd > KXVOPT > CVXOPT (+ KLU or not)

try:
    import kvxopt
    from kvxopt import umfpack   # test if shared libs can be found
    from kvxopt import spmatrix as kspmatrix
    KIP_ADD = True
except ImportError:
    kvxopt = None
    kspmatrix = None
    KIP_ADD = False

if KIP_ADD is False:
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
    from cvxopt import printing                          # NOQA
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
    from kvxopt import printing                          # NOQA
    IP_ADD = KIP_ADD

printing.options['dformat'] = '%.1f'
printing.options['width'] = -1

from andes.utils.texttable import Texttable              # NOQA
from andes.utils.paths import get_dot_andes_path         # NOQA

# --- constants ---

deg2rad = math.pi/180
jac_names = ('fx', 'fy', 'gx', 'gy')
jac_types = ('c', '')

dilled_vars = ['f_args', 'g_args', 'j_args', 's_args',
               'ia_args', 'ii_args', 'ij_args',
               'ijac', 'jjac', 'vjac', 'j_names',
               'init_seq']

jac_full_names = list()
for jname in jac_names:
    for jtype in jac_types:
        jac_full_names.append(jname + jtype)

# --- lazy import packages ---
tqdm = LazyImport('from tqdm import tqdm')

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


# --- Shared functions ---

def set_latex():
    """
    Enables LaTeX for matplotlib based on the `with_latex` option and `dvipng` availability.

    Returns
    -------
    bool
        True for LaTeX on, False for off
    """

    if find_executable('dvipng'):
        mpl.rc('text', usetex=True)

        no_warn_file = os.path.join(get_dot_andes_path(), '.no_warn_latex')
        if not os.path.isfile(no_warn_file):
            print('Using LaTeX for rendering. If an error occurs:')
            print('a) If you are using `andes plot`, disable with option "-d",')
            print('b) If you are using `plot()`, set "latex=False".')

            try:
                with open(os.path.join(get_dot_andes_path(), '.no_warn_latex'), 'w') as f:
                    f.write('0')
            except OSError:
                pass

        return True

    return False
