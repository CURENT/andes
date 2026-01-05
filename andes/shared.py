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

import coloredlogs  # NOQA
import numpy as np  # NOQA
import psutil
from shutil import which  # NOQA
from kvxopt import printing  # NOQA
from kvxopt import (div, klu, matrix, mul, sparse, spdiag, spmatrix,  # NOQA
                    umfpack)
from kvxopt.lapack import gesv  # NOQA

from andes.utils.lazyimport import LazyImport

printing.options['dformat'] = '%.1f'
printing.options['width'] = -1


# --- SYSTEM INFO ---
NCPUS_PHYSICAL = psutil.cpu_count(logical=False)

# --- MATH CONSTANTS ---
deg2rad = math.pi/180
rad2deg = 180 / math.pi
pi2o3 = math.pi * 2 / 3
sqrt3 = math.sqrt(3)
isqrt3 = math.sqrt(1/3)

# --- NAME CONSTANTS ---
jac_names = ('fx', 'fy', 'gx', 'gy')
jac_types = ('c', '')

dilled_vars = ['f_args', 'g_args', 'j_args', 's_args', 'sns_args',
               'ia_args', 'ii_args', 'ij_args',
               'ijac', 'jjac', 'vjac',
               'j_names', 'init_seq', "need_diag_eps"]

jac_full_names = list()
for jname in jac_names:
    for jtype in jac_types:
        jac_full_names.append(jname + jtype)

# --- lazy import packages ---
tqdm = LazyImport('from tqdm import tqdm')
tqdm_nb = LazyImport('from tqdm.notebook import tqdm')


def get_tqdm(no_tqdm=False):
    """
    Get the appropriate tqdm progress bar for the current environment.

    Returns a tuple of (tqdm_class, disable_flag) where:
    - tqdm_class: tqdm or tqdm_nb depending on environment
    - disable_flag: True if progress bar should be disabled

    Environment behavior:
    - Terminal (TTY): CLI tqdm, enabled
    - Interactive Jupyter: tqdm_nb widget, enabled
    - Batch execution (nbconvert, scripts): CLI tqdm, disabled

    Parameters
    ----------
    no_tqdm : bool
        If True, always disable the progress bar.

    Returns
    -------
    tuple
        (tqdm_class, disable_flag)
    """
    from andes.utils.misc import has_notebook_frontend, is_tty

    if no_tqdm:
        return tqdm, True

    if has_notebook_frontend():
        # Interactive Jupyter with working widgets
        return tqdm_nb, False
    elif is_tty():
        # Terminal with TTY
        return tqdm, False
    else:
        # Batch execution (nbconvert, Sphinx, scripts) - disabled
        return tqdm, True


pd = LazyImport('import pandas')
numba = LazyImport('import numba')
cupy = LazyImport('import cupy')
mpl = LazyImport('import matplotlib')
unittest = LazyImport('import unittest')
pandapower = LazyImport('import pandapower')
pypowsybl = LazyImport('import pypowsybl')
GridCal_Engine = LazyImport('from GridCal import Engine')

plt = LazyImport('from matplotlib import pyplot')
Pool = LazyImport('from pathos.multiprocessing import Pool')
Process = LazyImport('from multiprocess import Process')

newton_krylov = LazyImport('from scipy.optimize import newton_krylov')
fsolve = LazyImport('from scipy.optimize import fsolve')
solve_ivp = LazyImport('from scipy.integrate import solve_ivp')
odeint = LazyImport('from scipy.integrate import odeint')

Oct2PyError = LazyImport('from oct2py import Oct2PyError')
