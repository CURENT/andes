from cvxopt import matrix, sparse, div
from ..utils.jactools import diag0
from ..consts import DEBUG
import importlib
import math

try:
    klu = importlib.import_module('cvxoptklu.klu')
    KLU = True
except:
    KLU = False
umfpack = importlib.import_module('cvxopt.umfpack')
lib = umfpack
F = None

solvers = {'nr': 'newton',
           'newton': 'newton',
           'fdpf': 'fdpf',
           'fdbx': 'fdpf',
           'fdxb': 'fdpf',
           }


def run(system):
    """Entry function of power flow routine"""

    # default sparselib setup
    if system.Settings.sparselib not in system.Settings.sparselib_alt:
        system.Settings.sparselib = 'umfpack'
        globals()['lib'] = umfpack
    elif system.Settings.sparselib == 'klu':
        if not KLU:
            system.Settings.sparselib = 'umfpack'
            globals()['lib'] = umfpack
        else:
            globals()['lib'] = klu

    # default solver setup
    if system.SPF.solver.lower() not in solvers.keys():
            system.SPF.solver = 'NR'

    func_name = solvers.get(system.SPF.solver.lower())
    run_powerflow = importlib.import_module('andes.routines.powerflow')
    run_powerflow = getattr(run_powerflow, func_name)

    convergence, niter = run_powerflow(system)
    if convergence:
        system.SPF.solved = True
        post_processing(system, convergence)


def calcInc(system):
    global F
    exec(system.Call.newton)

    A = sparse([[system.DAE.Fx, system.DAE.Gx], [system.DAE.Fy, system.DAE.Gy]])
    inc = matrix([system.DAE.f, system.DAE.g])

    if system.DAE.factorize:
        F = lib.symbolic(A)
        system.DAE.factorize = False

    # matrix2mat('PF_Gy.mat', [system.DAE.Gy], ['Gy'])

    try:
        N = lib.numeric(A, F)
        if system.Settings.sparselib.lower() == 'klu' and KLU:
            lib.solve(A, F, N, inc)
        elif system.Settings.sparselib.lower() == 'umfpack':
            lib.solve(A, N, inc)
    except ValueError:
        system.Log.warning('Unexpected symbolic factorization. Refactorizing...')
        system.DAE.factorize = True
    except ArithmeticError:
        system.Log.error('Jacobian matrix is singular.')
        diag0(system.DAE.Gy, 'unamey', system)

    return -inc


def fdpf(system):
    """Fast Decoupled power flow solver routine"""

    # sparse library set up
    sparselib = system.Settings.sparselib.lower()

    # general settings
    niter = 1
    iter_max = system.SPF.maxit
    convergence = True
    tol = system.Settings.tol
    system.Settings.error = tol + 1
    err_vec = []
    if (not system.Line.Bp) or (not system.Line.Bpp):
        system.Line.build_b()

    # initialize indexing and Jacobian
    ngen = system.SW.n + system.PV.n
    sw = system.SW.a
    sw.sort(reverse=True)
    no_sw = system.Bus.a[:]
    no_swv = system.Bus.v[:]
    for item in sw:
        no_sw.pop(item)
        no_swv.pop(item)
    gen = system.SW.a + system.PV.a
    gen.sort(reverse=True)
    no_g = system.Bus.a[:]
    no_gv = system.Bus.v[:]
    for item in gen:
        no_g.pop(item)
        no_gv.pop(item)
    Bp = system.Line.Bp[no_sw, no_sw]
    Bpp = system.Line.Bpp[no_g, no_g]

    # F: symbolic, N: numeric
    Fp = lib.symbolic(Bp)
    Fpp = lib.symbolic(Bpp)
    Np = lib.numeric(Bp, Fp)
    Npp = lib.numeric(Bpp, Fpp)
    exec(system.Call.fdpf)

    # main loop
    while system.Settings.error > tol:
        # P-theta
        da = matrix(div(system.DAE.g[no_sw], system.DAE.y[no_swv]))
        if sparselib == 'umfpack':
            lib.solve(Bp, Np, da)
        elif sparselib == 'klu':
            lib.solve(Bp, Fp, Np, da)
        system.DAE.y[no_sw] += da
        exec(system.Call.fdpf)
        normP = max(abs(system.DAE.g[no_sw]))

        # Q-V
        dV = matrix(div(system.DAE.g[no_gv], system.DAE.y[no_gv]))
        if sparselib == 'umfpack':
            lib.solve(Bpp, Npp, dV)
        elif sparselib == 'klu':
            lib.solve(Bpp, Fpp, Npp, dV)
        system.DAE.y[no_gv] += dV
        exec(system.Call.fdpf)
        normQ = max(abs(system.DAE.g[no_gv]))

        err = max([normP, normQ])
        err_vec.append(err)
        system.Settings.error = err

        msg = 'Iter{:4d}.  Max. Mismatch = {:8.7f}'.format(niter, err_vec[-1])
        system.Log.info(msg)
        niter += 1
        system.SPF.iter = niter

        if niter > 4 and err_vec[-1] > 1000 * err_vec[0]:
            system.Log.info('Blown up in {0} iterations.'.format(niter))
            convergence = False
            break

        if niter > iter_max:
            system.Log.info('Reached maximum number of iterations.')
            convergence = False
            break

    return convergence, niter


def newton(system):
    """newton power flow routine"""
    niter = 0
    iter_max = system.SPF.maxit
    convergence = False
    tol = system.Settings.tol
    system.Settings.error = tol + 1
    err_vec = []
    # main loop
    while system.Settings.error > tol:
        inc = calcInc(system)
        system.DAE.x += inc[:system.DAE.n]
        system.DAE.y += inc[system.DAE.n:system.DAE.n + system.DAE.m]

        niter += 1
        system.SPF.iter = niter
        system.Settings.error = max(abs(inc))
        err_vec.append(system.Settings.error)

        msg = 'Iter{:4d}.  Max. Mismatch = {:8.7f}'.format(niter, system.Settings.error)
        system.Log.info(msg)

        if niter > 4 and err_vec[-1] > 1000 * err_vec[0]:
            system.Log.info('Blown up in {0} iterations.'.format(niter))
            break
        if niter > iter_max:
            system.Log.info('Reached maximum number of iterations.')
            break

    if err_vec[-1] < tol:
        convergence = True

    return convergence, niter


def post_processing(system, convergence):
    if convergence:
        exec(system.Call.pfload)
        system.Bus.Pl = system.DAE.g[system.Bus.a]
        system.Bus.Ql = system.DAE.g[system.Bus.v]

        exec(system.Call.pfgen)
        system.Bus.Pg = system.DAE.g[system.Bus.a]
        system.Bus.Qg = system.DAE.g[system.Bus.v]

        if system.PV.n:
            system.PV.qg = system.DAE.y[system.PV.q]
        if system.SW.n:
            system.SW.pg = system.DAE.y[system.SW.p]
            system.SW.qg = system.DAE.y[system.SW.q]

        exec(system.Call.seriesflow)

        system.Area.seriesflow(system.DAE)
