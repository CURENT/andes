import importlib
import math

from cvxopt import matrix, sparse
from cvxopt import div
from cvxopt import umfpack

from ..utils import elapsed
from ..utils.jactools import diag0
from ..consts import DEBUG


try:
    from cvxoptklu import klu
    KLU = True
except ImportError:
    KLU = False


lib = umfpack
F = None

solvers = {'NR': 'newton',
           'Newton': 'newton',
           'FDPF': 'fdpf',
           'FDBX': 'fdpf',
           'FDXB': 'fdpf',
           }


def run(system):
    """Entry function of power flow routine

    :return: convergence truth value
    :rtype: bool
    """
    t, s = elapsed()

    # default sparselib setup
    assert system.Settings.sparselib in system.Settings.sparselib_alt, \
        "Invalid sparse library <{}>".format(system.Settings.sparselib)

    if system.Settings.sparselib == 'klu' and KLU:
        globals()['lib'] = klu
    else:
        system.Settings.sparselib = 'umfpack'
        globals()['lib'] = umfpack

    # default solver setup
    if system.SPF.solver not in solvers.keys():
            system.SPF.solver = 'NR'

    func_name = solvers.get(system.SPF.solver)
    run_powerflow = importlib.import_module('andes.routines.powerflow')
    run_powerflow = getattr(run_powerflow, func_name)

    # info print out
    system.Log.info('Power Flow Analysis:')
    system.Log.info('Sparse Solver: ' + system.Settings.sparselib.upper())
    system.Log.info('Solution Method: ' + system.SPF.solver.upper())
    system.Log.info('Flat-start: ' + ('Yes' if system.SPF.flatstart else 'No') + '\n')

    convergence, niter = run_powerflow(system)
    system.status['pf_solved'] = convergence

    if convergence:
        post_processing(system)

    t, s = elapsed(t)

    system.Log.info('Power flow {} in {}'.format(['failed', 'converged'][convergence], s))
    return convergence


def newton(system):
    """
    Standard Newton power flow routine

    :param system: power system instance
    :return: (convergence flag, number of iterations)
    """

    convergence = False
    dae = system.DAE
    iter_mis = []

    # main loop
    niter = 0
    while True:
        inc = calc_inc(system)
        dae.x += inc[:dae.n]
        dae.y += inc[dae.n:dae.n + dae.m]

        niter += 1
        system.SPF.iter = niter

        max_mis = max(abs(inc))
        iter_mis.append(max_mis)
        system.Settings.error = max_mis

        system.Log.info(' Iter{:3d}.  Max. Mismatch = {:8.7f}'.format(niter, max_mis))

        if max_mis < system.Settings.tol:
            convergence = True
            break
        elif niter > 5 and max_mis > 1000 * iter_mis[0]:
            system.Log.info('Blown up in {0} iterations.'.format(niter))
            break
        if niter > system.SPF.maxit:
            system.Log.info('Reached maximum number of iterations.')
            break

    return convergence, niter


def calc_inc(system):
    """
    Calculate variable correction increments for Newton method

    :param system: system instance
    :return: a matrix of variable increment from solving -Ax = b
    """

    global F
    exec(system.Call.newton)

    A = sparse([[system.DAE.Fx, system.DAE.Gx],
                [system.DAE.Fy, system.DAE.Gy]])

    inc = matrix([system.DAE.f, system.DAE.g])

    if system.DAE.factorize:
        F = lib.symbolic(A)
        system.DAE.factorize = False

    try:
        N = lib.numeric(A, F)
        if system.Settings.sparselib.lower() == 'klu':
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
    """
    Fast Decoupled power flow solver routine
    """

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


def post_processing(system):
    """
    Post processing for power flow routine. Computes nodal and line injections.

    :param system: power system object
    :return: None
    """
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
