import importlib
from cvxopt.base import matrix, spmatrix, sparse, div

klu = importlib.import_module('cvxopt.klu')
umfpack = importlib.import_module('cvxopt.umfpack')
F = None

all_solvers = {'nr': 'newton',
               'newton': 'newton',
               'fdpf': 'fdpf',
               'fdbx': 'fdpf',
               'fdxb': 'fdpf',
              }


def run(system):
    if system.PF.solver.lower() not in all_solvers.keys():
        system.PF.solver = 'NR'
    func_name = all_solvers.get(system.PF.solver.lower())
    run_pf = func_name + '(system)'
    eval(run_pf)


def calcInc(system):
    global F
    exec(system.Calls.newton)

    A = sparse([[system.DAE.Fx, system.DAE.Gx], [system.DAE.Fy, system.DAE.Gy]])
    inc = matrix([system.DAE.f, system.DAE.g])
    sparselib = system.Settings.sparselib.lower()
    if sparselib == 'umfpack':
        lib = umfpack
    elif sparselib == 'klu':
        lib = klu
    else:
        # use UMFPACK as default for official CVXOPT compatibility
        sparselib = 'umfpack'
        lib = umfpack

    if system.DAE.factorize:
        F = lib.symbolic(A)
        system.DAE.factorize = False

    # matrix2mat('PF_Gy.mat', [system.DAE.Gy], ['Gy'])

    try:
        N = lib.numeric(A, F)
        if sparselib == 'klu':
            lib.solve(A, F, N, inc)
        elif sparselib == 'umfpack':
            lib.solve(A, N, inc)
    except:
        system.Log.warning('Unexpected symbolic factorization')
        system.DAE.factorize = True
    return -inc


def fdpf(system):
    """Fast Decoupled power flow solver routine"""

    # Sparse library set up
    sparselib = system.Settings.sparselib.lower()
    if sparselib == 'umfpack':
        lib = umfpack
    elif sparselib == 'klu':
        lib = klu
    else:
        sparselib = 'klu'
        lib = klu

    # General settings
    iteration = 1
    iter_max = system.PF.maxit
    convergence = True
    tol = system.Settings.tol
    system.Settings.error = tol + 1
    err_vec = []

    # Initialization indexing and Jacobian
    ngen = system.SW.n + system.PV.n
    sw = system.SW._geta()
    sw.sort(reverse=True)
    no_sw = system.Bus.a[:]
    no_swv = system.Bus.v[:]
    for item in sw:
        no_sw.pop(item)
        no_swv.pop(item)
    gen = system.SW._geta() + system.PV._geta()
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
    exec(system.Calls.fdpf)

    # Main loop
    while system.Settings.error > tol:
        # P-theta
        da = matrix(div(system.DAE.g[no_sw], system.DAE.y[no_swv]))
        if sparselib == 'umfpack':
            lib.solve(Bp, Np, da)
        elif sparselib == 'klu':
            lib.solve(Bp, Fp, Np, da)
        system.DAE.y[no_sw] += da
        exec(system.Calls.fdpf)
        normP = max(abs(system.DAE.g[no_sw]))

        # Q-V
        dV = matrix(div(system.DAE.g[no_gv], system.DAE.y[no_gv]))
        if sparselib == 'umfpack':
            lib.solve(Bpp, Npp, dV)
        elif sparselib == 'klu':
            lib.solve(Bpp, Fpp, Npp, dV)
        system.DAE.y[no_gv] += dV
        exec(system.Calls.fdpf)
        normQ = max(abs(system.DAE.g[no_gv]))

        err = max([normP, normQ])
        err_vec.append(err)
        system.Settings.error = err

        msg = 'Iter{:4d}.  Max. Mismatch = {:8.7f}'.format(iteration, err_vec[-1])
        system.Log.info(msg)
        iteration += 1
        system.PF.iter = iteration

        # Stop if error increases too much
        if iteration > 4 and err_vec[-1] > 1000*err_vec[0]:
            # stop if the error increases too much
            system.Log.info('The error is increasing too much.')
            system.Log.info('Convergence is likely not reachable.')
            convergence = False
            break
        # Or maximum iterations reached
        if iteration > iter_max:
            system.Log.info('Reached maximum number of iterations.')
            convergence = False
            break

    if convergence:
        post_processing(system, convergence)

    return convergence, iteration


def newton(system):
    """newton power flow routine"""
    iteration = 1
    iter_max = system.PF.maxit
    convergence = True
    tol = system.Settings.tol
    system.Settings.error = tol + 1
    err_vec = []
    # main loop
    while system.Settings.error > tol:
        inc = calcInc(system)
        system.DAE.y += inc

        system.Settings.error = max(abs(inc))
        err_vec.append(system.Settings.error)

        msg = 'Iter{:4d}.  Max. Mismatch = {:8.7f}'.format(iteration, system.Settings.error)
        system.Log.info(msg)
        iteration += 1
        system.PF.iter = iteration

        if iteration > 4 and err_vec[-1] > 1000*err_vec[0]:
            # stop if the error increases too much
            system.Log.info('The error is increasing too much.')
            system.Log.info('Convergence is likely not reachable.')
            convergence = False
            break

        if iteration > iter_max:
            system.Log.info('Reached maximum number of iterations.')
            convergence = False
            break

    if convergence:
        post_processing(system, convergence)

    return convergence, iteration


def post_processing(system, convergence):
    if convergence:
        exec(system.Calls.pfload)
        system.Bus.Pl = system.DAE.g[system.Bus.a]
        system.Bus.Ql = system.DAE.g[system.Bus.v]

        exec(system.Calls.pfgen)
        system.Bus.Pg = system.DAE.g[system.Bus.a]
        system.Bus.Qg = system.DAE.g[system.Bus.v]

        system.PV.qg = system.DAE.y[system.PV.q]
        system.SW.pg = system.DAE.y[system.SW.p]
        system.SW.qg = system.DAE.y[system.SW.q]

        # todo: update algebraic variables, including PV/SW generations, bus voltage and angle
        # todo: set qg, a for PV, set pg, qg, a for SW!!

        exec(system.Calls.seriesflow)
        system.Settings.pfsolved = True
