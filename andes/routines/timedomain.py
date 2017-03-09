from cvxopt import matrix, spmatrix, sparse, spdiag
from cvxopt.klu import numeric, symbolic, solve, linsolve
from ..utils.jactools import *
import progressbar
F = []


def first_time_step(system):
    """Compute the first time step"""
    settings = system.TDS
    if not system.DAE.n:
        freq = 1.0
    elif system.DAE.n == 1:
        B = matrix(system.DAE.Gx)
        linsolve(system.DAE.Gy, B)
        As = system.DAE.Fx - system.DAE.Fy*B
        freq = abs(As[0, 0])
    else:
        freq = 20.0

    if freq > system.Settings.freq:
        freq = float(system.Settings.freq)

    tspan = abs(settings.tf - settings.t0)
    tcycle = 1 / freq
    settings.deltatmax = min(5 * tcycle, tspan / 100.0)
    settings.deltat = min(tcycle, tspan / 100.0)
    settings.deltatmin = min(tcycle / 64, settings.deltatmax / 20)

    if settings.fixt:
        if settings.tstep <= 0:
            system.Log.warning('Fixed time step is negative or zero')
            system.Log.warning('Switching to automatic time step')
            settings.fixt = False
        else:
            settings.deltat = settings.tstep
            if settings.tstep < settings.deltatmin:
                system.Log.warning('Fixed time step is below the estimated minimum')
    return settings.deltat


def run(system):
    """Entry function of Time Domain Simulation"""
    global F
    bar = progressbar.ProgressBar(redirect_stdout=True,
                                  widgets=[' [', progressbar.Percentage(), progressbar.Bar(),
                                           progressbar.AdaptiveETA(), '] '])
    dae = system.DAE
    settings = system.TDS

    # check settings
    maxit = settings.maxit
    tol = settings.tol
    In = spdiag([1] * dae.n)

    # initialization
    t = settings.t0
    step = 0
    inc = matrix(0, (dae.m + dae.n, 1), 'd' )
    dae.factorize = True
    dae.mu = 1.0
    dae.kg = 0.0
    switch = 0
    nextpc = 0.1
    h = first_time_step(system)

    # time vector for faults and breaker events
    fixed_times = system.Call.get_times()

    # compute max rotor angle difference
    diff_max = anglediff()

    # store the initial value
    system.VarOut.store(t)

    # main loop
    bar.start()
    while t <= settings.tf and t + h > t and not diff_max:

        # last time step length
        if t + h > settings.tf:
            h = settings.tf - t

        # unable to converge and
        if h == 0.0:
            break

        actual_time = t + h

        # check for the occurrence of a disturbance
        for item in fixed_times:
            if (item > t) and (item < t+h):
                actual_time = item
                h = actual_time - t
                switch = True
                break

        # set global time
        system.DAE.t = actual_time

        # backup actual variables
        xa = matrix(dae.x)
        ya = matrix(dae.y)
        fn = matrix(dae.f)

        # apply fixed_time interventions and perturbations
        if switch:
            system.Fault.checktime(actual_time)
            # system.Breaker.get_times(actual_time)
            switch = False

        if settings.disturbance:
            system.Call.disturbance(actual_time)

        niter = 0
        settings.error = tol + 1

        while settings.error > tol and niter < maxit:
            if settings.method == 'fwdeuler':
                # predictor of x
                exec(system.Call.int_f)
                f0 = dae.f
                dae.x = xa + h * f0
                dae.y += calcInc(system)

                # corrector step
                exec(system.Call.int_f)
                dae.x = xa + 0.5 * h * (f0 + dae.f)
                inc = calcInc(system)
                dae.y += inc

                settings.error = abs(max(inc))
                niter += 1

            elif settings.method in ['euler', 'trapezoidal']:
                exec(system.Call.int)

                # complete Jacobian matrix DAE.Ac
                if settings.method == 'euler':
                    dae.Ac = sparse([[In - h*dae.Fx, dae.Gx],
                                     [   - h*dae.Fy, dae.Gy]], 'd')
                    dae.q = dae.x - xa - h*dae.f
                elif settings.method == 'trapezoidal':  # use implicit trapezoidal method by default
                    dae.Ac = sparse([[In - h*0.5*dae.Fx, dae.Gx],
                                     [   - h*0.5*dae.Fy, dae.Gy]], 'd')
                    dae.q = dae.x - xa - h*0.5*(dae.f + fn)

                # anti-windup limiters
                #     exec(system.Call.windup)

                if dae.factorize:
                    F = symbolic(dae.Ac)
                    dae.factorize = False
                inc = -matrix([dae.q, dae.g])

                try:
                    N = numeric(dae.Ac, F)
                    solve(dae.Ac, F, N, inc)
                except ArithmeticError:
                    system.Log.error('Singular matrix')
                    niter = maxit + 1  # force quit
                    diag0(dae.Gy, 'unamey', system)
                    diag0(dae.Fx, 'unamex', system)
                except ValueError:
                    system.Log.warning('Unexpected symbolic factorization')
                    F = symbolic(dae.Ac)
                    try:
                        N = numeric(dae.Ac, F)
                        solve(dae.Ac, F, N, inc)
                    except ArithmeticError:
                        system.Log.error('Singular matrix')
                        niter = maxit + 1
                dae.x += inc[:dae.n]
                dae.y += inc[dae.n: dae.m+dae.n]
                settings.error = max(abs(inc))
                niter += 1

        if niter >= maxit:
            h = time_step(system, False, niter, t)
            system.Log.debug('Reducing time step (delta t={:.5g}s)'.format(h))
            dae.x = matrix(xa)
            dae.y = matrix(ya)
            dae.f = matrix(fn)
            continue

        # update output variables and time step
        t = actual_time
        step += 1
        system.VarOut.store(t)
        h = time_step(system, True, niter, t)

        # plot variables and display iteration status
        perc = (t - settings.t0) / (settings.tf - settings.t0) * 100
        bar.update(perc)

        # compute max rotor angle difference
        diff_max = anglediff()

    bar.finish()

def time_step(system, convergence, niter, t):
    """determine the time step during time domain simulations
        convergence: 1 - last step computation converged
                     0 - last step not converged
        niter:  number of iterations """
    settings = system.TDS
    if convergence:
        if niter >= 8:
            settings.deltat = max(settings.deltat * 0.5, settings.deltatmin)
        elif niter <= 3:
            settings.deltat = min(settings.deltat * 1.1, settings.deltatmax)
        else:
            settings.deltat = max(settings.deltat * 0.9, settings.deltatmin)
        if settings.fixt:  # adjust fixed time step if niter is high
            settings.deltat = min(settings.tstep, settings.deltat)
    else:
        settings.deltat *= 0.9
        if settings.deltat < settings.deltatmin:
            settings.deltat = 0

    if system.Fault.istime(t):
        settings.deltat = min(settings.deltat, 0.002778)
    if settings.method == 'fwdeuler':
        settings.deltat = min(settings.deltat, settings.tstep)

    return settings.deltat


def anglediff():
    """Compute angle difference"""
    return False


def calcInc(system):
    """Calculate algebraic variable increment"""
    global F
    exec(system.Call.int_g)

    A = system.DAE.Gy
    inc = system.DAE.g

    if system.DAE.factorize:
        F = symbolic(A)
        system.DAE.factorize = False

    try:
        N = numeric(A, F)
        solve(A, F, N, inc)
    except ValueError:
        system.Log.warning('Unexpected symbolic factorization. Refactorizing...')
        F = symbolic(dae.Ac)
        try:
            N = numeric(dae.Ac, F)
            solve(dae.Ac, F, N, inc)
        except ArithmeticError:
            system.Log.error('Singular matrix')
            niter = maxit + 1
    except ArithmeticError:
        system.Log.error('Jacobian matrix is singular.')
        diag0(system.DAE.Gy, 'unamey', system)
    return -inc
