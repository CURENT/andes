import sys
import importlib

import progressbar

from math import isnan
from numpy import array
from time import monotonic as time, sleep

from cvxopt import matrix, spmatrix, sparse, spdiag
from cvxopt.klu import numeric, symbolic, solve, linsolve

from ..utils.jactools import *

from .eigenanalysis import run as run_eig
try:
    from ..utils.matlab import write_mat
except:
    pass
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
    if not system.SPF.solved:
        system.Log.warning('Power flow not solved. Time domain simulation will not continue.')
        return False

    global F
    retval = True
    bar = None
    if system.pid == -1 and system.Settings.progressbar:
        bar = progressbar.ProgressBar(
                                      widgets=[' [', progressbar.Percentage(), progressbar.Bar(),
                                               progressbar.AdaptiveETA(), '] '])
    dae = system.DAE
    settings = system.TDS

    # check settings
    maxit = settings.maxit
    qrt = settings.qrt
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
    if system.TDS.compute_flows:
        dae.init_fg()
        compute_flows(system)
    system.VarOut.store(t)

    if system.Settings.dime_enable:
        system.Streaming.sync_and_handle()
        system.Streaming.vars_to_modules()

    # perturbation file
    PERT = 0  # 0 - not loaded, 1 - loaded, -1 - error
    callpert = None
    if system.Files.pert:
        try:
            sys.path.append(system.Files.path)
            module = importlib.import_module(system.Files.pert[:-3])
            callpert = getattr(module, 'pert')
            PERT = 1
        except:
            PERT = -1

    # main loop
    if bar:
        bar.start()
    rt_headroom = 0
    settings.qrtstart = time()
    t_jac = -1
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
            system.Fault.check_time(actual_time)
            system.Breaker.check_time(actual_time)
            dae.rebuild = True
            switch = False
        else:
            dae.rebuild = False

        if PERT == 1:  # pert file loaded
            callpert(actual_time, system)
        elif PERT == -1:
            system.Log.warning('Pert file is discarded due to import errors.')
            PERT = 0

        niter = 0
        settings.error = tol + 1
        t = actual_time

        while settings.error > tol and niter < maxit:
            if settings.method == 'fwdeuler':
                # predictor of x
                exec(system.Call.int_f)
                f0 = dae.f
                dae.x = xa + h * f0
                inc = calcInc(system)
                dae.y += inc

                # corrector step
                exec(system.Call.int_f)
                dae.x = xa + 0.5 * h * (f0 + dae.f)
                inc = calcInc(system)
                dae.y += inc

                settings.error = abs(max(inc))
                niter += 1

            elif settings.method in ['euler', 'trapezoidal']:

                if actual_time - t_jac >= 1:
                    dae.rebuild = True
                    t_jac = actual_time
                elif niter > 3:
                    dae.rebuild = True
                elif dae.factorize:
                    dae.rebuild = True

                if dae.rebuild:
                    try:
                        exec(system.Call.int)
                    except OverflowError:
                        system.Log.error('Data overflow. Convergence is not likely.')
                        t = settings.tf + 1
                        retval = False
                        break
                else:
                    exec(system.Call.int_fg)

                # complete Jacobian matrix DAE.Ac
                if settings.method == 'euler':
                    dae.Ac = sparse([[In - h*dae.Fx, dae.Gx],
                                     [   - h*dae.Fy, dae.Gy]], 'd')
                    dae.q = dae.x - xa - h*dae.f
                elif settings.method == 'trapezoidal':  # use implicit trapezoidal method by default
                    dae.Ac = sparse([[In - h*0.5*dae.Fx, dae.Gx],
                                     [   - h*0.5*dae.Fy, dae.Gy]], 'd')
                    dae.q = dae.x - xa - h*0.5*(dae.f + fn)

                # windup limiters
                if dae.rebuild:
                    dae.reset_Ac()

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

                # for debugging
                if niter > 15:
                    pass
                inc_x = inc[:dae.n]
                inc_y = inc[dae.n: dae.m+dae.n]
                dae.x += inc_x
                dae.y += inc_y

                settings.error = max(abs(inc))
                if isnan(settings.error):
                    t = settings.tf + 1
                    system.Log.error('Iteration error: NaN detected.')
                    retval = False
                    break
                niter += 1

        if niter >= maxit:
            inc_g = inc[dae.n: dae.m+dae.n]
            max_g_err_sign = 1 if abs(max(inc_g)) > abs(min(inc_g)) else -1
            if max_g_err_sign == 1:
                max_g_err_idx = list(inc_g).index(max(inc_g))
            else:
                max_g_err_idx = list(inc_g).index(min(inc_g))
            system.Log.debug('Maximum mismatch = {:.4g} at equation <{}>'.format(max(abs(inc_g)), system.VarName.unamey[max_g_err_idx]))

            h = time_step(system, False, niter, t)
            system.Log.debug('Reducing time step h={:.4g}s for t={:.4g}'.format(h, t))
            dae.x = matrix(xa)
            dae.y = matrix(ya)
            dae.f = matrix(fn)
            continue

        # update output variables and time step
        step += 1
        h = time_step(system, True, niter, t)

        compute_flows(system)

        system.VarOut.store(t)
        if system.Settings.dime_enable:
            system.Streaming.sync_and_handle()
            system.Streaming.vars_to_modules()

        # plot variables and display iteration status
        perc = max(min((t - settings.t0) / (settings.tf - settings.t0) * 100, 100), 0)
        if bar:
            bar.update(perc)

        if perc > nextpc:
            system.Log.debug(' * Simulation time = {:.4f}s, step = {}, max mismatch = {:.4f}, niter = {}'.format(t, step, settings.error, niter))
            nextpc += 10
        # compute max rotor angle difference
        diff_max = anglediff()

        # quasi-real-time check and wait
        rt_end = settings.qrtstart + (t - settings.t0) * settings.kqrt
        if settings.qrt:
            if time() - rt_end > 0:  # the ending time has passed
                if time() - rt_end > settings.kqrt:  # simulation is too slow
                    system.Log.warning('Simulation over-run at simulation time {:4.4g} s.'.format(t))
            else:  # wait to finish
                rt_headroom += (rt_end - time())
                while time() - rt_end < 0:
                    sleep(1e-4)
    if bar:
        bar.finish()
    if settings.qrt:
        system.Log.debug('Quasi-RT headroom time: {} s.'.format(str(rt_headroom)))
    if t != settings.tf:
        system.Log.error('Reached minimum time step. Convergence is not likely.')
        retval = False

    return retval


def time_step(system, convergence, niter, t):
    """determine the time step during time domain simulations
        convergence: 1 - last step computation converged
                     0 - last step not converged
        niter:  number of iterations """
    settings = system.TDS
    if convergence:
        if niter >= 15:
            settings.deltat = max(settings.deltat * 0.5, settings.deltatmin)
        elif niter <= 8:
            settings.deltat = min(settings.deltat * 1.1, settings.deltatmax)
        else:
            settings.deltat = max(settings.deltat * 0.9, settings.deltatmin)
        if settings.fixt:  # adjust fixed time step if niter is high
            settings.deltat = min(settings.tstep, settings.deltat)
    else:
        settings.deltat *= 0.9
        if settings.deltat < settings.deltatmin:
            settings.deltat = 0

    if system.Fault.istime(t) or system.Breaker.istime(t):
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


def compute_flows(system):
    if system.TDS.compute_flows:
        # compute and append series injections on buses
        dae = system.DAE

        exec(system.Call.bus_injection)
        bus_inj = dae.g[:2 * system.Bus.n]

        exec(system.Call.seriesflow)
        dae.y = matrix([dae.y, bus_inj, system.Line._line_flows])
