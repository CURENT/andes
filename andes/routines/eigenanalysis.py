import numpy.linalg
from cvxopt.klu import linsolve
from cvxopt.lapack import gesv
from cvxopt import matrix, spmatrix, mul, div, sparse
from math import ceil
from ..formats.txt import dump_data
from ..consts import *
from matplotlib.pyplot import *
import scipy
from scipy.sparse import linalg

try:
    from petsc4py import PETSc
    PETSC = True
except:
    PETSC = False
try:
    from slepc4py import SLEPc
    import slepc4py
    SLEPC = True
except:
    SLEPC = False


def state_matrix(system):
    """Return state matrix"""
    Gyx = matrix(system.DAE.Gx)
    linsolve(system.DAE.Gy, Gyx)
    return system.DAE.Fx - system.DAE.Fy*Gyx


def eigs(As):
    """Solve eigenvalues from state matrix"""

    # if (not SLEPC) or (not SLEPC):
    return numpy.linalg.eig(As)
    # try:

    As = sparse(As)
    As = scipy.sparse.csc_matrix((numpy.array(As.CCS[2]).flatten(),
                                      numpy.array(As.CCS[1]).flatten(),
                                      numpy.array(As.CCS[0]).flatten()
                                      ), shape=As.size)
    # return linalg.eigs(As, k=1326)
    As_csr = As.tocsr()

    opts = PETSc.Options()
    A = PETSc.Mat().createAIJ(size=As_csr.shape,
                              csr=(As_csr.indptr, As_csr.indices, As_csr.data))
    A.setFromOptions()
    A.setUp()

    xr, tmp = A.getVecs()
    xi, tmp = A.getVecs()

    # Setup the eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(A, None)
    E.setDimensions(500, PETSc.DECIDE)
    E.setProblemType( SLEPc.EPS.ProblemType.GNHEP )
    E.setFromOptions()

    # Solve the eigensystem
    E.solve()

    Print = PETSc.Sys.Print
    Print("")
    its = E.getIterationNumber()
    Print("Number of iterations of the method: %i" % its)
    sol_type = E.getType()
    Print("Solution method: %s" % sol_type)
    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %i" % nev)
    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    nconv = E.getConverged()
    Print("Number of converged eigenpairs: %d" % nconv)
    if nconv > 0:
        Print("")
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, xr, xi)
            error = E.computeError(i)
            if k.imag != 0.0:
              Print(" %9f%+9f j  %12g" % (k.real, k.imag, error))
            else:
              Print(" %12f       %12g" % (k.real, error))
        Print("")
    # except:
    #     return numpy.linalg.eigvals(As)


def part_factor(As):
    """Compute participation factor of states in eigenvalues"""
    mu, N = eigs(As)
    N = matrix(N)
    n = len(mu)
    idx = range(n)
    W = matrix(spmatrix(1.0, idx, idx, (n, n), N.typecode))
    gesv(N, W)
    partfact = mul(abs(W.T), abs(N))
    b = matrix(1.0, (1, n))
    WN = b * partfact
    partfact = partfact.T

    for item in idx:
        mu_real = mu[item].real
        mu_imag = mu[item].imag
        mu[item] = complex(round(mu_real, 4), round(mu_imag, 4))
        partfact[item, :] /= WN[item]

    # participation factor:
    return matrix(mu), matrix(partfact)


def dump_results(system, mu, partfact):
    """Format results"""
    if system.Files.no_output:
        return

    text = []
    header = []
    rowname = []
    data = []

    neig = len(mu)
    mu_real = mu.real()
    mu_imag = mu.imag()
    npositive = sum(1 for x in mu_real if x > 0)
    nzero = sum(1 for x in mu_real if x == 0)
    nnegative = sum(1 for x in mu_real if x < 0)

    numeral = []
    for idx, item in enumerate(range(neig)):
        if mu_real[idx] == 0:
            marker = '*'
        elif mu_real[idx] > 0:
            marker = '**'
        else:
            marker = ''
        numeral.append('#' + str(idx+1) + marker)

    # compute frequency, undamped frequency and damping
    freq = [0] * neig
    ufreq = [0] * neig
    damping = [0] * neig
    for idx, item in enumerate(mu):
        if item.imag == 0:
            freq[idx] = 0
            ufreq[idx] = 0
            damping[idx] = 0
        else:
            freq[idx] = abs(item)/2/pi
            ufreq[idx] = abs(item.imag/2/pi)
            damping[idx] = - div(item.real, abs(item)) * 100

    # obtain most associated variables
    var_assoc = []
    for prow in range(neig):
        temp_row = partfact[prow, :]
        name_idx = list(temp_row).index(max(temp_row))
        var_assoc.append(system.VarName.unamex[name_idx])

    pf = []
    for prow in range(neig):
        temp_row = []
        for pcol in range(neig):
            temp_row.append(round(partfact[prow, pcol], 5))
        pf.append(temp_row)

    text.append(system.Report.info)
    header.append([''])
    rowname.append(['EIGENVALUE ANALYSIS REPORT'])
    data.append('')

    text.append('STATISTICS\n')
    header.append([''])
    rowname.append(['Positives', 'Zeros', 'Negatives'])
    data.append([npositive, nzero, nnegative])

    text.append('EIGENVALUE DATA\n')
    header.append(['Most Associated', 'Real', 'Imag', 'Damped Freq.', 'Frequency', 'Damping [%]'])
    rowname.append(numeral)
    data.append([var_assoc, list(mu_real), list(mu_imag), ufreq, freq, damping])

    cpb = 7  # columns per block
    nblock = int(ceil(neig / cpb))

    if nblock <= 10:
        for idx in range(nblock):
            start = cpb*idx
            end = cpb*(idx+1)
            text.append('PARTICIPATION FACTORS [{}/{}]\n'.format(idx+1, nblock))
            header.append(numeral[start:end])
            rowname.append(system.VarName.unamex)
            data.append(pf[start:end])

    dump_data(text, header, rowname, data, system.Files.eig)
    system.Log.info('Report saved.')


def run(system):
    if not system.SPF.solved:
        system.Log.warning('Power flow not solved. Eigenvalue analysis will not continue.')
        return True

    system.DAE.factorize = True
    exec(system.Call.int)

    As = state_matrix(system)
    if not As:
        system.Log.info('No dynamic model loaded. Eivgenvalue analysis will not continue.')
        return False
    else:
        system.Log.info('System dynamic order: {:g}'.format(system.DAE.n))
        mu, pf = part_factor(As)
        dump_results(system, mu, pf)
        mu_real = mu.real()
        mu_imag = mu.imag()
        p_mu_real, p_mu_imag, z_mu_real, z_mu_imag, n_mu_real, n_mu_imag = [], [], [], [], [], []
        for re, im in zip(mu_real, mu_imag):
            if re == 0:
                z_mu_real.append(re)
                z_mu_imag.append(im)
            elif re > 0:
                p_mu_real.append(re)
                p_mu_imag.append(im)
            elif re < 0:
                n_mu_real.append(re)
                n_mu_imag.append(im)

        if len(p_mu_real) > 0:
            system.Log.warning('System is not stable due to {} positive eigenvalues.'.format(len(p_mu_real)))
        else:
            system.Log.info('System is small-signal stable in the initial neighbourhood.')

        if system.SSSA.plot:
            fig, ax = subplots()
            ax.scatter(n_mu_real, n_mu_imag, marker='x', s=26, color='green')
            ax.scatter(z_mu_real, z_mu_imag, marker='o', s=26, color='orange')
            ax.scatter(p_mu_real, p_mu_imag, marker='x', s=26, color='red')
            show()
        return True
