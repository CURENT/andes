import numpy.linalg
from cvxopt.klu import linsolve
from cvxopt.lapack import gesv
from cvxopt import matrix, spmatrix, mul
from ..formats.txt import dump_data


def state_matrix(system):
    """Return state matrix"""
    Gyx = matrix(system.DAE.Gx)
    linsolve(system.DAE.Gy, Gyx)
    return matrix(system.DAE.Fx - system.DAE.Fy*Gyx)


def eigs(As):
    """Solve eigenvalues from state matrix"""
    return numpy.linalg.eigvals(As)


def part_factor(As):
    """Compute participation factor of states in eigenvalues"""
    mu, N = numpy.linalg.eig(As)
    N = matrix(N)
    n = len(mu)
    idx = range(n)
    W = matrix(spmatrix(1.0, idx, idx, (n, n), N.typecode))
    gesv(N, W)
    partfact = mul(abs(W.T), abs(N))
    b = matrix(1.0, (1, n))
    WN = b * partfact
    partfact = partfact.T

    mu_real = []
    mu_imag = []
    for item in idx:
        mu_real = mu[item].real
        mu_imag = mu[item].imag
        mu[item] = complex(round(mu_real, 5), round(mu_imag, 5))
        partfact[item, :] /= WN[item]

    return mu, partfact


def dump_results(system, mu, partfact):
    """Format results"""
    if system.Files.no_output:
        return

    text = []
    header = []
    rowname = []
    data = []
    neig = len(mu)

    numeral = []
    for idx, item in enumerate(range(neig)):
        if mu.real[idx] == 0:
            marker = '*'
        elif mu.real[idx] > 0:
            marker = '**'
        else:
            marker = ''
        numeral.append('#' + str(idx+1) + marker)

    pf = []
    for prow in range(neig):
        temp_row = []
        for pcol in range(neig):
            temp_row.append(round(partfact[prow, pcol], 5))
        pf.append(temp_row)

    text.append('EIGENVALUES\n')
    header.append(['Real', 'Imag'])
    rowname.append(numeral)
    data.append([list(mu.real), list(mu.imag)])

    cpb = 7 # columns per block
    nblock = round(neig / cpb)

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
    system.DAE.factorize = True
    exec(system.Call.int)

    As = state_matrix(system)
    mu, pf = part_factor(As)
    dump_results(system, mu, pf)
