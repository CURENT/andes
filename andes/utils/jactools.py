from cvxopt import spmatrix, matrix


def diag0(m, name, system):
    """Check matrix m for diagonal 0 elements"""
    pos = []
    names = []
    pairs = ''
    size = m.size
    diag = m[0:size[0]**2:size[0]+1]
    for idx in range(size[0]):
        if abs(diag[idx]) <= 1e-8:
            pos.append(idx)
    for idx in pos:
        names.append(system.VarName.__dict__[name][idx])
    if names:
        for i, j in zip(pos, names):
            pairs += '{0}: {1}\n'.format(i, j)
        system.Log.debug('Jacobian diagonal check:')
        system.Log.debug(pairs)
