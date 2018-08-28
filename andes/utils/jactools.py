def diag0(jac, name, system):
    """
    Check matrix ``jac`` for diagonal elements that equals 0
    """
    pos = []
    names = []
    pairs = ''
    size = jac.size
    diag = jac[0:size[0] ** 2:size[0] + 1]

    for idx in range(size[0]):
        if abs(diag[idx]) <= 1e-8:
            pos.append(idx)

    for idx in pos:
        names.append(system.VarName.__dict__[name][idx])

    if len(names) > 0:
        for i, j in zip(pos, names):
            pairs += '{0}: {1}\n'.format(i, j)
        system.Log.debug('Jacobian diagonal check:')
        system.Log.debug(pairs)
