from sympy import Symbol, diff, sin, cos, exp, Integer

# --- INPUT ---
outfile = 'eq_out.txt'
consts = ['rsh', 'xsh', 'iLsh',
          'pref0', 'qref0', 'wref0', 'vref0',
          'iM', 'D', 'iTt',
          'Kp1', 'Ki1', 'Kp2', 'Ki2', 'Kp3', 'Ki3', 'Kp4', 'Ki4', 'KQ',
          ]
algebs = ['wref', 'vref',
          'P', 'Q',
          'vd', 'vq',
          'Idref', 'Iqref',
          'udref', 'uqref',
          'a', 'v',
          'v1', 'v2',
          ]
states = ['Id', 'Iq', 'ud', 'uq',
          'Md', 'Mq', 'Nd', 'Nq',
          'adq', 'xw',
          ]
interfaces = ['a', 'v', 'v1', 'v2']
algeb_eq = ['wref - wref0 - xw',  # wref
            'vref - vref0 - KQ*(qref0 - Q)',  # vref
            'vd * Id + vq * Iq - P',  # P
            '-vd * Iq + vq* Id - Q',  # Q
            'v * cos(adq - a) - vd',  # vd
            '-v * sin(adq - a) - vq',  # vq
            'Kp3 * (vref - vd) + Nd - Idref',  # Idref
            '- Kp4 * vq + Nq - Iqref',  # Iqref
            'vd - xsh*Iqref + Kp1*(Idref - Id) + Md - udref',  # udref
            'vq + xsh*Idref + Kp2*(Iqref - Iq) + Mq - uqref',  # uqref
            '-P',  # a
            '-Q',  # v
            '(ud * Id + uq * Iq) / (v1 - v2)',  # v1
            '-(ud * Id + uq * Iq) / (v1 - v2)',  # v2
            ]

diff_eq = ['-rsh * iLsh * Id + Iq + iLsh * (ud - vd)',  # Id
           '-rsh * iLsh * Iq - Id + iLsh * (uq - vq)',  # Iq
           'iTt*(udref - ud)',  # ud
           'iTt*(uqref - uq)',  # uq
           'Ki1*(Idref - Id)',  # Md
           'Ki2*(Iqref - Iq)',  # Mq
           'Ki3*(vref - vd)',  # Nd
           'Ki4*(-vq)',  # Nq
           'xw',  # adq
           'iM*(pref0 - P - D*xw)']  # xw
# --- INPUT ENDS ---


def symbolify(consts, algebs, states, interfaces, diff_eq, algeb_eq):
    sym_consts = []
    sym_algebs = []
    sym_states = []
    sym_interfaces = []
    sym_f = []
    sym_g = []

    # check for duplicate names
    var_names = consts + algebs + states
    if len(set(var_names)) != len(var_names):
        raise NameError('Duplicated names are declared!')

    # convert consts and variables into sympy.Symbol
    for item in consts:
        call = '{} = Symbol(item)'.format(item)
        exec(call)
        sym_consts.append(eval(item))

    for item in algebs:
        call = '{} = Symbol(item)'.format(item)
        exec(call)
        sym_algebs.append(eval(item))

    for item in states:
        call = '{} = Symbol(item)'.format(item)
        exec(call)
        sym_states.append(eval(item))

    for item in interfaces:
        call = '{} = Symbol(item)'.format(item)
        exec(call)
        sym_interfaces.append(eval(item))

    # convert equations into symbolic expression
    for item in algeb_eq:
        expr = eval('{}'.format(item))
        sym_g.append(expr)

    for item in diff_eq:
        expr = eval('{}'.format(item))
        sym_f.append(expr)

    return sym_consts, sym_algebs, sym_states, sym_interfaces, sym_f, sym_g


def stringfy(expr, sym_const, sym_states, sym_algebs):
    """Convert the right-hand-side of an equation into CVXOPT matrix operations"""
    expr_str = []
    if expr.is_Atom:
        if expr in sym_const:
            expr_str = 'self.{}'.format(expr)
        elif expr in sym_states:
            expr_str = 'dae.x[self.{}]'.format(expr)
        elif expr in sym_algebs:
            expr_str = 'dae.y[self.{}]'.format(expr)
        elif expr.is_Number:
            if expr.is_negative:
                expr_str = '{}'.format(expr)
            else:
                expr_str = str(expr)
        else:
            raise AttributeError('Unknown free symbol')
    else:
        nargs = len(expr.args)
        arg_str = []
        for arg in expr.args:
            arg_str.append(stringfy(arg, sym_const, sym_states, sym_algebs))

        if expr.is_Add:
            expr_str = ''
            for idx, item in enumerate(arg_str):
                if idx == 0:
                    if item[1] == ' ':
                        item = item[0] + item[2:]
                if idx > 0:
                    if item[0] == '-':
                        item = ' ' + item
                    elif item[1] == '-':
                        pass
                    else:
                        item = ' + ' + item
                expr_str += item

        elif expr.is_Mul:
            if nargs == 2 and expr.args[0].is_Integer:  # number * matrix
                if expr.args[0].is_positive:
                    expr_str = '{}*{}'.format(*arg_str)
                elif expr.args[0] == Integer('-1'):
                    expr_str = '- {}'.format(arg_str[1])
                else:  # negative but not -1
                    expr_str = '{}*{}'.format(*arg_str)
            else:  # matrix dot multiplication
                if expr.args[0] == Integer('-1'):
                    # bring '-' out of mul()
                    expr_str = ', '.join(arg_str[1:])
                    expr_str = '- mul(' + expr_str + ')'
                else:
                    expr_str = ', '.join(arg_str)
                    expr_str = 'mul(' + expr_str + ')'
        elif expr.is_Function:
            expr_str = ', '.join(arg_str)
            expr_str = str(expr.func) + '(' + expr_str + ')'
        elif expr.is_Pow:
            expr_str = '({})**{}'.format(*arg_str)
        elif expr.is_Div:
            expr_str = ', '.join(arg_str)
            expr_str = 'div(' + expr_str + ')'
        else:
            raise NotImplemented
    return expr_str


def derive(sym_algebs, sym_states, sym_f, sym_g):
    """Derive the jacobians of equation f and g.
    Save to Fx, Fy and Gx Gy in a list of three elements: [equation_idx, var_idx, expression]"""
    Fx = []
    Fy = []
    Gx = []
    Gy = []

    for eq_idx, expr in enumerate(sym_g):
        try:
            free_syms = expr.free_symbols
        except AttributeError:
            free_syms = []

        for sym in free_syms:
            if sym in sym_algebs:
                # take the derivative and go to Gy
                sym_idx = sym_algebs.index(sym)
                Gy.append([eq_idx, sym_idx, expr.diff(sym)])
            elif sym in sym_states:
                # take the derivative and go to Gx
                sym_idx = sym_states.index(sym)
                Gx.append([eq_idx, sym_idx, expr.diff(sym)])
            else:
                pass  # skip constants

    for eq_idx, expr in enumerate(sym_f):
        try:
            free_syms = expr.free_symbols
        except AttributeError:
            free_syms = []

        for sym in free_syms:
            if sym in sym_algebs:
                sym_idx = sym_algebs.index(sym)
                Fy.append([eq_idx, sym_idx, expr.diff(sym)])
            elif sym in sym_states:
                sym_idx = sym_states.index(sym)
                Fx.append([eq_idx, sym_idx, expr.diff(sym)])

    return Fx, Fy, Gx, Gy


def reformat(sym_consts, sym_algebs, sym_states, sym_interfaces, sym_f, sym_g, Fx, Fy, Gx, Gy):
    """Save equations into Python functions"""
    jacobians = ['Gy', 'Gx', 'Fx', 'Fy']

    fcall = []
    gcall = []
    gycall = []
    fxcall = []
    jac0 = []

    jac0_line = 'dae.add_jac({}0, {}, self.{}, self.{})'
    call_line = 'dae.add_jac({}, {}, self.{}, self.{})'

    mapping = {'F': sym_states,
               'G': sym_algebs,
               'y': sym_algebs,
               'x': sym_states}

    # format f and g equations
    for sym, eq in zip(sym_states, sym_f):
        string_eq = stringfy(eq, sym_consts, sym_states, sym_algebs)
        template = 'dae.f[self.{}] = {}'
        fcall.append(template.format(sym, string_eq))

    for sym, eq in zip(sym_algebs, sym_g):
        string_eq = stringfy(eq, sym_consts, sym_states, sym_algebs)
        if sym in sym_interfaces:
            template = 'dae.g += spmatrix({1}, self.{0}, [0] * self.n, (dae.m, 1), \'d\')'
        else:
            template = 'dae.g[self.{0}] = {1}'
        gcall.append(template.format(sym, string_eq))

    # format Jacobians
    for jac in jacobians:
        for item in eval(jac):
            eqname = mapping[jac[0]][item[0]]
            varname = mapping[jac[1]][item[1]]
            equation = item[2]
            try:
                free_syms = equation.free_symbols
            except AttributeError:
                free_syms = []

            string_eq = stringfy(equation, sym_consts, sym_states, sym_algebs)

            isjac0 = 1
            for sym in free_syms:
                if sym in sym_consts:
                    continue
                elif sym in sym_algebs + sym_states:
                    isjac0 = 0
                    break
                else:
                    raise KeyError

            if isjac0:
                jac0.append(jac0_line.format(jac, string_eq, eqname, varname))
            else:
                if jac == 'Gy':
                    gycall.append(call_line.format(jac, string_eq, eqname, varname))
                else:
                    fxcall.append(call_line.format(jac, string_eq, eqname, varname))

    return fcall, gcall, fxcall, gycall, jac0


def write(fcall, gcall, fxcall, gycall, jac0):
    fid = open(outfile, 'w')

    # write f and g equation into fcall and gcall
    fid.write('def gcall(self, dae):\n')
    for item in gcall:
        fid.writelines('    ' + item + '\n')
    fid.write('\n')

    fid.write('def fcall(self, dae):\n')
    for item in fcall:
        fid.writelines('    ' + item + '\n')
    fid.write('\n')

    fid.write('def gycall(self, dae):\n')
    for item in gycall:
        fid.writelines('    ' + item + '\n')
    fid.writelines('\n')

    fid.write('def fxcall(self, dae):\n')
    for item in fxcall:
        fid.writelines('    ' + item + '\n')
    fid.writelines('\n')

    fid.write('def jac0(self, dae):\n')
    for item in jac0:
        fid.writelines('    ' + item + '\n')

    fid.close()


def stats(sym_consts, sym_algebs, sym_states, fcall, gcall, fxcall, gycall, jac0):
    print('Statistics:')
    print('')
    print('constants: {}'.format(len(sym_consts)))
    print('algebraics: {}'.format(len(sym_algebs)))
    print('states: {}'.format(len(sym_states)))
    print('')
    print('differential equations: {}'.format(len(fcall)))
    print('constants: {}'.format(len(gcall)))
    print('')
    print('fxcall lines: {}'.format(len(fxcall)))
    print('gycall lines: {}'.format(len(gycall)))
    print('jac0 lines: {}'.format(len(jac0)))


if __name__ == "__main__":
    s_const, s_algeb, s_states, s_interfaces, s_f, s_g = symbolify(consts, algebs, states, interfaces, diff_eq, algeb_eq)
    Fx, Fy, Gx, Gy = derive(s_algeb, s_states, s_f, s_g)
    fcall, gcall, fxcall, gycall, jac0 = reformat(s_const, s_algeb, s_states, s_interfaces, s_f, s_g, Fx, Fy, Gx, Gy)
    write(fcall, gcall, fxcall, gycall, jac0)
    stats(s_const, s_algeb, s_states, fcall, gcall, fxcall, gycall, jac0)
