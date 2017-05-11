from sympy import Symbol, diff, sin, cos, exp, Integer
from andes.main import elapsed

# --- INPUT ---
outfile = 'VSC1.txt'
name = 'VSC1'
group = 'AC/DC'

data = {'rsh': 0.00025,
        'xsh': 0.006,
        }
descr = {'rsh': 'ac interface resistance',
         'xsh': 'ac interface reactance',
         }
units = {}

params = ['rsh', 'xsh']

fnamex = []
fnamey = []

service = []
mandatory = []
zeros = []

powers = []
voltages = []
currents = []
z = []
y = []
dccurrents = []
dcvoltages = []
r = []
g = []
times = []

ac = {}
dc = {}
ctrl = {}

calls = ['gcall']

algebs = ['wref', 'vref',
          'p', 'q',
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


def declarations(name=None, group=None, data=None, descr=None, units=None, params=None, fnamex=None, fnamey=None,
                 service=None, mandatory=None, zeros=None, powers=None, voltages=None, currents=None, z=None,
                 y=None, dcvoltages=None, dccurrents=None, r=None, g=None, times=None, ac=None, dc=None, ctrl=None):
    out = []
    space4, space8 = '    ', '        '
    out.append('class {}(ModelBase):'.format(name))
    out.append(space4 + 'def __init__(self, system, name):')
    out.append(space8 + 'super().__init__(system, name)')
    if not group:
        print('*Error: Group name is not defined!')

    out.append(space8 + 'self._group = {}'.format(group))

    list_extend = space8 + 'self._{}.extend({})'
    dict_update = space8 + 'self._{}.update({})'

    if name:
        out.append(space8 + 'self._name = {}'.format(name))
    if data:
        out.append(dict_update.format('data', data))
    if units:
        out.append(dict_update.format('unit', units))

    if params:
        out.append(list_extend.format('params', params))
    if descr:
        out.append(list_extend.format('deacr', descr))

    if algebs:
        out.append(list_extend.format('algebs', algebs))
    if states:
        out.append(list_extend.format('states', states))
    if service:
        out.append(list_extend.format('services', service))
    if mandatory:
        out.append(list_extend.format('mandatory', mandatory))
    if zeros:
        out.append(list_extend.format('zeros', zeros))

    perunits = {'powers': powers,
                'voltages': voltages,
                'currents': currents,
                'z': z,
                'y': y,
                'dccurrents': dccurrents,
                'dcvoltages': dcvoltages,
                'r': r,
                'g': g,
                'times': times,
                }
    interfaces = {'ac': ac,
                  'dc': dc,
                  'ctrl': ctrl,
                  }

    for key, val in perunits.items():
        if val:
            out.append(list_extend.format(key, val))

    for key, val in interfaces.items():
        if val:
            out.append(dict_update.format(key, val))


    fid = open(outfile, 'w')
    for line in out:
        fid.write(line + '\n')

    fid.close()






consts = ['rsh', 'xsh', 'iLsh',
          'pref0', 'qref0', 'wref0', 'vref0',
          'iM', 'D', 'iTt',
          'Kp1', 'Ki1', 'Kp2', 'Ki2', 'Kp3', 'Ki3', 'Kp4', 'Ki4', 'KQ',
          ]
interfaces = ['a', 'v', 'v1', 'v2']

algeb_eq = ['wref - wref0 - xw',  # wref
            'vref - vref0 + KQ*(q - qref0)',  # vref
            'vd * Id + vq * Iq - p',  # P
            'vd * Iq - vq* Id - q',  # Q
            'v * cos(adq - a) - vd',  # vd
            'v * sin(adq - a) - vq',  # vq
            'Kp3 * (vref - vd) + Nd - Idref',  # Idref
            '- Kp4 * vq + Nq - Iqref',  # Iqref
            'vd + xsh*Iqref + Kp1*(Idref - Id) + Md - udref',  # udref
            'vq - xsh*Idref + Kp2*(Iqref - Iq) + Mq - uqref',  # uqref
            '-p',  # a
            '-q',  # v
            '(ud * Id + uq * Iq) / (v1 - v2)',  # v1
            '-(ud * Id + uq * Iq) / (v1 - v2)',  # v2
            ]

diff_eq = ['-rsh*iLsh*Id - Iq + iLsh*(ud - vd)',  # Id
           '-rsh*iLsh*Iq + Id + iLsh*(uq - vq)',  # Iq
           'iTt*(udref - ud)',  # ud
           'iTt*(uqref - uq)',  # uq
           'Ki1*(Idref - Id)',  # Md
           'Ki2*(Iqref - Iq)',  # Mq
           'Ki3*(vref - vd)',  # Nd
           'Ki4*(-vq)',  # Nq
           'wref - wref0',  # adq
           'iM * (pref0 - p - D*xw)'
           ]  # xw
init = []
serv = {}
# --- INPUT ENDS ---


def equations(consts, algebs, states, interfaces, diff_eq, algeb_eq, serv=None, init=None):
    sym_consts = []
    sym_algebs = []
    sym_states = []
    sym_interfaces = []
    sym_f = []
    sym_g = []
    sym_serv = []
    sym_init = []

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

    for var, eq in serv.items():
        if var not in consts:
            print('* Warning: declaring undefined service variable <{}>'.format(var))
        call = '{} = Symbol(var)'.format(var)
        exec(call)
        expr = eval('{}'.format(eq))
        sym_serv.append([eval(var), expr])

    for item in init:
        var = item[0]
        eq = item[1]
        if var not in states + algebs:
            print('* Warning: initializing undefined variable <{}>'.format(var))
        call = '{} = Symbol(var)'.format(var)
        exec(call)
        expr = eval('{}'.format(eq))
        sym_init.append([eval(var), expr])

    # convert equations into symbolic expression
    for item in algeb_eq:
        expr = eval('{}'.format(item))
        sym_g.append(expr)

    for item in diff_eq:
        expr = eval('{}'.format(item))
        sym_f.append(expr)

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
            template = 'dae.g += spmatrix({1}, self.{0}, [0]*self.n, (dae.m, 1), \'d\')'
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

    """Format initialization and service calls"""
    initcall = []
    for item in sym_init:
        rhs = stringfy(item[1], sym_const, sym_state, sym_algeb)
        if item[0] in sym_algeb:
            xy = 'y'
        elif item[0] in sym_state:
            xy = 'x'
        else:
            raise KeyError
        out = 'dae.{}[self.{}] = {}'.format(xy, item[0], rhs)
        initcall.append(out)

    servcall = []
    for item in sym_serv:
        rhs = stringfy(item[1], sym_const)
        out = 'self.{} = {}'.format(item[0], rhs)
        servcall.append(out)

    fid = open(outfile, 'a')

    if servcall:

        fid.write('def servcall(self):\n')
        for item in servcall:
            fid.writelines('    ' + item + '\n')
        fid.write('\n')

    if initcall:
        fid.write('def init1(self, dae):\n')
        fid.writelines('    self.servcall()\n')
        for item in initcall:
            fid.writelines('    ' + item + '\n')
        fid.write('\n')

    # write f and g equation into fcall and gcall
    space4 = '    '
    space8 = space4 * 2
    fid.write(space4 + 'def gcall(self, dae):\n')
    for item in gcall:
        fid.writelines(space8 + item + '\n')
    fid.write('\n')

    fid.write(space4 + 'def fcall(self, dae):\n')
    for item in fcall:
        fid.writelines(space8 + item + '\n')
    fid.write('\n')

    fid.write(space4 + 'def gycall(self, dae):\n')
    for item in gycall:
        fid.writelines(space8 + item + '\n')
    fid.writelines('\n')

    fid.write(space4 + 'def fxcall(self, dae):\n')
    for item in fxcall:
        fid.writelines(space8 + item + '\n')
    fid.writelines('\n')

    fid.write(space4 + 'def jac0(self, dae):\n')
    for item in jac0:
        fid.writelines(space8 + item + '\n')

    fid.close()

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


def stringfy(expr, sym_const=None, sym_states=None, sym_algebs=None):
    """Convert the right-hand-side of an equation into CVXOPT matrix operations"""
    if not sym_const:
        sym_const = []
    if not sym_states:
        sym_states = []
    if not sym_algebs:
        sym_algebs = []
    expr_str = []
    if type(expr) == int:
        return
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
            raise NotImplementedError
    return expr_str

if __name__ == "__main__":
    t, s = elapsed()
    declarations()
    equations(consts, algebs, states, interfaces, diff_eq, algeb_eq, serv, init)
    _, s = elapsed(t)
    print('Elapsed time: {}'.format(s))
