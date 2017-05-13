from sympy import Symbol, diff, sin, cos, exp, Integer
from andes.main import elapsed
from andes.utils.math import to_number
# --- INPUT ---
# outfile = 'VSC1.txt'
# name = 'VSC1'
# doc_string = "VSC Type 1"
# group = 'AC/DC'
#
# data = {'rsh': 0.00025,
#         'xsh': 0.006,
#         }
# descr = {'rsh': 'ac interface resistance',
#          'xsh': 'ac interface reactance',
#          }
# units = {}
#
# params = ['rsh', 'xsh']
#
# fnamex = []
# fnamey = []
#
# mandatory = []
# zeros = []
#
# powers = []
# voltages = []
# currents = []
# z = []
# y = []
# dccurrents = []
# dcvoltages = []
# r = []
# g = []
# times = []
#
# ac = {}
# dc = {}
# ctrl = {}
#
# consts = ['rsh', 'xsh', 'iLsh',
#           'pref0', 'qref0', 'wref0', 'vref0',
#           'iM', 'D', 'iTt',
#           'Kp1', 'Ki1', 'Kp2', 'Ki2', 'Kp3', 'Ki3', 'Kp4', 'Ki4', 'KQ',
#           ]
#
# # consts = list(data.keys()) + list(service_eq.keys())
#
# algebs = ['wref', 'vref', 'p', 'q', 'vd', 'vq',
#           'Idref', 'Iqref', 'udref', 'uqref',
#           'a', 'v','v1', 'v2',
#           ]
# interfaces = ['a', 'v', 'v1', 'v2']
#
# states = ['Id', 'Iq', 'ud', 'uq',
#           'Md', 'Mq', 'Nd', 'Nq',
#           'adq', 'xw',
#           ]
#
# # --- equation section ---
# # initialization equations
# init1_eq = []
#
# # service variable and equation declaration
# service_eq = {}
#
# # algebraic equations in g(x, y) = 0 form
# #   defined in the order of the algeb variables
# algeb_eq = ['wref - wref0 - xw',  # wref
#             'vref - vref0 + KQ*(q - qref0)',  # vref
#             'vd * Id + vq * Iq - p',  # P
#             'vd * Iq - vq* Id - q',  # Q
#             'v * cos(adq - a) - vd',  # vd
#             'v * sin(adq - a) - vq',  # vq
#             'Kp3 * (vref - vd) + Nd - Idref',  # Idref
#             '- Kp4 * vq + Nq - Iqref',  # Iqref
#             'vd + xsh*Iqref + Kp1*(Idref - Id) + Md - udref',  # udref
#             'vq - xsh*Idref + Kp2*(Iqref - Iq) + Mq - uqref',  # uqref
#             '-p',  # a
#             '-q',  # v
#             '(ud * Id + uq * Iq) / (v1 - v2)',  # v1
#             '-(ud * Id + uq * Iq) / (v1 - v2)',  # v2
#             ]
# windup = {}
# hard_limit = {'Idref': ['Idmin', 'Idmax'],
#               }
#
# # differential equations in f(x, y) = derivative(x) form
# #   defined in the order of the state variables
# diff_eq = ['-rsh*iLsh*Id - Iq + iLsh*(ud - vd)',  # Id
#            '-rsh*iLsh*Iq + Id + iLsh*(uq - vq)',  # Iq
#            'iTt*(udref - ud)',  # ud
#            'iTt*(uqref - uq)',  # uq
#            'Ki1*(Idref - Id)',  # Md
#            'Ki2*(Iqref - Iq)',  # Mq
#            'Ki3*(vref - vd)',  # Nd
#            'Ki4*(-vq)',  # Nq
#            'wref - wref0',  # adq
#            'iM * (pref0 - p - D*xw)'
#            ]  # xw
#
# anti_windup = {'Nd': ['Ta', 'Ndmin', 'Ndmax'],
#                } # [time_constant, min, max]
# --- INPUT ENDS ---


def card_parser(file):
    """Parse an ANDES card file into internal variables"""
    try:
        fid = open(file, 'r')
        raw_file = fid.readlines()
    except IOError:
        print('* IOError while reading input card file.')
        return

    ret_dict = {}
    ret_dict['outfile'] = file.split('.')[0] + '.py'
    key, val = None, None
    for lineno, line in enumerate(raw_file):
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        if '=' in line:  # defining a field
            key, val = line.split('=')
            key, val = key.strip(), val.strip()
            val = [] if val == '' else val
            ret_dict.update({key: val})
            if val:
                val = val.split(';')
        else:
            val.extend(line.split(';'))
        if val:
            val = de_blank(val)
            ret_dict[key] = val

    for key, val in ret_dict.items():
        if not val:
            continue
        if type(val) == list:
            if ':' in val[0]:
                new_val = {}
                for item in val:
                    m, n = item.split(':')
                    m, n = m.strip(), n.strip()
                    if ',' in n:
                        n = n.split(',')
                        n = de_blank(n)
                        n = [to_number(i) for i in n]
                    else:
                        n = to_number(n)
                    new_val.update({m.strip(): n})
                ret_dict[key] = new_val

    ret_dict['name'] = ret_dict['name'][0]
    ret_dict['doc_string'] = ret_dict['doc_string'][0]
    ret_dict['group'] = ret_dict['group'][0]
    if ret_dict['service_eq'] == []:
        ret_dict['service_eq'] = {}
    return ret_dict

def add_quotes(string):
    return '\'{}\''.format(string)

def de_blank(val):
    """Remove blank elements in `val` and return `ret`"""
    ret = list(val)
    if type(val) == list:
        for idx, item in enumerate(val):
            if item.strip() == '':
                ret.remove(item)
            else:
                ret[idx] = item.strip()
    return ret

def to_list(string):
    if ';' not in string:
        if ':' in string:
            key, val = string.split(':')
            return dict(key=val)

def run(outfile='', name='', doc_string='', group='', data={}, descr={},
        units={}, params=[], fnamex=[], fnamey=[], mandatory=[], zeros=[],
        powers=[], currents=[], voltages=[], z=[], y=[], dccurrents=[],
        dcvoltages=[], r=[], g=[], times=[], ac={}, dc={}, ctrl={},
        consts=[], algebs=[], interfaces=[], states=[], init1_eq=[], service_eq={},
        algeb_eq=[], windup={}, hard_limit={}, diff_eq=[], anti_windup={}, **kwargs):
    space4 = '    '
    space8 = space4 * 2
    """Input data consistency check"""
    to_check = {'param': params,
                'mandatory': mandatory,
                'zero': zeros,
                'power': powers,
                'voltage': voltages,
                'currents': currents,
                'z': z,
                'y': y,
                'dccurrent': dccurrents,
                'dcvoltage': dcvoltages,
                'r': r,
                'g': g,
                'times': times,
                }
    if not data:
        print('* Error: <data> dictionary is not defined.')
        return

    for key, val in to_check.items():
        if not val:
            continue
        for item in val:
            if item not in data.keys():
                print('* Warning: {} <{}> is not in data.'.format(key, item))

    for key, val in hard_limit.items():
        if key not in algebs:
            print('* Warning: variable <{}> in hard_limit not defined.'.format(key))
        for item in val:
            if item not in consts:
                print('* Warning: const <{}> in hard_limit not defined.'.format(item))
    for key, val in windup.items():
        if key not in algebs:
            print('* Warning: variable <{}> in windup not defined.'.format(key))
        for item in val:
            if item not in consts:
                print('* Warning: const <{}> in windup not defined.'.format(item))
    for key, val in anti_windup.items():
        if key not in states:
            print('* Warning: variable <{}> in anti_windup not defined.'.format(key))
        for item in val:
            if item not in consts:
                print('* Warning: const <{}> in anti_windup not defined.'.format(item))

    """Equation and variable number check"""
    nalgebs, nalgeb_eq, nstates, ndiff_eq = len(algebs), len(algeb_eq), len(states), len(diff_eq)
    if nalgebs != nalgeb_eq:
        print('* Warning: there are {} algebs and {} algeb equations.'.format(nalgebs, nalgeb_eq))
    if nstates != ndiff_eq:
        print('* Warning: there are {} states and {} differential equations.'.format(nstates, ndiff_eq))

    # check for duplicate names
    var_names = consts + algebs + states
    if len(set(var_names)) != len(var_names):
        raise NameError('Duplicated names are declared!')

    """Set up sympy symbols for variables, constants and equations"""
    sym_consts, sym_algebs, sym_states, sym_interfaces = [], [], [], []
    sym_f, sym_g, sym_serv, sym_init1 = [], [], [], []
    sym_hard_limit, sym_windup, sym_anti_windup = [], [], []

    states_anti_windup = list(anti_windup.keys())
    algebs_windup = list(windup.keys())
    algebs_hard_limit = list(hard_limit.keys())

    # convert consts and variables into sympy.Symbol
    sym_maping = {'consts': sym_consts,
                  'algebs': sym_algebs,
                  'states': sym_states,
                  'interfaces': sym_interfaces,
                  'states_anti_windup': sym_anti_windup,
                  'algebs_windup': sym_windup,
                  'algebs_hard_limit': sym_hard_limit,
                  }
    for key, val in sym_maping.items():
        for item in eval(key):
            call = '{} = Symbol(item)'.format(item)
            exec(call)
            val.append(eval(item))

    # convert service_eq.keys() into sympy.Symbols and values into equations
    for var, eq in service_eq.items():
        if var not in consts:
            print('* Warning: declaring undefined service variable <{}>'.format(var))
        call = '{} = Symbol(var)'.format(var)
        exec(call)
        expr = eval('{}'.format(eq))
        sym_serv.append([eval(var), expr])

    for item in init1_eq:
        var = item[0]
        eq = item[1]
        if var not in states + algebs:
            print('* Warning: initializing undefined variable <{}>'.format(var))
        call = '{} = Symbol(var)'.format(var)
        exec(call)
        expr = eval('{}'.format(eq))
        sym_init1.append([eval(var), expr])

    # convert equations into symbolic expression
    for item in algeb_eq:
        expr = eval('{}'.format(item))
        sym_g.append(expr)

    for item in diff_eq:
        expr = eval('{}'.format(item))
        sym_f.append(expr)

    """Derive the jacobians of equation f and g.
    Save to Fx, Fy and Gx Gy in a list of three elements: [equation_idx, var_idx, expression]"""
    Fx, Fy, Gx, Gy = list(), list(), list(), list()
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

    """Save equations into callable CVXOPT functions"""
    fcall, gcall = [], []
    gycall, fxcall, jac0 = [], [], []
    jac0_line = 'dae.add_jac({}0, {}, self.{}, self.{})'
    call_line = 'dae.add_jac({}, {}, self.{}, self.{})'

    # format f and g equations
    fcall_anti_windup_1 = 'dae.f[self.{0}] = div({0} - dae.x[self.{0}], self.{1})'
    fcall_anti_windup_2 = 'dae.anti_windup(self.{0}, self.{1}, self.{2})'

    for sym, eq in zip(sym_states, sym_f):
        string_eq = stringfy(eq, sym_consts, sym_states, sym_algebs)
        # handling anti_windup
        if sym in sym_anti_windup:
            template = '{0} = {1}'
        else:
            template = 'dae.f[self.{0}] = {1}'
        fcall.append(template.format(sym, string_eq))
        if sym in sym_anti_windup:
            val = eval('anti_windup[\'{}\']'.format(sym))
            fcall.append(fcall_anti_windup_1.format(sym, val[0]))
            fcall.append(fcall_anti_windup_2.format(sym, val[1], val[2]))

    gcall_windup = 'dae.windup(self.{0}, self.{1}, self.{2})'
    gcall_hard_limit = 'dae.hard_limit(self.{0}, self.{1}, self.{2})'

    for sym, eq in zip(sym_algebs, sym_g):
        string_eq = stringfy(eq, sym_consts, sym_states, sym_algebs)
        if sym in sym_interfaces:
            template = 'dae.g += spmatrix({1}, self.{0}, [0]*self.n, (dae.m, 1), \'d\')'
        else:
            template = 'dae.g[self.{0}] = {1}'
        gcall.append(template.format(sym, string_eq))

        if sym in sym_windup:
            val = eval('windup[\'{}\']'.format(sym))
            gcall.append(gcall_windup.format(sym, val[0], val[1]))
        elif sym in sym_hard_limit:
            val = eval('hard_limit[\'{}\']'.format(sym))
            gcall.append(gcall_hard_limit.format(sym, val[0], val[1]))

    # format Jacobians
    jacobians = ['Gy', 'Gx', 'Fx', 'Fy']
    mapping = dict(F=sym_states, G=sym_algebs, y=sym_algebs, x=sym_states)

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
    init1call = []
    for item in sym_init1:
        rhs = stringfy(item[1], sym_consts, sym_states, sym_algebs)
        if item[0] in sym_algebs:
            xy = 'y'
        elif item[0] in sym_states:
            xy = 'x'
        else:
            raise KeyError
        out = 'dae.{}[self.{}] = {}'.format(xy, item[0], rhs)
        init1call.append(out)

    servcall = []
    for item in sym_serv:
        rhs = stringfy(item[1], sym_consts)
        out = 'self.{} = {}'.format(item[0], rhs)
        servcall.append(out)

    calls = {'gcall': not not gcall,
             'fcall': not not fcall,
             'gycall': not not gycall,
             'fxcall': not not fxcall,
             'jac0': not not jac0,
             'init1': not not init1call,
             }

    """Build function call strings"""
    out_calls = []
    if servcall:
        out_calls.append(space4 + 'def servcall(self):')
        for item in servcall:
            out_calls.append(space8 + item)
        out_calls.append('')

    if init1call:
        out_calls.append(space4 + 'def init1(self, dae):')
        out_calls.append(space8 + 'self.servcall()')
        for item in init1call:
            out_calls.append(space8 + item)

    if gcall:
        out_calls.append(space4 + 'def gcall(self, dae):')
        for item in gcall:
            out_calls.append(space8 + item)
        out_calls.append('')

    if fcall:
        out_calls.append(space4 + 'def fcall(self, dae):')
        for item in fcall:
            out_calls.append(space8 + item)
        out_calls.append('')

    if gycall:
        out_calls.append(space4 + 'def gycall(self, dae):')
        for item in gycall:
            out_calls.append(space8 + item)
        out_calls.append('')
    if fxcall:
        out_calls.append(space4 + 'def fxcall(self, dae):')
        for item in fxcall:
            out_calls.append(space8 + item)
        out_calls.append('')
    if jac0:
        out_calls.append(space4 + 'def jac0(self, dae):')
        for item in jac0:
            out_calls.append(space8 + item)

    """Class definitions in out_init"""
    # bulk update or extend of dict and list
    param_assign = space8 + 'self._{} = {}'
    list_extend = space8 + 'self._{}.extend({})'
    dict_update = space8 + 'self._{}.update({})'

    out_init = list()  # def __init__ call strings
    out_init.append('from cvxopt import matrix, spmatrix')
    out_init.append('from ..consts import *')
    out_init.append('from .base import ModelBase\n\n')
    out_init.append('class {}(ModelBase):'.format(name))
    if doc_string:
        out_init.append(space4 + "\"\"\"{}\"\"\"".format(add_quotes(doc_string)))
    out_init.append(space4 + 'def __init__(self, system, name):')
    out_init.append(space8 + 'super().__init__(system, name)')
    if not group:
        print('*Error: Group name is not defined!')
    else:
        out_init.append(param_assign.format('group', add_quotes(group)))
    if name:
        out_init.append(param_assign.format('name', add_quotes(name)))

    meta_dict_upd = {'data': data,
                     'units': units,
                     'descr': descr,
                     'ac': ac,
                     'dc': dc,
                     'ctrl': ctrl,
                     }
    meta_list_ext = {'params': params,
                     'algebs': algebs,
                     'states': states,
                     'services': list(service_eq.keys()),
                     'mandatory': mandatory,
                     'zeros': zeros,
                     'powers': powers,
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

    for key, val in meta_list_ext.items():
        if val:
            out_init.append(list_extend.format(key, val))
    for key, val in meta_dict_upd.items():
        if val:
            out_init.append(dict_update.format(key, val))

    out_init.append(dict_update.format('calls', calls))

    out_init.append(space8 + 'self._inst_meta()')
    out_init.append('')

    # write to file
    fid = open(outfile, 'w')
    for line in out_init:
        fid.write(line + '\n')
    for line in out_calls:
        fid.write(line + '\n')
    fid.close()

    print('Statistics:')
    print('')
    print('constants: {}'.format(len(sym_consts)))
    print('algebraics: {}'.format(len(sym_algebs)))
    print('states: {}'.format(len(sym_states)))
    print('')
    print('differential equations: {}'.format(len(fcall)))
    print('algebraic equations: {}'.format(len(gcall)))
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
    inputs_dict = card_parser('AVR.andc')
    run(**inputs_dict)
    _, s = elapsed(t)
    print('Elapsed time: {}'.format(s))
