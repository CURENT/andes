import sys

import pprint
try:
    from sympy import Symbol, diff, sin, cos, exp, Integer  # NOQA
except ImportError:
    raise ImportError('Please install sympy to parse ANDES cards.')

from andes.utils.math import to_number
import logging

logger = logging.getLogger(__name__)


def testlines(fid):
    try:
        first = fid.readline()
        if 'card' in first:
            return True
        else:
            return False
    except IOError:
        print('* IOError while reading input card file.')


def read(file, system):
    """Parse an ANDES card file into internal variables"""
    try:
        fid = open(file, 'r')
        raw_file = fid.readlines()
    except IOError:
        print('* IOError while reading input card file.')
        return

    ret_dict = dict()
    ret_dict['outfile'] = file.split('.')[0].lower() + '.py'
    key, val = None, None
    for idx, line in enumerate(raw_file):
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        elif '#' in line:
            line = line.split('#')[0]
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

    ret_dict_ord = dict(ret_dict)

    for key, val in ret_dict.items():
        if not val:
            continue
        if type(val) == list:
            if ':' in val[0]:
                new_val = {}  # return in a dictionary
                new_val_ord = [
                ]  # return in an ordered list with the dict keys at 0
                for item in val:
                    try:
                        m, n = item.split(':')
                    except ValueError:
                        print('* Error: check line <{}>'.format(item))
                        return
                    m, n = m.strip(), n.strip()
                    if ',' in n:
                        n = n.split(',')
                        n = de_blank(n)
                        n = [to_number(i) for i in n]
                    else:
                        n = to_number(n)
                    new_val.update({m.strip(): n})
                    new_val_ord.append([m.strip(), n])
                ret_dict[key] = new_val
                ret_dict_ord[key] = new_val_ord

    ret_dict['name'] = ret_dict['name'][0]
    ret_dict['doc_string'] = ret_dict['doc_string'][0]
    ret_dict['group'] = ret_dict['group'][0]
    ret_dict['service_keys'] = list(ret_dict['service_eq'].keys())
    ret_dict['consts'] = list(ret_dict['data'].keys()) + list(
        ret_dict['service_eq'].keys())
    ret_dict['init1_eq'] = ret_dict_ord['init1_eq']
    ret_dict['service_eq'] = ret_dict_ord['service_eq']
    ret_dict['ctrl'] = ret_dict_ord['ctrl']

    copy_algebs = []
    copy_states = []
    for item in ret_dict['ctrl']:
        key, val = item
        if val[3] == 'y':
            copy_algebs.append(key)
        elif val[3] == 'x':
            copy_states.append(key)
        elif val[3] == 'c':
            ret_dict['consts'].append(key)
    ret_dict['copy_algebs'] = copy_algebs
    ret_dict['copy_states'] = copy_states

    return run(system, **ret_dict)


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


def run(system,
        outfile='',
        name='',
        doc_string='',
        group='',
        data={},
        descr={},
        units={},
        params=[],
        fnamex=[],
        fnamey=[],
        mandatory=[],
        zeros=[],
        powers=[],
        currents=[],
        voltages=[],
        z=[],
        y=[],
        dccurrents=[],
        dcvoltages=[],
        r=[],
        g=[],
        times=[],
        ac={},
        dc={},
        ctrl={},
        consts=[],
        algebs=[],
        interfaces=[],
        states=[],
        init1_eq={},
        service_eq={},
        algeb_eq=[],
        windup={},
        hard_limit={},
        diff_eq=[],
        anti_windup={},
        copy_algebs=[],
        copy_states=[],
        service_keys=[],
        **kwargs):
    retval = True
    space4 = '    '
    space8 = space4 * 2
    """Input data consistency check"""
    to_check = {
        'param': params,
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

    for item in interfaces:
        if item not in algebs + copy_algebs:
            print('* Warning: interface <{}> is not defined'.format(item))

    for key, val in hard_limit.items():
        if key not in algebs:
            print('* Warning: variable <{}> in hard_limit not defined.'.format(
                key))
        for item in val:
            if type(item) in (int, float):
                pass
            elif item not in consts:
                print(
                    '* Warning: const <{}> in hard_limit not defined.'.format(
                        item))

    for key, val in windup.items():
        if key not in algebs:
            print(
                '* Warning: variable <{}> in windup not defined.'.format(key))
        for item in val:
            if type(item) in (int, float):
                continue
            elif item not in consts:
                print('* Warning: const <{}> in windup not defined.'.format(
                    item))

    for key, val in anti_windup.items():
        if key not in states:
            print(
                '* Warning: variable <{}> in anti_windup not defined.'.format(
                    key))
        for item in val:
            if type(item) in (int, float):
                continue
            elif item not in consts:
                print(
                    '* Warning: const <{}> in anti_windup not defined.'.format(
                        item))
    """Equation and variable number check"""
    nalgebs, nalgeb_eq, nstates, ndiff_eq, ninterfaces = len(algebs), len(
        algeb_eq), len(states), len(diff_eq), len(interfaces)

    if nalgebs + ninterfaces != nalgeb_eq:
        print('* Warning: there are {} algebs and {} algeb equations.'.format(
            nalgebs, nalgeb_eq))

    if nstates != ndiff_eq:
        print('* Warning: there are {} states and {} differential equations.'.
              format(nstates, ndiff_eq))

    # check for duplicate names
    var_names = consts + algebs + states + copy_algebs + copy_states
    if len(set(var_names)) != len(var_names):
        raise NameError('Duplicated names are declared!')
    """Set up sympy symbols for variables, constants and equations"""
    sym_consts, sym_algebs, sym_states, sym_interfaces = [], [], [], []
    sym_algebs_ext, sym_states_ext = [], []
    sym_f, sym_g, sym_serv, sym_init1 = [], [], [], []
    sym_hard_limit, sym_windup, sym_anti_windup = [], [], []

    states_anti_windup = list(anti_windup.keys())
    # algebs_windup = list(windup.keys())
    # algebs_hard_limit = list(hard_limit.keys())

    # remove interface variables in copy_algebs
    for item in interfaces:
        if item in copy_algebs:
            copy_algebs.remove(item)

    # algebs_ext = algebs + interfaces + copy_algebs
    # states_ext = states + copy_states

    for idx, var in enumerate(states):
        if var in states_anti_windup:
            tpl = '({} - {}) / {}'
            diff_eq[idx] = tpl.format(diff_eq[idx], var, anti_windup[var][0])

    # convert consts and variables into sympy.Symbol
    sym_maping = {
        'consts': sym_consts,
        'algebs': sym_algebs,
        'states': sym_states,
        'algebs_ext': sym_algebs_ext,
        'states_ext': sym_states_ext,
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
    # for var, eq in service_eq.items():
    if service_eq:
        for item in service_eq:
            var = item[0]
            eq = item[1]
            if var not in consts:
                print('* Warning: declaring undefined service variable <{}>'.
                      format(var))
            call = '{} = Symbol(var)'.format(var)
            exec(call)
            expr = eval('{}'.format(eq))
            sym_serv.append([eval(var), expr])

    if init1_eq:
        for item in init1_eq:
            var = item[0]
            eq = item[1]
            if var not in states + algebs:
                print('* Warning: initializing undefined variable <{}>'.format(
                    var))
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
            if sym in sym_algebs_ext:
                # take the derivative and go to Gy
                sym_idx = sym_algebs_ext.index(sym)
                Gy.append([eq_idx, sym_idx, expr.diff(sym)])
            elif sym in sym_states_ext:
                # take the derivative and go to Gx
                sym_idx = sym_states_ext.index(sym)
                Gx.append([eq_idx, sym_idx, expr.diff(sym)])
            else:
                pass  # skip constants

    for eq_idx, expr in enumerate(sym_f):
        try:
            free_syms = expr.free_symbols
        except AttributeError:
            free_syms = []

        for sym in free_syms:
            if sym in sym_algebs_ext:
                sym_idx = sym_algebs_ext.index(sym)
                Fy.append([eq_idx, sym_idx, expr.diff(sym)])
            elif sym in sym_states_ext:
                sym_idx = sym_states_ext.index(sym)
                Fx.append([eq_idx, sym_idx, expr.diff(sym)])
    """Save equations into callable CVXOPT functions"""
    fcall, gcall = [], []
    gycall, fxcall, jac0 = [], [], []
    jac0_line = 'dae.add_jac({}0, {}, self.{}, self.{})'
    call_line = 'dae.add_jac({}, {}, self.{}, self.{})'

    # format f and g equations
    fcall_anti_windup = 'dae.anti_windup(self.{0}, {1}, {2})'

    for sym, eq in zip(sym_states, sym_f):
        string_eq = stringfy(eq, sym_consts, sym_states_ext, sym_algebs_ext)
        # handling anti_windup
        template = 'dae.f[self.{0}] = {1}'
        fcall.append(template.format(sym, string_eq))
        if sym in sym_anti_windup:
            val = eval('anti_windup[\'{}\']'.format(sym))
            if type(val[1]) not in (int, float):
                val[1] = 'self.' + val[1]
            if type(val[2]) not in (int, float):
                val[2] = 'self.' + val[2]
            fcall.append(fcall_anti_windup.format(sym, val[1], val[2]))

    gcall_windup = 'dae.windup(self.{0}, self.{1}, self.{2})'
    gcall_hard_limit = 'dae.hard_limit(self.{0}, {1}, {2})'

    for sym, eq in zip(sym_algebs + sym_interfaces, sym_g):
        string_eq = stringfy(eq, sym_consts, sym_states_ext, sym_algebs_ext)
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
            val_formatted = list(val)
            for idx, item in enumerate(val):
                if type(item) in (int, float):
                    pass
                else:
                    val_formatted[idx] = 'self.{}'.format(item)
            gcall.append(
                gcall_hard_limit.format(sym, val_formatted[0],
                                        val_formatted[1]))

    # format Jacobians
    jacobians = ['Gy', 'Gx', 'Fx', 'Fy']
    mapping = dict(
        F=sym_states_ext, G=sym_algebs_ext, y=sym_algebs_ext, x=sym_states_ext)

    for jac in jacobians:
        for item in eval(jac):
            eqname = mapping[jac[0]][item[0]]
            varname = mapping[jac[1]][item[1]]
            equation = item[2]
            try:
                free_syms = equation.free_symbols
            except AttributeError:
                free_syms = []

            string_eq = stringfy(equation, sym_consts, sym_states_ext,
                                 sym_algebs_ext)

            isjac0 = 1
            for sym in free_syms:
                if sym in sym_consts:
                    continue
                elif sym in sym_algebs_ext + sym_states_ext:
                    isjac0 = 0
                    break
                else:
                    raise KeyError

            if isjac0:
                jac0.append(jac0_line.format(jac, string_eq, eqname, varname))
            else:
                if jac == 'Gy':
                    gycall.append(
                        call_line.format(jac, string_eq, eqname, varname))
                else:
                    fxcall.append(
                        call_line.format(jac, string_eq, eqname, varname))
    """Format initialization and service calls"""
    init1call = []
    for item in sym_init1:
        rhs = stringfy(item[1], sym_consts, sym_states_ext, sym_algebs_ext)
        if item[0] in sym_algebs_ext:
            out = 'dae.y[self.{}] = {}'.format(item[0], rhs)
        elif item[0] in sym_states_ext:
            out = 'dae.x[self.{}] = {}'.format(item[0], rhs)
        else:
            raise KeyError
        init1call.append(out)

    servcall = []
    for item in ctrl:
        key, val = item
        out = 'self.copy_data_ext(\'{}\', \'{}\', \'dest={}\', idx=self.{})'.format(
            val[0], val[1], key, val[2])
        servcall.append(out)
    for item in sym_serv:
        rhs = stringfy(item[1], sym_consts, sym_states_ext, sym_algebs_ext)
        out = 'self.{} = {}'.format(item[0], rhs)
        servcall.append(out)

    calls = {
        'gcall': not not gcall,
        'fcall': not not fcall,
        'gycall': not not gycall,
        'fxcall': not not fxcall,
        'jac0': (not not jac0) or (not not sym_algebs),
        'init1': (not not init1call) or (not not servcall),
    }
    """Build function call strings"""
    out_calls = []
    if servcall:
        out_calls.append(space4 + 'def servcall(self, dae):')
        for item in servcall:
            out_calls.append(space8 + item)
        out_calls.append('')

    if init1call or servcall:
        out_calls.append(space4 + 'def init1(self, dae):')
        out_calls.append(space8 + 'self.servcall(dae)')
        for item in init1call:
            out_calls.append(space8 + item)
        out_calls.append('')

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

    tinyGy = 'dae.add_jac(Gy0, 1e-6, self.{0}, self.{0})'
    if jac0 or sym_states:
        out_calls.append(space4 + 'def jac0(self, dae):')
        for item in jac0:
            out_calls.append(space8 + item)
        for item in sym_algebs:
            out_calls.append(space8 + tinyGy.format(item))
    """Class definitions in out_init"""
    # bulk update or extend of dict and list
    param_assign = space8 + 'self._{} = {}'
    list_extend = space8 + 'self._{}.extend({})'
    dict_update = space8 + 'self._{}.update(\n{})'

    out_init = list()  # def __init__ call strings
    out_init.append('from cvxopt import matrix, spmatrix')
    out_init.append('from cvxopt import mul, div, sin, cos, exp')
    out_init.append('from ..consts import *')
    out_init.append('from .base import ModelBase\n\n')
    out_init.append('class {}(ModelBase):'.format(name))
    if doc_string:
        out_init.append(space4 + "\"\"\"{}\"\"\"".format(doc_string))
    out_init.append(space4 + 'def __init__(self, system, name):')
    out_init.append(space8 + 'super().__init__(system, name)')
    if not group:
        print('*Error: Group name is not defined!')
    else:
        out_init.append(param_assign.format('group', add_quotes(group)))
    if name:
        out_init.append(param_assign.format('name', add_quotes(name)))

    meta_dict_upd = {
        'data': data,
        'units': units,
        'config_descr': descr,
        'ac': ac,
        'dc': dc,
        # 'ctrl': ctrl,
    }
    meta_list_ext = {
        'params': params,
        'algebs': algebs,
        'states': states,
        'fnamex': fnamex,
        'fnamey': fnamey,
        'service': service_keys,
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

    for key in sorted(meta_list_ext.keys()):
        val = meta_list_ext[key]
        if val:
            out_init.append(list_extend.format(key, val))
    for key in sorted(meta_dict_upd.keys()):
        val = meta_dict_upd[key]
        if val:
            out_init.append(
                dict_update.format(key, pprint.pformat(val, indent=12)))

    out_init.append(space8 + 'self.calls.update({})'.format(calls))

    out_init.append(space8 + 'self._init()')
    out_init.append('')

    # write to file
    try:
        fid = open(outfile, 'w')
        for line in out_init:
            fid.write(line + '\n')
        for line in out_calls:
            fid.write(line + '\n')
        fid.close()
    except IOError:
        logger.error('IOError while writing card output.')
        retval = False

    if retval:
        logger.info(
            'Card file successfully saved to <{}> with'.format(outfile))
        logger.info(
            '* constants: {}, algebs: {}, interfaces: {}, states: {}'.format(
                len(sym_consts), len(sym_algebs), len(interfaces),
                len(sym_states)))
        logger.info('* diff equations: {}, algeb equations: {}'.format(
            len(fcall), len(gcall)))
        logger.info('* fxcall: {}, gycall: {}, jac0: {}'.format(
            len(fxcall), len(gycall), len(jac0)))

    sys.exit(0)
    # return retval


def stringfy(expr, sym_const=None, sym_states=None, sym_algebs=None):
    """Convert the right-hand-side of an equation into CVXOPT matrix operations"""
    if not sym_const:
        sym_const = []
    if not sym_states:
        sym_states = []
    if not sym_algebs:
        sym_algebs = []
    expr_str = []
    if type(expr) in (int, float):
        return expr
    if expr.is_Atom:
        if expr in sym_const:
            expr_str = 'self.{}'.format(expr)
        elif expr in sym_states:
            expr_str = 'dae.x[self.{}]'.format(expr)
        elif expr in sym_algebs:
            expr_str = 'dae.y[self.{}]'.format(expr)
        elif expr.is_Number:
            if expr.is_Integer:
                expr_str = str(int(expr))
            else:
                expr_str = str(float(expr))
            # if expr.is_negative:
            #     expr_str = '{}'.format(expr)
            # else:
            #     expr_str = str(expr)
        else:
            raise AttributeError('Unknown free symbol <{}>'.format(expr))
    else:
        nargs = len(expr.args)
        arg_str = []
        for arg in expr.args:
            arg_str.append(stringfy(arg, sym_const, sym_states, sym_algebs))

        if expr.is_Add:
            expr_str = ''
            for idx, item in enumerate(arg_str):
                if idx == 0:
                    if len(item) > 1 and item[1] == ' ':
                        item = item[0] + item[2:]
                if idx > 0:
                    if item[0] == '-':
                        item = ' ' + item
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
            if arg_str[1] == '-1':
                expr_str = 'div(1, {})'.format(arg_str[0])
            else:
                expr_str = '({})**{}'.format(*arg_str)
        elif expr.is_Div:
            expr_str = ', '.join(arg_str)
            expr_str = 'div(' + expr_str + ')'
        else:
            raise NotImplementedError
    return expr_str
