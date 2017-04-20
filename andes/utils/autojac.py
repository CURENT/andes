from sympy import Symbol, diff, sin, cos

outfile = 'eq_out.ext'
consts = ['rsh', 'xsh', 'iLsh', 'pref0',
          'iM', 'D', 'iTt',
          'Kp1', 'Ki1', 'Kp2', 'Ki2', 'Kp3', 'Ki3', 'Kp4', 'Ki4']
algebs = ['wref', 'vref', 'P','Q', 'usd', 'usq',
          'Idref', 'Iqref', 'vdref', 'vqref', 'a', 'v']
states = ['Id', 'Iq', 'ucd', 'ucq', 'adq',
          'Md', 'Mq', 'Nd', 'Nq', 'xw',]

# algeb_eq = ['1',  # wref
#             '1',  # vref
#             '1',  # P
#             '1',  # Q
#             'v * cos(a - adq)',  # usd
#             'v * sin(a - adq)',  # usq
#             'Kp3 * (vref - usd) + Nd - Idref',  # Idref
#             '- Kp4 * usq + Nq - Iqref',  # Iqref
#             'usd + xsh*Iqref + Kp1*(Idref - Id) + Md - vdref',  # vdref
#             'usq - xsh*Idref + Kp2*(Iqref - Iq) + Mq - vqref',  # vqref
#             '1',  # a
#             '1',  # v
#             ]
algeb_eq = []
diff_eq = ['-rsh*iLsh*Id + Iq + iLsh*(ucd - usd)',  # Id
           '-rsh*iLsh*Iq - Id + iLsh*(ucq - usq)',  # Iq
           'iTt*(vdref - ucd)',  # ucd
           'iTt*(vqref - ucq)',  # ucq
           'xw',                # adq
           'Ki1*(Idref - Id)',  # Md
           'Ki2*(Iqref - Iq)',  # Mq
           'Ki3*(vref - usd)',  # Nd
           'Ki4*(-usq)',        # Nq
           'iM*(pref0 - P - D*xw)']  # xw


symbol_consts = []
symbol_algebs = []
symbol_states = []


Fx = []
Fy = []
Gx = []
Gy = []

jacobians = ['Gy', 'Gx', 'Fx', 'Fy']

gycall = []
fxcall = []
jac0 = []

jac0_line = 'dae.add_jac({}0, {}, self.{}, self.{})'
jac_line = 'dae.add_jac({}, {}, self.{}, self.{})'

var_names = consts + algebs + states
if len(set(var_names)) != len(var_names):
    raise NameError('Duplicated names are declared!')

for item in consts:
    call = '{} = Symbol(item)'.format(item)
    exec(call)
    symbol_consts.append(eval(item))

for item in algebs:
    call = '{} = Symbol(item)'.format(item)
    exec(call)
    symbol_algebs.append(eval(item))

for item in states:
    call = '{} = Symbol(item)'.format(item)
    exec(call)
    symbol_states.append(eval(item))

for eq_idx, item in enumerate(algeb_eq):
    exec('g = {}'.format(item))
    try:
        free_syms = g.free_symbols
    except AttributeError:
        free_syms = []

    for sym in free_syms:
        if sym in symbol_algebs:
            # take derivative and go to Gy
            sym_idx = symbol_algebs.index(sym)
            Gy.append([eq_idx, sym_idx, g.diff(sym)])
        elif sym in symbol_states:
            # take derivative and go to Gx
            sym_idx = symbol_states.index(sym)
            Gx.append([eq_idx, sym_idx, g.diff(sym)])

for eq_idx, item in enumerate(diff_eq):
    exec('f = {}'.format(item))
    try:
        free_syms = f.free_symbols
    except:
        free_syms = []

    for sym in free_syms:
        if sym in symbol_algebs:
            sym_idx = symbol_algebs.index(sym)
            Fy.append([eq_idx, sym_idx, f.diff(sym)])
        elif sym in symbol_states:
            sym_idx = symbol_states.index(sym)
            Fx.append([eq_idx, sym_idx, f.diff(sym)])

mapping = {'F': symbol_states,
           'G': symbol_algebs,
           'y': symbol_algebs,
           'x': symbol_states}

for jac in jacobians:
    for item in eval(jac):
        eqname = mapping[jac[0]][item[0]]
        varname = mapping[jac[1]][item[1]]
        equation = item[2]
        free_syms = equation.free_symbols

        isjac0 = 1
        for sym in free_syms:
            if sym in symbol_consts:
                continue
            elif sym in symbol_algebs:
                isjac0 = 0
                break
            elif sym in symbol_states:
                isjac0 = 0
                break
            else:
                raise KeyError

        if isjac0:
            jac0.append(jac0_line.format(jac, equation, eqname, varname))
        else:
            if jac == 'Gy':
                gycall.append(jac_line.format(jac, equation, eqname, varname))
            else:
                fxcall.append(jac_line.format(jac, equation, eqname, varname))

fid = open(outfile, 'w')
for item in gycall:
    fid.writelines(item + '\n')
fid.writelines('\n')

for item in fxcall:
    fid.writelines(item + '\n')
fid.writelines('\n')

for item in jac0:
    fid.writelines(item + '\n')

fid.close()
