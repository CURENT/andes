"""
PSS/E file parser.

Include a RAW parser and a DYR parser.
"""

import logging
import re
import os

from andes.shared import deg2rad, pd, yaml
from andes.utils.misc import to_number
from collections import defaultdict
logger = logging.getLogger(__name__)


def testlines(fid):
    """
    Check the raw file for frequency base
    """
    first = fid.readline()
    first = first.strip().split('/')
    first = first[0].split(',')

    # get raw file version
    if len(first) >= 3:
        version = int(first[2])
        logger.debug(f'PSSE raw version {version} detected')

        if version < 32 or version > 33:
            logger.warning('RAW file is not v32 or v33. Errors may occur.')

        return True

    else:
        return False


def get_block_lines(b, mdata):
    """
    Return the number of lines based on data
    """
    line_counts = [1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]

    if b == 5:  # for transformer
        if mdata[0][2] == 0:  # two-winding transformer
            return 4
        else:  # three-winding transformer
            return 5

    return line_counts[b]


def read(system, file):
    """read PSS/E RAW file v32 format"""

    blocks = [
        'bus', 'load', 'fshunt', 'gen', 'branch', 'transf', 'area',
        'twotermdc', 'vscdc', 'impedcorr', 'mtdc', 'msline', 'zone',
        'interarea', 'owner', 'facts', 'swshunt', 'gne', 'Q'
    ]
    rawd = re.compile(r'rawd\d\d')

    ret = True
    block_idx = 0  # current block index
    mva = 100

    raw = {}
    for item in blocks:
        raw[item] = []

    data = []
    mdata = []  # multi-line data
    dev_line = 0  # line counter for multi-line models

    # read file into `line_list`
    with open(file, 'r') as f:
        line_list = [line.rstrip('\n') for line in f]

    # parse file into `raw` with to_number conversions
    for num, line in enumerate(line_list):
        line = line.strip()
        # get basemva and nominal frequency
        if num == 0:
            data = line.split('/')[0]
            data = data.split(',')

            mva = float(data[1])
            system.config.mva = mva
            try:
                system.config.freq = float(data[5])
            except IndexError:
                logger.warning('System frequency is set to 60 Hz.\n'
                               'Consider using a higher version PSS/E raw file.')
                system.config.freq = 60.0

            # get raw file version
            version = 0
            if len(data) >= 3:
                version = int(data[2])
            else:
                if rawd.search(line):
                    version = int(rawd.search(line).group(0).strip('rawd'))  # NOQA

            continue

        elif num == 1 or num == 2:  # store the case info line
            if len(line) > 0:
                logger.info("  " + line)
            continue
        elif num >= 3:
            if line[0:2] == '0 ' or line[0:3] == ' 0 ':  # end of block
                block_idx += 1
                continue
            elif line[0] == 'Q':  # end of file
                break
            data = line.split(',')

        data = [to_number(item) for item in data]
        mdata.append(data)
        dev_line += 1

        block_lines = get_block_lines(block_idx, mdata)
        if dev_line >= block_lines:
            if block_lines == 1:
                mdata = mdata[0]
            raw[blocks[block_idx]].append(mdata)
            mdata = []
            dev_line = 0

    # add device elements to system
    bus_params, bus_idx_list, sw = _parse_bus_v33(raw, system)
    max_bus = max(bus_idx_list)

    _parse_load_v33(raw, system)
    _parse_fshunt_v33(raw, system)
    _parse_gen_v33(raw, system, sw)
    _parse_line_v33(raw, system)
    _parse_transf_v33(raw, system, max_bus)
    _parse_swshunt_v33(raw, system)
    _parse_area_v33(raw, system)

    return ret


def _read_dyr_dict(file):
    """
    Parse dyr file into a dict where keys are model names and values are dataframes.
    """
    with open(file, 'r') as f:
        input_list = [line.strip() for line in f]

    # concatenate multi-line device data
    input_concat_dict = defaultdict(list)
    multi_line = list()
    for i, line in enumerate(input_list):
        if line == '':
            continue
        if '/' not in line:
            multi_line.append(line)
        else:
            multi_line.append(line.split('/')[0])
            single_line = ' '.join(multi_line)

            if single_line.strip() == '':
                continue

            single_list = single_line.split("'")

            psse_model = single_list[1].strip()
            input_concat_dict[psse_model].append(single_list[0] + single_list[2])
            multi_line = list()

    # construct pandas dataframe for all models
    dyr_dict = dict()   # input data from dyr file

    for psse_model, all_rows in input_concat_dict.items():
        dev_params_num = [([to_number(cell) for cell in row.split()]) for row in all_rows]
        dyr_dict[psse_model] = pd.DataFrame(dev_params_num)

    return dyr_dict


def read_add(system, file):
    """
    Read an addition PSS/E dyr file.

    Parameters
    ----------
    system : System
        System instance to which data will be loaded
    file : str
        Path to the additional `dyr` file

    Returns
    -------
    bool
        data parsing status
    """
    dyr_dict = _read_dyr_dict(file)
    system.dyr_dict = dyr_dict

    # read yaml and set header for each pss/e model
    dirname = os.path.dirname(__file__)
    with open(f'{dirname}/psse-dyr.yaml', 'r') as f:
        dyr_yaml = yaml.full_load(f)

    sorted_models = sort_psse_models(dyr_yaml)

    for psse_model in dyr_dict:
        if psse_model in dyr_yaml:
            if 'inputs' in dyr_yaml[psse_model]:
                dyr_dict[psse_model].columns = dyr_yaml[psse_model]['inputs']

    # collect not supported models
    not_supported = []
    for model in dyr_dict:
        if model not in sorted_models:
            not_supported.append(model)

    # print out debug messages
    if len(dyr_dict):
        logger.debug(f'dyr contains models {", ".join(dyr_dict.keys())}')

    if len(not_supported):
        logger.warning(f'Models not yet supported: {", ".join(not_supported)}')
    else:
        logger.debug('All dyr models are supported.')

    # load data into models
    for psse_model in sorted_models:
        if psse_model not in dyr_dict:
            # device not exist
            continue

        if psse_model not in dyr_yaml:
            logger.error(f"PSS/E Model <{psse_model}> is not supported.")
            continue

        logger.debug(f'Parsing PSS/E model {psse_model}')

        dest = dyr_yaml[psse_model]['destination']
        find = {}

        if 'find' in dyr_yaml[psse_model]:
            for name, source in dyr_yaml[psse_model]['find'].items():

                for model, conditions in source.items():
                    allow_none = conditions.pop('allow_none', 0)
                    cond_names = conditions.keys()
                    cond_values = []

                    for col in conditions.values():
                        if col in find:
                            cond_values.append(find[col])
                        else:
                            cond_values.append(dyr_dict[psse_model][col])

                    try:
                        find[name] = system.__dict__[model].find_idx(cond_names, cond_values,
                                                                     allow_none=allow_none)
                    except IndexError as e:
                        logger.error("Data file likely contains references to unsupported models.")
                        logger.error(e)
                        return False

        if 'get' in dyr_yaml[psse_model]:
            for name, source in dyr_yaml[psse_model]['get'].items():
                for model, conditions in source.items():
                    idx_name = conditions['idx']
                    if idx_name in dyr_dict[psse_model]:
                        conditions['idx'] = dyr_dict[psse_model][idx_name]
                    else:
                        conditions['idx'] = find[idx_name]
                    find[name] = system.__dict__[model].get(**conditions)

        if 'outputs' in dyr_yaml[psse_model]:
            output_keys = list(dyr_yaml[psse_model]['outputs'].keys())
            output_exprs = list(dyr_yaml[psse_model]['outputs'].values())
            out_dict = {}

            for idx in range(len(output_exprs)):
                out_key = output_keys[idx]
                expr = output_exprs[idx]
                if expr in find:
                    out_dict[out_key] = find[expr]
                elif ';' in expr:
                    args, func = expr.split(';')
                    func = eval(func)
                    args = args.split(',')
                    argv = [pairs.split('.') for pairs in args]
                    argv = [dyr_dict[model][param] for model, param in argv]
                    out_dict[output_keys[idx]] = func(*argv)
                else:
                    out_dict[output_keys[idx]] = dyr_dict[psse_model][expr]

            df = pd.DataFrame.from_dict(out_dict)
            for row in df.to_dict(orient='records'):
                system.add(dest, row)

        system.link_ext_param(system.__dict__[dest])

    return True


def _parse_bus_v33(raw, system):
    # version 32:
    #   0,   1,      2,     3,    4,   5,  6,   7,  8
    #   ID, NAME, BasekV, Type, Area Zone Owner Vm, Va
    #
    out = defaultdict(list)
    bus_idx_list = list()
    sw = dict()

    for data in raw['bus']:

        idx = data[0]
        bus_idx_list.append(idx)
        ty = data[3]
        a0 = data[8] * deg2rad

        if ty == 3:
            sw[idx] = a0
        param = {'idx': idx, 'name': data[1], 'Vn': data[2],
                 'v0': data[7], 'a0': a0,
                 'area': data[4], 'zone': data[5], 'owner': data[6]}
        out['Bus'].append(param)

    _add_devices_from_dict(out, system)

    return out, bus_idx_list, sw


def _parse_load_v33(raw, system):
    # version 32:
    #  0,  1,      2,    3,    4,    5,    6,      7,   8,  9, 10,   11
    # Bus, Id, Status, Area, Zone, PL(MW), QL (MW), IP, IQ, YP, YQ, OWNER

    mva = system.config.mva
    out = defaultdict(list)

    for data in raw['load']:
        bus = data[0]

        vn = system.Bus.get(src='Vn', idx=bus, attr='v')
        v0 = system.Bus.get(src='v0', idx=bus, attr='v')

        param = {'bus': bus, 'u': data[2], 'Vn': vn,
                 'p0': (data[5] + data[7] * v0 + data[9] * v0 ** 2) / mva,
                 'q0': (data[6] + data[8] * v0 - data[10] * v0 ** 2) / mva,
                 'owner': data[11]}

        out['PQ'].append(param)

    _add_devices_from_dict(out, system)

    return out


def _parse_fshunt_v33(raw, system):
    # 0,    1,      2,      3,      4
    # Bus, name, Status, g (MW), b (Mvar)

    mva = system.config.mva
    out = defaultdict(list)

    for data in raw['fshunt']:
        bus = data[0]
        vn = system.Bus.get(src='Vn', idx=bus, attr='v')

        param = {'bus': bus, 'Vn': vn, 'u': data[2],
                 'Sn': mva, 'g': data[3] / mva, 'b': data[4] / mva}

        out['Shunt'].append(param)

    _add_devices_from_dict(out, system)

    return out


def _parse_gen_v33(raw, system, sw):
    #  0, 1, 2, 3, 4, 5, 6, 7,    8,   9,10,11, 12, 13, 14,   15, 16,17,18,19
    #  I,ID,PG,QG,QT,QB,VS,IREG,MBASE,ZR,ZX,RT,XT,GTAP,STAT,RMPCT,PT,PB,O1,F1

    mva = system.config.mva
    out = defaultdict(list)
    gen_idx = 0

    for data in raw['gen']:

        bus = data[0]
        subidx = data[1]
        vn = system.Bus.get(src='Vn', idx=bus, attr='v')
        gen_mva = data[8]
        gen_idx += 1
        status = data[14]

        param = {'Sn': gen_mva, 'Vn': vn, 'u': status,
                 'bus': bus, 'subidx': subidx,
                 'idx': gen_idx,
                 'p0': data[2] / mva,
                 'q0': data[3] / mva,
                 'pmax': data[16] / mva, 'pmin': data[17] / mva,
                 'qmax': data[4] / mva, 'qmin': data[5] / mva,
                 'v0': data[6],
                 'ra': data[9],   # ra - armature resistance
                 'xs': data[10],  # xs - synchronous reactance
                 }

        if data[0] in sw.keys():
            param.update({'a0': sw[data[0]]})
            out['Slack'].append(param)
        else:
            out['PV'].append(param)

    _add_devices_from_dict(out, system)

    return out


def _parse_line_v33(raw, system):
    #
    # I,J,CKT,R,X,B,RATEA,RATEB,RATEC,GI,BI,GJ,BJ,ST,LEN,O1,F1,...,O4,F4
    #

    out = defaultdict(list)
    for data in raw['branch']:
        param = {
            'bus1': data[0], 'bus2': data[1],
            'r': data[3], 'x': data[4], 'b': data[5],
            'Vn1': system.Bus.get(src='Vn', idx=data[0], attr='v'),
            'Vn2': system.Bus.get(src='Vn', idx=data[1], attr='v'),
        }
        out['Line'].append(param)

    _add_devices_from_dict(out, system)

    return out


def _parse_transf_v33(raw, system, max_bus):

    out = defaultdict(list)
    xf_3_count = 1

    for data in raw['transf']:
        if len(data) == 4:

            # """
            # I,J,K,CKT,CW,CZ,CM,MAG1,MAG2,NMETR,'NAME',STAT,O1,F1,...,O4,F4
            # R1-2,X1-2,SBASE1-2
            # WINDV1,NOMV1,ANG1,RATA1,RATB1,RATC1,COD1,CONT1,RMA1,RMI1,VMA1,VMI1,NTP1,TAB1,CR1,CX1
            # WINDV2,NOMV2
            #
            # """

            Sn = system.config.mva
            bus_Vn1 = system.Bus.get(src='Vn', idx=data[0][0], attr='v')
            bus_Vn2 = system.Bus.get(src='Vn', idx=data[0][1], attr='v')

            Vn1 = data[2][1] if data[2][1] != 0.0 else bus_Vn1
            Vn2 = data[3][1] if data[3][1] != 0.0 else bus_Vn2
            transf = True
            tap = data[2][0]  # pu or in kV
            phi = data[2][2] * deg2rad  # `ANG1` is entered in degree; convert to rad

            # CW - Winding I/O code, 1-turn ratio on pu bus base kV, 2: winding V, 3: turn ratio pu on norn wind V
            if data[0][4] == 1:
                tap = tap
            elif data[0][4] == 2:
                tap = (data[2][0] / bus_Vn1) / (data[3][0] / bus_Vn2)
            else:
                tap = tap * (Vn1 / bus_Vn1) / (Vn2 / bus_Vn2)

            # CZ - Z code, 1-system base, 2-winding base, 3-load loss and |z|
            if data[0][5] == 1:
                Sn = system.config.mva
            elif data[0][5] == 2:
                Sn = data[1][2]
            else:
                raise NotImplementedError('Impedance code 3 not implemented')

            # CM - Y code, 1-system base, 2-No load loss and exc. loss
            if data[0][6] == 2:
                raise NotImplementedError('Admittance code 2 not implemented')

            param = {'bus1': data[0][0],
                     'bus2': data[0][1],
                     'u': data[0][11],
                     'b': data[0][8],
                     'r': data[1][0],
                     'x': data[1][1],
                     'trans': transf,
                     'tap': tap,
                     'phi': phi,
                     'Sn': Sn,
                     'Vn1': Vn1,
                     'Vn2': Vn2,
                     }

            out['Line'].append(param)

        else:

            # I, J, K, CKT, CW, CZ, CM, MAG1, MAG2, NMETR, 'NAME', STAT, Ol, Fl,...,o4, F4
            # R1-2, X1-2, SBASE1-2, R2-3, X2-3, SBASE2-3, R3-1, X3-1, SBASE3-1, VMSTAR, ANSTAR
            # WINDV1, NOMV1, ANG1, RATA1, BATB1, RATC1, COD1, CONT1, RMA1, RMI1, VMA1, VMI1, NTP1, TAB1, CR1, CX1
            # WINDV2, NOMV2, ANG2, RATA2, BATB2, RATC2, COD2, CONT2, RMA2, RMI2, VMA2, VMI2, NTP2, TAB2, CR2, CX2
            # WINDV3, NOMV3, ANG3, RATA3, BATB3, RATC3, COD3, CONT3, RMA3, RMI3, VMA3, VMI3, NTP3, TAB3, CR3, CX3

            new_bus = data[0][2] + 1

            if new_bus in system.Bus.idx.v:
                new_bus = max_bus + xf_3_count
                logger.warning(f'{new_bus} exists.')

            param = {'idx': new_bus,
                     'name': '_'.join([str(i) for i in data[0][:3]]),
                     'Vn': 1.0,
                     'v0': data[1][-2],
                     'a0': data[1][-1] * deg2rad
                     }

            out['Bus'].append(param)

            r = []
            x = []
            r.append((data[1][0] + data[1][6] - data[1][3])/2)
            r.append((data[1][3] + data[1][0] - data[1][6])/2)
            r.append((data[1][6] + data[1][3] - data[1][0])/2)
            x.append((data[1][1] + data[1][7] - data[1][4])/2)
            x.append((data[1][4] + data[1][1] - data[1][7])/2)
            x.append((data[1][7] + data[1][4] - data[1][1])/2)

            for i in range(0, 3):
                param = {'trans': True,
                         'bus1': data[0][i],
                         'bus2': new_bus,
                         'u': data[0][11],
                         'b': data[0][8],
                         'r': r[i],
                         'x': x[i],
                         'tap': data[2+i][0],
                         'phi': data[2+i][2] * deg2rad,
                         'Vn1': system.Bus.get(src='Vn', idx=data[0][i], attr='v'),
                         'Vn2': 1.0,
                         }

                out['Line'].append(param)

            xf_3_count += 1

    _add_devices_from_dict(out, system)

    return out, xf_3_count


def _parse_swshunt_v33(raw, system):

    # I, MODSW, ADJM, STAT, VSWHI, VSWLO, SWREM, RMPCT, RMIDNT,
    # BINIT, N1, B1, N2, B2, ... N8, B8

    out = defaultdict(list)
    mva = system.config.mva
    for data in raw['swshunt']:
        bus = data[0]
        vn = system.Bus.get(src='Vn', idx=bus, attr='v')
        param = {'bus': bus, 'Vn': vn, 'Sn': mva, 'u': data[3],
                 'b': data[9] / mva}

        out['Shunt'].append(param)

    _add_devices_from_dict(out, system)

    return out


def _parse_area_v33(raw, system):

    out = defaultdict(list)

    for data in raw['area']:

        # ID, ISW, PDES, PTOL, ARNAME

        param = {'idx': data[0], 'name': data[4],
                 # 'isw': data[1],
                 # 'pdes': data[2],
                 # 'ptol': data[3],
                 }

        out['Area'].append(param)

    for data in raw['zone']:
        # """ID, NAME"""
        param = {'idx': data[0], 'name': data[1]}

        # TODO: add back
        # system.add('Zone', param)

    _add_devices_from_dict(out, system)

    return out


def _add_devices_from_dict(params, system):
    """
    Add devices from a dict, where the key is the
    model name, and the value is a list of parameters.

    """
    for name, plist in params.items():
        for p in plist:
            system.add(name, p)


def sort_psse_models(dyr_yaml):
    """
    Sort supported models so that model names are ordered by dependency.
    """
    from andes.models import non_jit
    from andes.utils.func import list_flatten

    andes_models = list_flatten(list(non_jit.values()))
    number = dict()

    for psse_model in dyr_yaml:
        dest = dyr_yaml[psse_model]['destination']
        if dest in andes_models:
            number[dest] = andes_models.index(dest)

    sorted_models = [k for k, v in sorted(number.items(), key=lambda item: item[1])]

    return sorted_models
