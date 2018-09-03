"""PSS/E file parser"""

import re

from ..consts import deg2rad
from ..utils.math import to_number
import logging

logger = logging.getLogger(__name__)


def testlines(fid):
    """Check the raw file for frequency base"""
    first = fid.readline()
    first = first.strip().split('/')
    first = first[0].split(',')
    if float(first[5]) == 50.0 or float(first[5]) == 60.0:
        return True
    else:
        return False


def read(file, system):
    """read PSS/E RAW file v32 format"""

    blocks = [
        'bus', 'load', 'fshunt', 'gen', 'branch', 'transf', 'area',
        'twotermdc', 'vscdc', 'impedcorr', 'mtdc', 'msline', 'zone',
        'interarea', 'owner', 'facts', 'swshunt', 'gne', 'Q'
    ]
    nol = [1, 1, 1, 1, 1, 4, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    rawd = re.compile('rawd\d\d')

    retval = True
    version = 0
    b = 0  # current block index
    raw = {}
    for item in blocks:
        raw[item] = []

    data = []
    mdata = []  # multi-line data
    mline = 0  # line counter for multi-line models

    # parse file into raw with to_number conversions
    fid = open(file, 'r')
    for num, line in enumerate(fid.readlines()):
        line = line.strip()
        if num == 0:  # get basemva and frequency
            data = line.split('/')[0]
            data = data.split(',')

            mva = float(data[1])
            system.config.mva = mva
            system.config.freq = float(data[5])
            version = int(data[2])

            if not version:
                version = int(rawd.search(line).group(0).strip('rawd'))
            if version < 32 or version > 33:
                logger.warning(
                    'RAW file version is not 32 or 33. Error may occur.')
            continue
        elif num == 1:  # store the case info line
            logger.info(line)
            continue
        elif num == 2:
            continue
        elif num >= 3:
            if line[0:2] == '0 ' or line[0:3] == ' 0 ':  # end of block
                b += 1
                continue
            elif line[0] == 'Q':  # end of file
                break
            data = line.split(',')

        data = [to_number(item) for item in data]
        mdata.append(data)
        mline += 1
        if mline >= nol[b]:
            if nol[b] == 1:
                mdata = mdata[0]
            raw[blocks[b]].append(mdata)
            mdata = []
            mline = 0
    fid.close()

    # add device elements to system
    sw = {}  # idx:a0
    for data in raw['bus']:
        """version 32:
          0,   1,      2,     3,    4,   5,  6,   7,  8
          ID, NAME, BasekV, Type, Area Zone Owner Va, Vm
        """
        idx = data[0]
        ty = data[3]
        a0 = data[8] * deg2rad
        if ty == 3:
            sw[idx] = a0
        param = {
            'idx': idx,
            'name': data[1],
            'Vn': data[2],
            'voltage': data[7],
            'angle': a0,
            'area': data[4],
            'zone': data[5],
            'owner': data[6],
        }
        system.Bus.elem_add(**param)

    for data in raw['load']:
        """version 32:
          0,  1,      2,    3,    4,    5,    6,      7,   8,  9, 10,   11
        Bus, Id, Status, Area, Zone, PL(MW), QL (MW), IP, IQ, YP, YQ, OWNER
        """
        bus = data[0]
        vn = system.Bus.get_field('Vn', bus)
        voltage = system.Bus.get_field('voltage', bus)
        param = {
            'bus': bus,
            'Vn': vn,
            'Sn': mva,
            'p': (data[5] + data[7] * voltage + data[9] * voltage**2) / mva,
            'q': (data[6] + data[8] * voltage - data[10] * voltage**2) / mva,
            'owner': data[11],
        }
        system.PQ.elem_add(**param)

    for data in raw['fshunt']:
        """
        0,    1,      2,      3,      4
        Bus, name, Status, g (MW), b (Mvar)
        """
        bus = data[0]
        vn = system.Bus.get_field('Vn', bus)
        param = {
            'bus': bus,
            'Vn': vn,
            'u': data[2],
            'Sn': mva,
            'g': data[3] / mva,
            'b': data[4] / mva,
        }
        system.Shunt.elem_add(**param)

    gen_idx = 0
    for data in raw['gen']:
        """
         0, 1, 2, 3, 4, 5, 6, 7,    8,   9,10,11, 12, 13, 14,   15, 16,17,18,19
         I,ID,PG,QG,QT,QB,VS,IREG,MBASE,ZR,ZX,RT,XT,GTAP,STAT,RMPCT,PT,PB,O1,F1
        """
        bus = data[0]
        vn = system.Bus.get_field('Vn', bus)
        gen_mva = data[8]  # unused yet
        gen_idx += 1
        status = data[14]
        param = {
            'Sn': gen_mva,
            'Vn': vn,
            'u': status,
            'idx': gen_idx,
            'bus': bus,
            'pg': status * data[2] / mva,
            'qg': status * data[3] / mva,
            'qmax': data[4] / mva,
            'qmin': data[5] / mva,
            'v0': data[6],
            'ra': data[9],  # ra  armature resistance
            'xs': data[10],  # xs synchronous reactance
            'pmax': data[16] / mva,
            'pmin': data[17] / mva,
        }
        if data[0] in sw.keys():
            param.update({
                'a0': sw[data[0]],
            })
            system.SW.elem_add(**param)
        else:
            system.PV.elem_add(**param)

    for data in raw['branch']:
        """
        I,J,CKT,R,X,B,RATEA,RATEB,RATEC,GI,BI,GJ,BJ,ST,LEN,O1,F1,...,O4,F4
        """
        param = {
            'bus1': data[0],
            'bus2': data[1],
            'r': data[3],
            'x': data[4],
            'b': data[5],
            'rate_a': data[6],
            'Vn': system.Bus.get_field('Vn', data[0]),
            'Vn2': system.Bus.get_field('Vn', data[1]),
        }
        system.Line.elem_add(**param)

    for data in raw['transf']:
        """
        I,J,K,CKT,CW,CZ,CM,MAG1,MAG2,NMETR,'NAME',STAT,O1,F1,...,O4,F4
        R1-2,X1-2,SBASE1-2
        WINDV1,NOMV1,ANG1,RATA1,RATB1,RATC1,COD1,CONT1,RMA1,RMI1,VMA1,VMI1,NTP1,TAB1,CR1,CX1
        WINDV2,NOMV2
        """
        if len(data[1]) < 5:
            ty = 2
        else:
            ty = 3
        if ty == 3:
            raise NotImplementedError(
                'Three-winding transformer not implemented')

        tap = data[2][0]
        phi = data[2][2]

        if tap == 1 and phi == 0:
            trasf = False
        else:
            trasf = True
        param = {
            'trasf': trasf,
            'bus1': data[0][0],
            'bus2': data[0][1],
            'u': data[0][11],
            'b': data[0][8],
            'r': data[1][0],
            'x': data[1][1],
            'tap': tap,
            'phi': phi,
            'rate_a': data[2][3],
            'Vn': system.Bus.get_field('Vn', data[0][0]),
            'Vn2': system.Bus.get_field('Vn', data[0][1]),
        }
        system.Line.elem_add(**param)

    for data in raw['swshunt']:
        # I, MODSW, ADJM, STAT, VSWHI, VSWLO, SWREM, RMPCT, ’RMIDNT’,
        # BINIT, N1, B1, N2, B2, ... N8, B8

        bus = data[0]
        vn = system.Bus.get_field('Vn', bus)
        param = {
            'bus': bus,
            'Vn': vn,
            'Sn': mva,
            'u': data[3],
            'b': data[9] / mva,
        }
        system.Shunt.elem_add(**param)

    for data in raw['area']:
        """ID, ISW, PDES, PTOL, ARNAME"""
        param = {
            'idx': data[0],
            'isw': data[1],
            'pdes': data[2],
            'ptol': data[3],
            'name': data[4],
        }
        system.Area.elem_add(**param)

    for data in raw['zone']:
        """ID, NAME"""
        param = {
            'idx': data[0],
            'name': data[1],
        }
        system.Zone.elem_add(**param)

    return retval


def readadd(file, system):
    """read DYR file"""
    dyr = {}
    data = []
    end = 0
    retval = True
    sep = ','

    fid = open(file, 'r')
    for line in fid.readlines():
        if line.find('/') >= 0:
            line = line.split('/')[0]
            end = 1
        if line.find(',') >= 0:  # mixed comma and space splitter not allowed
            line = [to_number(item.strip()) for item in line.split(sep)]
        else:
            line = [to_number(item.strip()) for item in line.split()]
        if not line:
            end = 0
            continue
        data.extend(line)
        if end == 1:
            field = data[1]
            if field not in dyr.keys():
                dyr[field] = []
            dyr[field].append(data)
            end = 0
            data = []
    fid.close()

    # elem_add device elements to system
    supported = [
        'GENROU',
        'GENCLS',
        'ESST3A',
        'ESDC2A',
        'SEXS',
        'EXST1',
        'ST2CUT',
        'IEEEST',
        'TGOV1',
    ]
    used = list(supported)
    for model in supported:
        if model not in dyr.keys():
            used.remove(model)
            continue
        for data in dyr[model]:
            add_dyn(system, model, data)

    needed = list(dyr.keys())
    for i in supported:
        if i in needed:
            needed.remove(i)

    logger.warning('Models currently unsupported: {}'.format(
        ', '.join(needed)))

    return retval


def add_dyn(system, model, data):
    """helper function to elem_add a device element to system"""
    if model == 'GENCLS':
        bus = data[0]
        data = data[3:]
        if bus in system.PV.bus:
            dev = 'PV'
            gen_idx = system.PV.idx[system.PV.bus.index(bus)]
        elif bus in system.SW.bus:
            dev = 'SW'
            gen_idx = system.SW.idx[system.SW.bus.index(bus)]
        else:
            raise KeyError
        # todo: check xl
        idx_PV = get_idx(system, 'StaticGen', 'bus', bus)
        u = get_param(system, 'StaticGen', 'u', idx_PV)
        param = {
            'bus': bus,
            'gen': gen_idx,
            # 'idx': bus,
            #  use `bus` for `idx`. Only one generator allowed on each bus
            'Sn': system.__dict__[dev].get_field('Sn', gen_idx),
            'Vn': system.__dict__[dev].get_field('Vn', gen_idx),
            'xd1': system.__dict__[dev].get_field('xs', gen_idx),
            'ra': system.__dict__[dev].get_field('ra', gen_idx),
            'M': 2 * data[0],
            'D': data[1],
            'u': u,
        }
        system.Syn2.elem_add(**param)

    elif model == 'GENROU':
        bus = data[0]
        data = data[3:]
        if bus in system.PV.bus:
            dev = 'PV'
            gen_idx = system.PV.idx[system.PV.bus.index(bus)]
        elif bus in system.SW.bus:
            dev = 'SW'
            gen_idx = system.SW.idx[system.SW.bus.index(bus)]
        else:
            raise KeyError
        idx_PV = get_idx(system, 'StaticGen', 'bus', bus)
        u = get_param(system, 'StaticGen', 'u', idx_PV)

        param = {
            'bus': bus,
            'gen': gen_idx,
            # 'idx': bus,
            #  use `bus` for `idx`. Only one generator allowed on each bus
            'Sn': system.__dict__[dev].get_field('Sn', gen_idx),
            'Vn': system.__dict__[dev].get_field('Vn', gen_idx),
            'ra': system.__dict__[dev].get_field('ra', gen_idx),
            'Td10': data[0],
            'Td20': data[1],
            'Tq10': data[2],
            'Tq20': data[3],
            'M': 2 * data[4],
            'D': data[5],
            'xd': data[6],
            'xq': data[7],
            'xd1': data[8],
            'xq1': data[9],
            'xd2': data[10],
            'xq2': data[10],  # xd2 = xq2
            'xl': data[11],
            'u': u,
        }
        system.Syn6a.elem_add(**param)

    elif model == 'ESST3A':
        bus = data[0]
        data = data[3:]
        syn = get_idx(system, 'Synchronous', 'bus', bus)
        param = {
            'syn': syn,
            'vrmax': data[8],
            'vrmin': data[9],
            'Ka': data[6],
            'Ta': data[7],
            'Tf': data[5],
            'Tr': data[0],
            'Kf': data[5],
            'Ke': 1,
            'Te': 1,
        }
        system.AVR1.elem_add(**param)

    elif model == 'ESDC2A':
        bus = data[0]
        data = data[3:]
        syn = get_idx(system, 'Synchronous', 'bus', bus)
        param = {
            'syn': syn,
            'vrmax': data[5],
            'vrmin': data[6],
            'Ka': data[1],
            'Ta': data[2],
            'Tf': data[10],
            'Tr': data[0],
            'Kf': data[9],
            'Ke': 1,
            'Te': data[8],
        }
        system.AVR1.elem_add(**param)

    elif model == 'EXST1':
        bus = data[0]
        data = data[3:]
        syn = get_idx(system, 'Synchronous', 'bus', bus)
        param = {
            'syn': syn,
            'vrmax': data[7],
            'vrmin': data[8],
            'Ka': data[5],
            'Ta': data[6],
            'Kf': data[10],
            'Tf': data[11],
            'Tr': data[0],
            'Te': data[4],
        }
        system.AVR1.elem_add(**param)

    elif model == 'SEXS':
        bus = data[0]
        data = data[3:]
        syn = get_idx(system, 'Synchronous', 'bus', bus)
        param = {
            'syn': syn,
            'vrmax': data[5],
            'vrmin': data[4],
            'K0': data[2],
            'T2': data[1],
            'T1': data[0],
            'Te': data[3],
        }
        system.AVR3.elem_add(**param)

    elif model == 'IEEEG1':
        bus = data[0]
        data = data[3:]
        syn = get_idx(system, 'Synchronous', 'bus', bus)

        pass

    elif model == 'TGOV1':
        bus = data[0]
        data = data[3:]
        syn = get_idx(system, 'Synchronous', 'bus', bus)
        param = {
            'gen': syn,
            'R': data[0],
            'T1': data[4],
            'T2': data[5],
        }
        system.TG2.elem_add(**param)

    elif model == 'ST2CUT':
        bus = data[0]
        data = data[3:]
        Ic1 = data[0]
        Ic2 = data[2]

        data = data[4:]
        syn = get_idx(system, 'Synchronous', 'bus', bus)
        avr = get_idx(system, 'AVR', 'syn', syn)
        param = {
            'avr': avr,
            'Ic1': Ic1,
            'Ic2': Ic2,
            'K1': data[0],
            'K2': data[1],
            'T1': data[2],
            'T2': data[3],
            'T3': data[4],
            'T4': data[5],
            'T5': data[6],
            'T6': data[7],
            'T7': data[8],
            'T8': data[9],
            'T9': data[10],
            'T10': data[11],
            'lsmax': data[12],
            'lsmin': data[13],
            'vcu': data[14],
            'vcl': data[15],
        }
        system.PSS1.elem_add(**param)

    elif model == 'IEEEST':
        bus = data[0]
        data = data[3:]
        Ic = data[0]

        data = data[2:]
        syn = get_idx(system, 'Synchronous', 'bus', bus)
        avr = get_idx(system, 'AVR', 'syn', syn)
        param = {
            'avr': avr,
            'Ic': Ic,
            'A1': data[0],
            'A2': data[1],
            'A3': data[2],
            'A4': data[3],
            'A5': data[4],
            'A6': data[5],
            'T1': data[6],
            'T2': data[7],
            'T3': data[8],
            'T4': data[9],
            'T5': data[10],
            'T6': data[11],
            'Ks': data[12],
            'lsmax': data[13],
            'lsmin': data[14],
            'vcu': data[15],
            'vcl': data[16],
        }
        system.PSS2.elem_add(**param)

    else:
        logger.warning('Skipping unsupported model <{}> on bus {}'.format(
            model, data[0]))


def get_idx(system, group, param, fkey):
    ret = None
    for key, item in system.devman.group.items():
        if key != group:
            continue
        for name, dev in item.items():
            int_id = system.__dict__[dev].uid[name]
            if system.__dict__[dev].__dict__[param][int_id] == fkey:
                ret = name
                break
    return ret


def get_param(system, group, param, fkey):
    ret = None
    for key, item in system.devman.group.items():
        if key != group:
            continue
        for name, dev in item.items():
            if name == fkey:
                int_id = system.__dict__[dev].uid[name]
                ret = system.__dict__[dev].__dict__[param][int_id]
    return ret
