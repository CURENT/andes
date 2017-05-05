"""PSS/E file parser"""

import re
from python_utils import converters
from cvxopt.base import matrix
from operator import itemgetter

from ..consts import *


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

    blocks = ['bus', 'load', 'fshunt', 'gen', 'branch', 'transf', 'area',
              'twotermdc', 'vscdc', 'impedcorr', 'mtdc', 'msline', 'zone',
              'interarea', 'owner', 'facts', 'swshunt', 'gne', 'Q']
    nol = [1, 1, 1, 1, 1, 4, 1,
           0, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0]
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
            system.Settings.mva = mva
            system.Settings.freq = float(data[5])
            version = int(data[2])

            if not version:
                version = int(rawd.search(line).group(0).strip('rawd'))
            if version < 32 or version > 33:
                system.Log.warning('RAW file version is not 32 or 33. Error may occur.')
            continue
        elif num == 1:  # store the case info line
            system.Log.info(line)
            continue
        elif num == 2:
            continue
        elif num >= 3:
            if line[0:2] == '0 ':  # end of block
                b += 1
                continue
            elif line[0] is 'Q':  # end of file
                break
            data = line.split(',')

        data = [to_number(item) for item in data]
        mdata.append(data)
        mline += 1
        if mline == nol[b]:
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
        param = {'idx': idx,
                 'name': data[1],
                 'Vn': data[2],
                 'voltage': data[7],
                 'angle': a0,
                 'area': data[4],
                 'region': data[5],
                 'owner': data[6],
                 }
        system.Bus.add(**param)

    for data in raw['load']:
        """version 32:
          0,  1,      2,    3,    4,    5,    6,      7,   8,  9, 10,   11
        Bus, Id, Status, Area, Zone, PL(MW), QL (MW), IP, IQ, YP, YQ, OWNER
        """
        bus = data[0]
        vn = system.Bus.get_by_idx('Vn', bus)
        voltage = system.Bus.get_by_idx('voltage', bus)
        param = {'bus': bus,
                 'Vn': vn,
                 'Sn': mva,
                 'p': (data[5] + data[7] * voltage + data[9] * voltage ** 2) / mva,
                 'q': (data[6] + data[8] * voltage - data[10] * voltage ** 2) / mva,
                 'owner': data[11],
                 }
        system.PQ.add(**param)

    for data in raw['fshunt']:
        """
        0,    1,      2,      3,      4
        Bus, name, Status, g (MW), b (Mvar)
        """
        param = {'bus': data[0],
                 'u': data[2],
                 'Sn': mva,
                 'g': data[3] / mva,
                 'b': data[4] / mva,
                 }
        system.Shunt.add(**param)

    gen_idx = 0
    for data in raw['gen']:
        """
         0, 1, 2, 3, 4, 5, 6, 7,    8,   9,10,11, 12, 13, 14,   15, 16,17,18,19
         I,ID,PG,QG,QT,QB,VS,IREG,MBASE,ZR,ZX,RT,XT,GTAP,STAT,RMPCT,PT,PB,O1,F1
        """
        gen_mva = data[8]  # unused yet
        gen_idx += 1
        param = {'Sn': gen_mva,
                 'Vn': system.Bus.get_by_idx('Vn', data[0]),
                 'u': data[14],
                 'idx': gen_idx,
                 'bus': data[0],
                 'pg': data[2] / gen_mva,
                 'qg': data[3] / gen_mva,
                 'qmax': data[4] / gen_mva,
                 'qmin': data[5] / gen_mva,
                 'v0': data[6],
                 'ra': data[9],  # ra  armature resistance
                 'xs': data[10],  # xs synchronous reactance
                 'pmax': data[16] / gen_mva,
                 'pmin': data[17] / gen_mva,
                 }
        if data[0] in sw.keys():
            param.update({'a0': sw[data[0]],
                          })
            system.SW.add(**param)
        else:
            system.PV.add(**param)

    for data in raw['branch']:
        """
        I,J,CKT,R,X,B,RATEA,RATEB,RATEC,GI,BI,GJ,BJ,ST,LEN,O1,F1,...,O4,F4
        """
        param = {'bus1': data[0],
                 'bus2': data[1],
                 'r': data[3],
                 'x': data[4],
                 'b': data[5],
                 'rate_a': data[6],
                 }
        system.Line.add(**param)

    for data in raw['transf']:
        """
        I,J,K,CKT,CW,CZ,CM,MAG1,MAG2,NMETR,'NAME',STAT,O1,F1,...,O4,F4
        R1-2,X1-2,SBASE1-2
        WINDV1,NOMV1,ANG1,RATA1,RATB1,RATC1,COD1,CONT1,RMA1,RMI1,VMA1,VMI1,NTP1,TAB1,CR1,CX1
        WINDV2,NOMV2
        """
        param = {'trasf': True,
                 'bus1': data[0][0],
                 'bus2': data[0][1],
                 'u': data[0][11],
                 'b': data[0][8],
                 'r': data[1][0],
                 'x': data[1][1],
                 'tap': data[2][0],
                 'phi': data[2][2],
                 'rate_a': data[2][3],
                 }
        system.Line.add(**param)
    return retval


def readadd(file, system):
    """read DYR file"""
    dyr = {}
    data = []
    end = 0
    retval = True

    fid = open(file, 'r')
    for line in fid.readlines():
        if line.find('/') >= 0:
            line = line.split('/')[0]
            end = 1
        sep = ',' if line.find(',') >= 0 else ''  # mixed comma and space splitter not allowed

        line = [to_number(item.strip()) for item in line.split(sep)]
        data.extend(line)
        if end == 1:
            field = data[1]
            if field not in dyr.keys():
                dyr[field] = []
            dyr[field].append(data)
            end = 0
            data = []
    fid.close()

    # add device elements to system
    for data in dyr['GENCLS']:
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
        param = {'bus': bus,
                 'gen': gen_idx,
                 'Sn': system.__dict__[dev].get_by_idx('Sn', gen_idx),
                 'Vn': system.__dict__[dev].get_by_idx('Vn', gen_idx),
                 'xd1': system.__dict__[dev].get_by_idx('xs', gen_idx),
                 'ra': system.__dict__[dev].get_by_idx('ra', gen_idx),
                 'M': 2 * data[0],
                 'D': data[1],
                 }
        system.Syn2.add(**param)

    return retval


def to_number(s):
    """Convert a string to a number. If not successful, return the string without blanks"""
    ret = s
    try:
        ret = float(s)
    except ValueError:
        ret = ret.strip('\'').strip()
        return ret

    try:
        ret = int(s)
    except ValueError:
        pass
    return ret
