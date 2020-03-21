""" Simple MATPOWER format parser
"""
import logging
import re

from andes.shared import deg2rad, np

logger = logging.getLogger(__name__)


def testlines(fid):
    return True  # hard coded


def read(system, file):
    """Read a MATPOWER data file into mpc and build andes device elements"""
    func = re.compile(r'function\s')
    mva = re.compile(r'\s*mpc.baseMVA\s*=\s*')
    bus = re.compile(r'\s*mpc.bus\s*=\s*\[?')
    gen = re.compile(r'\s*mpc.gen\s*=\s*\[')
    branch = re.compile(r'\s*mpc.branch\s*=\s*\[')
    area = re.compile(r'\s*mpc.areas\s*=\s*\[')
    gencost = re.compile(r'\s*mpc.gencost\s*=\s*\[')
    bus_name = re.compile(r'\s*mpc.bus_name\s*=\s*{')
    end = re.compile(r'\s*\];?')
    has_digit = re.compile(r'.*\d+\s*]?;?')

    ret = True
    field = None
    info = True

    base_mva = 100
    mpc = {
        'bus': [],
        'gen': [],
        'branch': [],
        'area': [],
        'gencost': [],
        'bus_name': [],
    }

    fid = open(file, 'r')

    for line in fid:
        line = line.strip().rstrip(';')
        if not line:
            continue
        elif func.search(line):  # skip function declaration
            continue
        elif len(line.split('%')[0]) == 0:
            if info is True:
                logger.info(line[1:])
                info = False
            else:
                continue
        elif mva.search(line):
            base_mva = float(line.split('=')[1])

        if not field:
            if bus.search(line):
                field = 'bus'
            elif gen.search(line):
                field = 'gen'
            elif branch.search(line):
                field = 'branch'
            elif area.search(line):
                field = 'area'
            elif gencost.search(line):
                field = 'gencost'
            elif bus_name.search(line):
                field = 'bus_name'
            else:
                continue
        elif end.search(line):
            field = None
            continue

        # parse mpc sections
        if field:
            if line.find('=') >= 0:
                line = line.split('=')[1]
            if line.find('[') >= 0:
                line = re.sub(r'\[', '', line)
            elif line.find('{') >= 0:
                line = re.sub(r'{', '', line)

            if line.find('\'') >= 0:  # bus_name
                line = line.split(';')
                data = [i.strip('\'').strip() for i in line]
                mpc['bus_name'].extend(data)
            else:
                if not has_digit.search(line):
                    continue
                line = line.split('%')[0].strip()
                line = line.split(';')
                for item in line:
                    if not has_digit.search(item):
                        continue
                    try:
                        data = np.array([float(val) for val in item.split()])
                    except Exception as e:
                        logger.error(f'Error parsing {system.files.case}')
                        raise e
                    mpc[field].append(data)

    fid.close()

    # convert mpc to np array
    mpc_array = dict()
    for key, val in mpc.items():
        mpc_array[key] = np.array(val)

    # list of buses with slack gen
    sw = []

    system.mva = base_mva

    for data in mpc['bus']:
        # idx  ty   pd   qd  gs  bs  area  vmag  vang  baseKV  zone  vmax  vmin
        # 0    1    2   3   4   5    6      7     8     9      10    11    12
        idx = int(data[0])
        ty = data[1]
        if ty == 3:
            sw.append(idx)
        pd = data[2] / base_mva
        qd = data[3] / base_mva
        gs = data[4] / base_mva
        bs = data[5] / base_mva
        area = data[6]
        vmag = data[7]
        vang = data[8] * deg2rad
        baseKV = data[9]
        if baseKV == 0:
            baseKV = 110
        zone = data[10]
        vmax = data[11]
        vmin = data[12]

        system.add('Bus', idx=idx, name='Bus ' + str(idx), Vn=baseKV,
                   v0=vmag, a0=vang,
                   vmax=vmax, vmin=vmin,
                   area=area, zone=zone)
        if pd != 0 or qd != 0:
            system.add('PQ', bus=idx, name='PQ ' + str(idx), Vn=baseKV, p0=pd, q0=qd)
        if gs or bs:
            system.add('Shunt', bus=idx, name='Shunt ' + str(idx), Vn=baseKV, g=gs, b=bs)

    gen_idx = 0
    for data in mpc['gen']:
        # bus  pg  qg  qmax  qmin  vg  mbase  status  pmax  pmin  pc1  pc2
        #  0   1   2    3     4     5    6      7       8    9    10    11
        # qc1min  qc1max  qc2min  qc2max  ramp_agc  ramp_10  ramp_30  ramp_q
        #  12      13       14      15      16        17       18      19
        # apf
        #  20

        bus_idx = int(data[0])
        gen_idx += 1
        vg = data[5]
        status = int(data[7])
        mbase = base_mva
        pg = data[1] / mbase
        qg = data[2] / mbase
        qmax = data[3] / mbase
        qmin = data[4] / mbase
        pmax = data[8] / mbase
        pmin = data[9] / mbase

        uid = system.Bus.idx2uid(bus_idx)
        vn = system.Bus.Vn.v[uid]
        a0 = system.Bus.a0.v[uid]

        if bus_idx in sw:
            system.add('Slack', idx=gen_idx, bus=bus_idx, busr=bus_idx,
                       name='Slack ' + str(bus_idx),
                       u=status,
                       Vn=vn, v0=vg, p0=pg, q0=qg, a0=a0,
                       pmax=pmax, pmin=pmin,
                       qmax=qmax, qmin=qmin)
        else:
            system.add('PV', idx=gen_idx, bus=bus_idx, busr=bus_idx,
                       name='PV ' + str(bus_idx),
                       u=status,
                       Vn=vn, v0=vg, p0=pg, q0=qg,
                       pmax=pmax, pmin=pmin,
                       qmax=qmax, qmin=qmin)

    for data in mpc['branch']:
        # fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle
        #  0     1      2   3   4     5       6       7       8      9
        # status	angmin	angmax	Pf	Qf	Pt	Qt
        #   10       11       12    13  14  15  16
        fbus = data[0]
        tbus = data[1]
        r = data[2]
        x = data[3]
        b = data[4]
        status = int(data[10])

        if data[8] == 0.0:  # not a transformer
            tf = False
            ratio = 1
            angle = 0
        elif data[8] == 1.0 and data[9] == 0.0:  # not a transformer
            tf = False
            ratio = 1
            angle = 0
        else:
            tf = True
            ratio = data[8]
            angle = data[9] * deg2rad

        vf = system.Bus.Vn.v[system.Bus.idx2uid(fbus)]
        vt = system.Bus.Vn.v[system.Bus.idx2uid(tbus)]
        system.add('Line', u=status, name=f'Line {fbus:.0f}-{tbus:.0f}',
                   Vn1=vf, Vn2=vt,
                   bus1=fbus, bus2=tbus,
                   r=r, x=x, b=b,
                   trans=tf, tap=ratio, phi=angle)

    if len(mpc['bus_name']) == len(system.Bus.name.v):
        system.Bus.name.v[:] = mpc['bus_name']

    return ret
