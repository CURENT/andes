""" Simple MATPOWER format parser
"""
import re
from ..consts import deg2rad

import logging
logger = logging.getLogger(__name__)


def testlines(fid):
    return True  # hard coded


def read(file, system):
    """Read a MATPOWER data file into mpc and build andes device elements"""
    func = re.compile('function\\s')
    mva = re.compile('\\s*mpc.baseMVA\\s*=\\s*')
    bus = re.compile('\\s*mpc.bus\\s*=\\s*\\[')
    gen = re.compile('\\s*mpc.gen\\s*=\\s*\\[')
    branch = re.compile('\\s*mpc.branch\\s*=\\s*\\[')
    area = re.compile('\\s*mpc.areas\\s*=\\s*\\[')
    gencost = re.compile('\\s*mpc.gencost\\s*=\\s*\\[')
    bus_name = re.compile('\\s*mpc.bus_name\\s*=\\s*\\{')
    end = re.compile('\\s*\\];?')
    hasdigit = re.compile('.*\\d+\\s*]?;?')
    comment = re.compile('\\s*%.*')

    retval = True
    field = None
    info = True

    basemva = 100
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
        elif comment.search(line):  # for comment lines
            if info:
                logger.info(line[1:72])
                info = False
            else:
                continue
        elif mva.search(line):
            basemva = float(line.split('=')[1])

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
                line = re.sub('\[', '', line)
            elif line.find('{') >= 0:
                line = re.sub('\{', '', line)

            if line.find('\'') >= 0:  # bus_name
                line = line.split(';')
                data = [i.strip('\'').strip() for i in line]
                mpc['bus_name'].extend(data)
            else:
                if not hasdigit.search(line):
                    continue
                line = line.split(';')
                for item in line:
                    data = [float(val) for val in item.split()]
                    mpc[field].append(data)

    fid.close()

    # add model elements to system
    sw = []

    system.mva = basemva

    for data in mpc['bus']:
        # idx  ty   pd   qd  gs  bs  area  vmag  vang  baseKV  zone  vmax  vmin
        # 0    1    2   3   4   5    6      7     8     9      10    11    12
        idx = int(data[0])
        ty = data[1]
        if ty == 3:
            sw.append(idx)
        pd = data[2] / basemva
        qd = data[3] / basemva
        gs = data[4] / basemva
        bs = data[5] / basemva
        area = data[6]
        vmag = data[7]
        vang = data[8] * deg2rad
        baseKV = data[9]
        zone = data[10]
        vmax = data[11]
        vmin = data[12]

        try:
            system.Bus.elem_add(
                idx=idx,
                name='Bus ' + str(idx),
                Vn=baseKV,
                voltage=vmag,
                angle=vang,
                vmax=vmax,
                vmin=vmin,
                area=area,
                region=zone)
            if pd or qd:
                system.PQ.elem_add(
                    bus=idx, name='PQ ' + str(idx), Vn=baseKV, p=pd, q=qd)
            if gs or bs:
                system.Shunt.elem_add(
                    bus=idx, name='Shunt ' + str(idx), Vn=baseKV, g=gs, b=bs)
        except KeyError:
            logger.error('Error adding <Bus> to powersystem object.')
            retval = False

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
        mbase = basemva
        pg = data[1] / mbase
        qg = data[2] / mbase
        qmax = data[3] / mbase
        qmin = data[4] / mbase
        pmax = data[8] / mbase
        pmin = data[9] / mbase

        try:
            vn = system.Bus.Vn[system.Bus.uid[bus_idx]]
            if bus_idx in sw:
                system.SW.elem_add(
                    idx=gen_idx,
                    bus=bus_idx,
                    busr=bus_idx,
                    name='SW ' + str(bus_idx),
                    u=status,
                    Vn=vn,
                    v0=vg,
                    pg=pg,
                    qg=qg,
                    pmax=pmax,
                    pmin=pmin,
                    qmax=qmax,
                    qmin=qmin,
                    a0=0.0)
            else:
                system.PV.elem_add(
                    idx=gen_idx,
                    bus=bus_idx,
                    busr=bus_idx,
                    name='PV ' + str(bus_idx),
                    u=status,
                    Vn=vn,
                    v0=vg,
                    pg=pg,
                    qg=qg,
                    pmax=pmax,
                    pmin=pmin,
                    qmax=qmax,
                    qmin=qmin)
        except KeyError:
            logger.error(
                'Error adding <SW> or <PV> to powersystem object.')
            retval = False

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
            angle = data[9]
        try:
            vf = system.Bus.Vn[system.Bus.uid[fbus]]
            vt = system.Bus.Vn[system.Bus.uid[tbus]]
            system.Line.elem_add(
                Vn=vf,
                Vn2=vt,
                bus1=fbus,
                bus2=tbus,
                r=r,
                x=x,
                b=b,
                u=status,
                trasf=tf,
                tap=ratio,
                phi=angle)
        except KeyError:
            logger.error('Error adding <Line> to powersystem object.')
            retval = False
    if len(mpc['bus_name']) == len(system.Bus.name):
        system.Bus.name[:] = mpc['bus_name']

    return retval
