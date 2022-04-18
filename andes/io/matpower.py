""" Simple MATPOWER format parser
"""
import logging
import re

import andes.io
import numpy as np
from andes.shared import deg2rad, rad2deg

logger = logging.getLogger(__name__)


def testlines(infile):
    """
    Test if this file is in the MATPOWER format.

    NOT YET IMPLEMENTED.
    """

    return True  # hard coded


def read(system, file):
    """
    Read a MATPOWER data file into mpc, and build andes device elements.
    """

    mpc = m2mpc(file)
    return mpc2system(mpc, system)


def m2mpc(infile: str) -> dict:
    """
    Parse MATPOWER file and return a dictionary with the data.

    Supported fields include

    - baseMVA
    - bus
    - bus_names
    - gen
    - branch
    - gencost (parsed but not used)
    - areas (parsed but not used)

    Parameters
    ----------
    infile : str
        Path to the MATPOWER file.

    Returns
    -------
    dict
        mpc struct names : numpy arrays
    """

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

    field = None
    info = True

    mpc = {
        'version': 2,  # not in use
        'baseMVA': 100,
        'bus': [],
        'gen': [],
        'branch': [],
        'area': [],
        'gencost': [],
        'bus_name': [],
    }

    input_list = andes.io.read_file_like(infile)

    for line in input_list:
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
            mpc["baseMVA"] = float(line.split('=')[1])

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
                        logger.error('Error parsing "%s"', infile)
                        raise e
                    mpc[field].append(data)

    # convert mpc to np array
    mpc_array = dict()
    for key, val in mpc.items():
        if isinstance(val, (float, int)):
            mpc_array[key] = val
        elif isinstance(val, list):
            if len(val) == 0:
                continue
            mpc_array[key] = np.array(val)
        else:
            raise NotImplementedError("Unkonwn type for mpc, ", type(val))

    return mpc_array


def mpc2system(mpc: dict, system) -> bool:
    """
    Load an mpc dict into an empty ANDES system.

    Parameters
    ----------
    system : andes.system.System
        Empty system to load the data into.
    mpc : dict
        mpc struct names : numpy arrays

    Returns
    -------
    bool
        True if successful, False otherwise.
    """

    # list of buses with slack gen
    sw = []

    system.config.mva = base_mva = mpc['baseMVA']

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
        fbus = int(data[0])
        tbus = int(data[1])
        r = data[2]
        x = data[3]
        b = data[4]
        rate_a = data[5]
        rate_b = data[6]
        rate_c = data[7]

        status = int(data[10])

        if (data[8] == 0.0) or (data[8] == 1.0 and data[9] == 0.0):
            # not a transformer
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
                   trans=tf, tap=ratio, phi=angle,
                   rate_a=rate_a, rate_b=rate_b, rate_c=rate_c)

    if ('bus_name' in mpc) and (len(mpc['bus_name']) == len(system.Bus.name.v)):
        system.Bus.name.v[:] = mpc['bus_name']

    return True


def _get_bus_id_caller(bus):
    """
    Helper function to get the bus id. If any of bus ``idx`` is a string, use
    ``uid`` + 1. Otherwise, use ``idx``.

    Parameters
    ----------
    bus : andes.models.bus.Bus
        Bus object

    Returns
    -------
    lambda function to that takes bus idx and returns bus id for matpower case

    """

    if np.array(bus.idx.v).dtype == np.object:
        return lambda x: bus.idx2uid(x) + 1
    else:
        return lambda x: x


def system2mpc(system) -> dict:
    """
    Convert data from an ANDES system to an mpc dict.

    In the ``gen`` section, slack generators preceeds PV generators.
    """

    mpc = dict(version='2',
               baseMVA=system.config.mva,
               bus=np.zeros((system.Bus.n, 13), dtype=np.float64),
               gen=np.zeros((system.PV.n + system.Slack.n, 21), dtype=np.float64),
               branch=np.zeros((system.Line.n, 17), dtype=np.float64),
               bus_name=np.zeros((system.Bus.n, ), dtype=object),
               )

    base_mva = system.config.mva

    to_busid = _get_bus_id_caller(system.Bus)

    # --- bus ---
    bus = mpc['bus']
    gen = mpc['gen']

    bus[:, 0] = to_busid(system.Bus.idx.v)
    bus[:, 1] = 1
    bus[:, 7] = system.Bus.v0.v
    bus[:, 8] = system.Bus.a0.v * rad2deg
    bus[:, 9] = system.Bus.Vn.v
    bus[:, 11] = system.Bus.vmax.v
    bus[:, 12] = system.Bus.vmin.v

    # area and zone not supported

    # --- PQ ---
    if system.PQ.n > 0:
        pq_pos = system.Bus.idx2uid(system.PQ.bus.v)
        bus[pq_pos, 2] = system.PQ.p0.v * base_mva
        bus[pq_pos, 3] = system.PQ.q0.v * base_mva

    # --- Shunt ---
    if system.Shunt.n > 0:
        shunt_pos = system.Bus.idx2uid(system.Shunt.bus.v)
        bus[shunt_pos, 4] = system.Shunt.g.v * base_mva
        bus[shunt_pos, 5] = system.Shunt.b.v * base_mva

    # --- PV ---
    if system.PV.n > 0:
        pv_pos = system.Bus.idx2uid(system.PV.bus.v)
        bus[pv_pos, 1] = 2
        gen[system.Slack.n:, 0] = to_busid(system.PV.bus.v)
        gen[system.Slack.n:, 1] = system.PV.p0.v * base_mva
        gen[system.Slack.n:, 2] = system.PV.q0.v * base_mva
        gen[system.Slack.n:, 3] = system.PV.qmax.v * base_mva
        gen[system.Slack.n:, 4] = system.PV.qmin.v * base_mva
        gen[system.Slack.n:, 5] = system.PV.v0.v
        gen[system.Slack.n:, 6] = base_mva
        gen[system.Slack.n:, 7] = system.PV.u.v
        gen[system.Slack.n:, 8] = system.PV.pmax.v * base_mva
        gen[system.Slack.n:, 9] = system.PV.pmin.v * base_mva

    # --- Slack ---
    if system.Slack.n > 0:
        slack_pos = system.Bus.idx2uid(system.Slack.bus.v)
        bus[slack_pos, 1] = 3
        bus[slack_pos, 8] = system.Slack.a0.v * rad2deg

        gen[:system.Slack.n, 0] = to_busid(system.Slack.bus.v)
        gen[:system.Slack.n, 1] = system.Slack.p0.v * base_mva
        gen[:system.Slack.n, 2] = system.Slack.q0.v * base_mva
        gen[:system.Slack.n, 3] = system.Slack.qmax.v * base_mva
        gen[:system.Slack.n, 4] = system.Slack.qmin.v * base_mva
        gen[:system.Slack.n, 5] = system.Slack.v0.v
        gen[:system.Slack.n, 6] = base_mva
        gen[:system.Slack.n, 7] = system.Slack.u.v
        gen[:system.Slack.n, 8] = system.Slack.pmax.v * base_mva
        gen[:system.Slack.n, 9] = system.Slack.pmin.v * base_mva

    if system.Line.n > 0:
        branch = mpc['branch']
        branch[:, 0] = to_busid(system.Line.bus1.v)
        branch[:, 1] = to_busid(system.Line.bus2.v)
        branch[:, 2] = system.Line.r.v
        branch[:, 3] = system.Line.x.v
        branch[:, 4] = system.Line.b.v
        branch[:, 5] = system.Line.rate_a.v
        branch[:, 6] = system.Line.rate_b.v
        branch[:, 7] = system.Line.rate_c.v
        branch[:, 8] = system.Line.tap.v
        branch[:, 9] = system.Line.phi.v * rad2deg
        branch[:, 10] = system.Line.u.v

    mpc['bus_name'] = np.array(system.Bus.name.v)

    return mpc
