"""
Parser for DOME RAW format 0.1
From Book "Power System Modeling and Scripting" by Dr. Federico Milano
"""
import re
import os
from math import ceil
import logging
logger = logging.getLogger(__name__)


def testlines(fid):
    return True  # hard coded yet


def alter(data, system):
    """Alter data in dm format devices"""
    device = data[0]
    action = data[1]
    if data[2] == '*':
        data[2] = '.*'
    regex = re.compile(data[2])
    prop = data[3]
    value = float(data[4])
    if action == 'MUL':
        for item in range(system.__dict__[device].n):
            if regex.search(system.__dict__[device].name[item]):
                system.__dict__[device].__dict__[prop][item] *= value
    elif action == 'REP':
        for item in range(system.__dict__[device].n):
            if regex.search(system.__dict__[device].name[item]):
                system.__dict__[device].__dict__[prop][item] = value
    elif action == 'DIV':
        if not value:
            return
        for item in range(system.__dict__[device].n):
            if regex.search(system.__dict__[device].name[item]):
                system.__dict__[device].__dict__[prop][item] /= value
    elif action == 'SUM':
        for item in range(system.__dict__[device].n):
            if regex.search(system.__dict__[device].name[item]):
                system.__dict__[device].__dict__[prop][item] += value
    elif action == 'SUB':
        for item in range(system.__dict__[device].n):
            if regex.search(system.__dict__[device].name[item]):
                system.__dict__[device].__dict__[prop][item] -= value
    elif action == 'POW':
        for item in range(system.__dict__[device].n):
            if regex.search(system.__dict__[device].name[item]):
                system.__dict__[device].__dict__[prop][item] **= value
    else:
        print('ALTER action <%s> is not defined', action)


def read(file, system, header=True):
    """Read a dm format file and elem_add to system"""
    retval = True
    fid = open(file, 'r')
    sep = re.compile(r'\s*,\s*')
    comment = re.compile(r'^#\s*')
    equal = re.compile(r'\s*=\s*')
    math = re.compile(r'[*/+-]')
    double = re.compile(r'[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

    # parse data
    while True:
        line = fid.readline()
        if not line:
            break
        line = line.replace('\n', '')
        line = line.strip()
        if not line:
            continue
        if comment.search(line):
            continue
        # span multiple line
        while line.endswith(',') or line.endswith(';'):
            newline = fid.readline()
            line = line.replace('\n', '')
            newline = newline.strip()
            if not newline:
                continue
            if comment.search(newline):
                continue
            line += ' ' + newline
        data = sep.split(line)
        device = data.pop(0)
        device = device.strip()
        if device == 'ALTER':
            alter(data, system)
            continue
        if device == 'INCLUDE':
            logger.debug('Parsing include file <{}>'.format(data[0]))
            newpath = data[0]
            if not os.path.isfile(newpath):
                newpath = os.path.join(system.files.path, data[0])
                if not os.path.isfile(newpath):
                    logger.warning(
                        'Unable to locate file in {}'.format(newpath))
                    retval = False
                    continue
            read(newpath, system, header=False)  # recursive call
            logger.debug('Parsing of include file <{}> completed.'.format(
                data[0]))
            continue
        kwargs = {}
        for item in data:
            pair = equal.split(item)
            key = pair[0].strip()
            value = pair[1].strip()
            if value.startswith('"'):
                value = value[1:-1]
            elif value.startswith('['):
                array = value[1:-1].split(';')
                if math.search(value):  # execute simple operations
                    value = list(map(lambda x: eval(x), array))
                else:
                    value = list(map(lambda x: float(x), array))
            elif double.search(value):
                if math.search(value):  # execute simple operations
                    value = eval(value)
                else:
                    value = float(value)
            elif value == 'True':
                value = True
            elif value == 'False':
                value = False
            else:
                value = int(value)
            kwargs[key] = value
        index = kwargs.pop('idx', None)
        namex = kwargs.pop('name', None)
        try:
            system.__dict__[device].elem_add(idx=index, name=namex, **kwargs)
        except KeyError:
            logger.error(
                'Error adding device {:s} to powersystem object.'.format(
                    device))
            logger.debug(
                'Make sure you have added the jit models in __init__.py'
            )

    fid.close()
    return retval


def write(file, system):
    """
    Write data in system to a dm file
    """

    # TODO: Check for bugs!!!

    out = list()
    out.append('# DOME format version 1.0')
    ppl = 7  # parameter per line
    retval = True
    dev_list = sorted(system.devman.devices)
    for dev in dev_list:
        model = system.__dict__[dev]
        if not model.n:
            continue

        out.append('')
        header = dev + ', '
        space = ' ' * (len(dev) + 2)
        keys = list(model._data.keys())
        keys.extend(['name', 'idx'])
        keys = sorted(keys)

        # remove non-existent keys
        for key in keys:
            if key not in model.__dict__.keys():
                keys.pop(key)

        nline = int(ceil(len(keys) / ppl))
        nelement = model.n
        vals = [''] * len(keys)

        # for each element, read values
        for elem in range(nelement):
            for idx, key in enumerate(keys):
                if model._flags['sysbase'] and key in model._store.keys():
                    val = model._store[key][elem]
                else:
                    val = model.__dict__[key][elem]

                if type(val) == float:
                    val = round(val, 5)
                elif type(val) == str:
                    val = '"{}"'.format(val)
                elif type(val) == map:
                    val = list(val)
                    val = '; '.join(str(i) for i in val)
                    val = '[{}]'.format(val)
                elif val is None:
                    val = 0
                vals[idx] = val

            pair = []
            for key, val in zip(keys, vals):
                pair.append('{} = {}'.format(key, val))

            for line in range(nline):
                string = ', '.join(pair[ppl * line:ppl * (line + 1)])
                if line == 0:  # append header or space
                    string = header + string
                else:
                    string = space + string
                if not line == nline - 1:  # add comma except for last line
                    string += ','

                out.append(string)

    fid = open(file, 'w')
    for line in out:
        fid.write(line + '\n')

    fid.close()
    return retval
