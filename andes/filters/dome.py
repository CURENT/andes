"""
Parser for DOME RAW format 0.1
From Book "Power System Modeling and Scripting" by Dr. Federico Milano
"""
import re


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
    """Read a dm format file and add to system"""
    retval = True
    fid = open(file, 'r')
    sep = re.compile(r'\s*,\s*')
    comment = re.compile(r'^#\s*')
    equal = re.compile(r'\s*=\s*')
    math = re.compile(r'[*/+-]')
    double = re.compile(r'[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

    # parse data
    while 1:
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
            if not newline:
                break
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
            print('Parsing include file <%s>', data[0])
            newfid = open(data[0], 'rt')
            read(newfid, system, header=False)  # recursive call
            newfid.close()
            print('Parsing of include file <%s> completed.', data[0])
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
                    value = map(lambda x: eval(x), array)
                else:
                    value = map(lambda x: float(x), array)
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
            system.__dict__[device].add(idx=index, name=namex, **kwargs)
        except KeyError:
            system.Log.error('Error adding device {:s} to powersystem object.'.format(device))
            system.Log.debug('  Check if you have new jit models added to models.__init__.py')

    fid.close()
    return retval
