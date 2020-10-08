"""
PSSE RAW parser from em_psse

https://github.com/anderson-optimization/em-psse

License Pending
"""
import os
from io import StringIO
from andes.shared import pd, yaml

import logging

logger = logging.getLogger(__name__)

dirname = os.path.dirname(__file__)
with open('{}/psse-modes.yaml'.format(dirname), 'r') as in_file:
    modes = yaml.full_load(in_file)


def get_signals(line_num, line, current_mode):
    # print(line_num,line)
    signals = []
    for m in modes:
        for s in m['signal']:
            if 'text' in s:
                if s['text'] in line:
                    signals.append((s, m))
            if 'line' in s:
                if s['line'] == line_num:
                    signals.append((s, m))
    return signals


def read_transformer(lines, records):
    iter_lines = iter(lines)
    # records=[]
    headers = [r['header'] for r in records]
    line_holder = {
        2: [
            [headers[0]],
            [headers[1]],
            [headers[2]],
            [headers[3]]
        ],
        3: [
            [headers[0]],
            [headers[1]],
            [headers[2]],
            [headers[3]],
            [headers[4]]
        ]
    }
    for line in iter_lines:
        rows = [line]
        rows.append(next(iter_lines))
        rows.append(next(iter_lines))
        rows.append(next(iter_lines))
        windings = None
        try:
            flag = int(line.split(',')[2])
            if flag != 0:
                raise ValueError('Not a 2 phase winding')
            windings = 2
        except Exception:
            windings = 3
            rows.append(next(iter_lines))

        for r in range(len(rows)):
            line_holder[windings][r].append(rows[r])

    dfs = {
        2: [],
        3: []
    }
    for winding in line_holder:
        rc = 0
        for record in line_holder[winding]:
            rc += 1
            text = StringIO(''.join(record))
            dfs[winding].append(pd.read_table(text, sep=',', error_bad_lines=False))
            logger.debug("{} {} {}".format(winding, rc, len(record)))
            logger.debug("{}".format(record[0]))
    return dfs


def read_twodc(lines, records):
    iter_lines = iter(lines)
    # records=[]
    headers = [r['header'] for r in records]
    line_holder = [
        [headers[0]],
        [headers[1]],
        [headers[2]]
    ]
    for line in iter_lines:
        rows = [line]
        rows.append(next(iter_lines))
        rows.append(next(iter_lines))

        for r in range(len(rows)):
            line_holder[r].append(rows[r])

    dfs = []
    rc = 0
    for record in line_holder:
        rc += 1
        text = StringIO(''.join(record))
        dfs.append(pd.read_table(text, sep=','))
        logger.debug("{} {}".format(rc, len(record)))
        logger.debug("{}".format(record[0]))

    return dfs


def parse_raw(in_file_name):
    """
    This function will parse a RAW file and return a PyPSA model
    """

    # Initialize output
    output = {}
    for item in modes:
        key = item['key']
        output[key] = {
            "name": item['name'],
            "key": key,
            "lines": []
        }
        # Set up column structure
        if 'columns' in item:
            output[key]['columns'] = item['columns']
            header = ",".join([c['name'] for c in item['columns']]).replace(' ', '') + '\n'
            output[key]['header'] = header
            output[key]['lines'].append(header)
        if 'records' in item:
            output[key]['records'] = item['records']
            for record in output[key]['records']:
                header = ",".join([c['name'] for c in record['columns']]).replace(' ', '') + '\n'
                record['header'] = header
        # Get parsing info
        output[key]['parse'] = item.get('parse', {})

    # Initialize header mode
    current_mode = {
        'key': 'header'
    }

    # Read file and store in container
    with open(in_file_name, 'r') as in_file:
        line = in_file.readline()
        line_num = 0
        while line:
            line_num += 1

            START = None
            STOP = None

            # Process signals
            signals = get_signals(line_num, line, current_mode)
            for signal, mode in signals:
                logger.debug("Signal: {} {} {}".format(line_num, signal['command'], mode['key']))
                if signal['command'] == 'start':
                    START = mode
                if signal['command'] == 'stop':
                    STOP = mode

            if current_mode and START and not STOP:
                raise ValueError('Current mode was never stopped')

            # Store lines
            if current_mode and (not STOP or 'keep_tail' in STOP):
                key = current_mode['key']
                output[key]['lines'].append(line)
            # print key

            if STOP and current_mode and current_mode['key'] != STOP['key']:
                raise ValueError('Attempting to stop a different mode', current_mode['key'], STOP['key'])
            elif START:
                current_mode = START
            elif STOP:
                current_mode = None

            line = in_file.readline()

    logger.debug("Captured Lines")
    for i in output:
        logger.debug('Item: {}, length: {}'.format(i, len(output[i]['lines'])))

    for i in output:
        if 'lines' not in output[i]:
            logger.debug('no lines {}'.format(i))
            continue
        lines = output[i]['lines']
        if len(lines) == 1:
            logger.debug('only header {}'.format(i))
            continue

        if 'read_table' in output[i]['parse']:
            text = StringIO(''.join(lines))
            output[i]['df'] = pd.read_table(text, sep=',')

        if 'read_transformer' in output[i]['parse']:
            output[i]['dfs'] = read_transformer(lines, output[i]['records'])
            df2 = pd.concat(output[i]['dfs'][2], axis=1, sort=False)
            df3 = pd.concat(output[i]['dfs'][3], axis=1, sort=False)
            output[i]['df'] = df3.append(df2, sort=False)

        if 'read_twodc' in output[i]['parse']:
            output[i]['dfs'] = read_twodc(lines, output[i]['records'])
            output[i]['df'] = pd.concat(output[i]['dfs'], axis=1, sort=False)

        logger.debug('{} {} {}'.format(i, len(lines), 'df' in output[i]))
        if 'df' in output[i]:
            logger.info('Parsed {} {}'.format(len(output[i]['df']), i))
            if len(lines) > 0:
                logger.debug("{}".format(lines[0]))

    return output
