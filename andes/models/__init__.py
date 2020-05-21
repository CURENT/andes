from .jit import JIT  # NOQA
from collections import OrderedDict  # NOQA


non_jit = OrderedDict([
    ('bus', ['Bus']),
    ('pq', ['PQ']),
    ('pv', ['PV', 'Slack']),
    ('shunt', ['Shunt']),
    ('line', ['Line']),
    ('area', ['Area']),
    ('synchronous', ['GENCLS', 'GENROU']),
    ('governor', ['TG2', 'TGOV1', 'IEEEG1']),
    ('exciter', ['EXDC2', 'IEEEX1', 'ESDC2A', 'EXST1', 'ESST3A', 'SEXS']),
    ('pss', ['IEEEST', 'ST2CUT']),
    ('measurement', ['BusFreq', 'BusROCOF']),
    ('coi', ['COI', ]),
    ('dcbase', ['Node', 'Ground', 'R', 'L', 'C', 'RCp', 'RCs', 'RLs', 'RLCs', 'RLCp']),
    ('vsc', ['VSCShunt']),
    ('timer', ['Toggler', 'Fault']),
    ('experimental', ['PI2']),
])
