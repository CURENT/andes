from .jit import JIT  # NOQA
from collections import OrderedDict  # NOQA


non_jit = OrderedDict([
    ('bus', ['Bus']),
    ('dcbase', ['Node', 'Ground', 'R', 'L', 'C', 'RCp', 'RCs', 'RLs', 'RLCs', 'RLCp']),
    ('pq', ['PQ']),
    ('pv', ['PV', 'Slack']),
    ('shunt', ['Shunt']),
    ('line', ['Line']),
    ('area', ['Area']),
    ('synchronous', ['GENCLS', 'GENROU']),
    ('governor', ['TG2', 'TGOV1']),
    ('exciter', ['EXDC2']),
    ('timer', ['Toggler']),
    ('experimental', ['PI2']),
])
