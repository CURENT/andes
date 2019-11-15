from .jit import JIT  # NOQA
from collections import OrderedDict  # NOQA


non_jits = OrderedDict([
    ('bus', ['Bus']),
    ('pq', ['PQ']),
    ('pv', ['PV', 'Slack']),
    ('shunt', ['Shunt']),
    ('line', ['Line']),
])
