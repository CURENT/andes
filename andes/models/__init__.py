from .jit import JIT  # NOQA
from collections import OrderedDict  # NOQA


# Notes:
# - `timer`s are moved to the beginning for initialization.
#   Connectivity statuses should be restored before initializing the rest.

non_jit = OrderedDict([
    ('timer', ['Toggler', 'Fault']),
    ('bus', ['Bus']),
    ('pq', ['PQ']),
    ('pv', ['PV', 'Slack']),
    ('shunt', ['Shunt']),
    ('line', ['Line']),
    ('area', ['Area', 'ACE']),
    ('synchronous', ['GENCLS', 'GENROU']),
    ('governor', ['TG2', 'TGOV1', 'TGOV1DB', 'IEEEG1']),
    ('exciter', ['EXDC2', 'IEEEX1', 'ESDC2A', 'EXST1', 'ESST3A', 'SEXS']),
    ('pss', ['IEEEST', 'ST2CUT']),
    ('motor', ['Motor3', 'Motor5']),
    ('measurement', ['BusFreq', 'BusROCOF', 'PMU']),
    ('coi', ['COI', ]),
    ('dcbase', ['Node', 'Ground', 'R', 'L', 'C', 'RCp', 'RCs', 'RLs', 'RLCs', 'RLCp']),
    ('vsc', ['VSCShunt']),
    ('renewable', ['REGCA1', 'REECA1', 'REPCA1', 'WTDTA1', 'WTDS', 'WTARA1', 'WTPTA1', 'WTTQA1', 'PVD1']),
    ('experimental', ['PI2', 'TestDB1', 'TestPI', 'TestLagAWFreeze']),
])
