"""
The package for DAE models in ANDES.
"""

from collections import OrderedDict

# Notes:
# - `timer`s are moved to the beginning for initialization.
#   Connectivity statuses should be restored before initializing the rest.


# `file_classes` records the `.py` files under `andes/models` and the classes in each file.
# Models will be initialized in the order given below.

file_classes = OrderedDict([
    ('info', ['Summary']),
    ('timer', ['Toggler', 'Fault', 'Alter']),
    ('bus', ['Bus']),
    ('pq', ['PQ']),
    ('pv', ['PV', 'Slack']),
    ('shunt', ['Shunt', 'ShuntSw']),
    ('line', ['Line', 'Jumper']),
    ('area', ['Area', 'ACE', 'ACEc']),
    ('dynload', ['ZIP', 'FLoad']),
    ('synchronous', ['GENCLS', 'GENROU']),
    ('governor', ['TG2', 'TGOV1', 'TGOV1N', 'TGOV1DB', 'IEEEG1', 'IEESGO']),
    ('exciter', ['EXDC2', 'IEEEX1', 'ESDC2A', 'EXST1', 'ESST3A', 'SEXS', 'IEEET1', 'EXAC1', 'EXAC4', 'ESST4B']),
    ('pss', ['IEEEST', 'ST2CUT']),
    ('motor', ['Motor3', 'Motor5']),
    ('measurement', ['BusFreq', 'BusROCOF', 'PMU']),
    ('dcbase', ['Node', 'Ground', 'R', 'L', 'C', 'RCp', 'RCs', 'RLs', 'RLCs', 'RLCp']),
    ('vsc', ['VSCShunt']),
    ('renewable', ['REGCA1', 'REECA1', 'REECA1E',
                   'REPCA1', 'WTDTA1', 'WTDS', 'WTARA1', 'WTPTA1', 'WTTQA1', 'WTARV1',
                   'REGCVSG', 'REGCVSG2']),
    ('distributed', ['PVD1', 'ESD1', 'EV1', 'EV2', 'PLK']),
    ('experimental', ['PI2', 'TestDB1', 'TestPI', 'TestLagAWFreeze', 'FixedGen']),
    ('coi', ['COI']),
])
