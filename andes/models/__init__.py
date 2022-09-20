"""
The package for DAE models in ANDES.
"""

# Notes:
# - `timer`s are moved to the beginning for initialization.
#   Connectivity statuses should be restored before initializing the rest.

# `file_classes` records the `.py` files under `andes/models` and the classes in
# each file. Models will be initialized in the order given below.

file_classes = list([
    ('info', ['Summary']),
    ('misc', ['Output']),
    ('timer', ['Toggle', 'Fault', 'Alter']),
    ('timeseries', ['TimeSeries']),
    ('bus', ['Bus']),
    ('static', ['PQ', 'PV', 'Slack']),
    ('shunt', ['Shunt', "ShuntTD", 'ShuntSw']),
    ('interface', ['Fortescue']),
    ('line', ['Line', 'Jumper']),
    ('area', ['Area', 'ACE', 'ACEc']),
    ('dynload', ['ZIP', 'FLoad']),
    ('synchronous', ['GENCLS', 'GENROU', 'PLBVFU1']),
    ('governor', ['TG2', 'TGOV1', 'TGOV1DB', 'TGOV1N', 'TGOV1NDB',
                  'IEEEG1', 'IEESGO', 'GAST', 'HYGOV', 'HYGOVDB', 'HYGOV4']),
    ('vcomp', ['IEEEVC']),
    ('exciter', ['EXDC2', 'IEEEX1', 'ESDC1A', 'ESDC2A', 'EXST1', 'ESST3A', 'SEXS',
                 'IEEET1', 'EXAC1', 'EXAC2', 'EXAC4', 'ESST4B', 'AC8B', 'IEEET3',
                 'ESAC1A', 'ESST1A', 'ESAC5A']),
    ('pss', ['IEEEST', 'ST2CUT']),
    ('motor', ['Motor3', 'Motor5']),
    ('measurement', ['BusFreq', 'BusROCOF', 'PMU', 'PLL1', 'PLL2']),
    ('dc', ['Node', 'Ground', 'R', 'L', 'C', 'RCp', 'RCs', 'RLs', 'RLCs', 'RLCp']),
    ('acdc', ['VSCShunt']),
    ('renewable', ['REGCA1', 'REGCP1']),
    ('renewable', ['REECA1', 'REECA1E', 'REECA1G']),
    ('renewable', ['REPCA1']),
    ('renewable', ['WTDTA1', 'WTDS', 'WTARA1', 'WTPTA1', 'WTTQA1', 'WTARV1',
                   'REGCV1', 'REGCV2', 'REGF1', 'REGF2', 'REGF3']),
    ('distributed', ['PVD1', 'ESD1', 'EV1', 'EV2', 'DGPRCT1', 'DGPRCTExt']),
    ('coi', ['COI']),
    # ('experimental', ['PI2', 'TestDB1', 'TestPI', 'TestLagAWFreeze', 'FixedGen']),
])


model_aliases = {"REGCVSG": "REGCV1", "REGCVSG2": "REGCV2", "Toggler": "Toggle"}
