from .jit import JIT  # NOQA

__all__ = ['jits', 'non_jits', 'all_models', 'all_models_list']

order = [
    'Bus', 'Node', 'Ground', 'line', 'pq', 'pv', 'zone', 'shunt',
    'measurement', 'synchronous', 'governor', 'avr', 'pss', 'windturbine',
    'wind', 'BArea', 'eAGC', 'AGC', 'R', 'L', 'C', 'RLs', 'RCs', 'RCp', 'RLCp',
    'RLCs', 'DCgen', 'vsc', 'Recorder'
]

non_jits = {
    'bus': {
        'Bus': 'Bus'
    },
    'pq': {
        'PQ': 'PQ'
    },
    'pv': {
        'PV': 'PV',
        'Slack': 'SW'
    },
    'line': {
        'Line': 'Line'
    },
    'shunt': {
        'Shunt': 'Shunt'
    },
    'zone': {
        'Zone': 'Zone',
        'Area': 'Area',
        'Region': 'Region',
    },
    'agc': {
        'BArea': 'BArea',
        'AGC': 'AGC',
        'eAGC': 'eAGC',
    },
    'dcbase': {
        'Node': 'Node',
        'Ground': 'Ground',
        'R': 'R',
        'L': 'L',
        'C': 'C',
        'RLs': 'RLs',
        'RCs': 'RCs',
        'RCp': 'RCp',
        'RLCp': 'RLCp',
        'RLCs': 'RLCs',
        'DCgen': 'DCgen',
    },
    'synchronous': {
        'Syn2': 'Syn2',
        'Syn6a': 'Syn6a',
    },
    'governor': {
        'TG2': 'TG2',
        'TG1': 'TG1'
    },
    'measurement': {
        'BusFreq': 'BusFreq',
        'PMU': 'PMU',
    },
    'avr': {
        'AVR3': 'AVR3',
        'AVR2': 'AVR2',
        'AVR1': 'AVR1',
    },
    'pss': {
        'PSS1': 'PSS1',
        'PSS2': 'PSS2',
    },
    'fault': {
        'Fault': 'Fault'
    },
    'breaker': {
        'Breaker': 'Breaker'
    },
    'event': {
        'GenTrip': 'GenTrip',
        'LoadShed': 'LoadShed',
    },
    'recorder': {
        'Recorder': 'Recorder'
    },
}

jits = {
    'vsc': {
        'VSC': 'VSC',
        'VSC1': 'VSC1',
        'VSC1_IE': 'VSC1_IE',
        'VSC1_IE2': 'VSC1_IE2',
        'VSC2A': 'VSC2A',
        'VSC2B': 'VSC2B'
    },
    'windturbine': {
        'WTG3': 'WTG3',
        'WTG4DC': 'WTG4DC',
    },
    'wind': {
        'Weibull': 'Weibull',
        'ConstWind': 'ConstWind'
    },
}

all_models = dict(jits)
all_models.update(non_jits)

all_models_list = []

for key in sorted(all_models.keys()):
    val = all_models[key]
    all_models_list.extend(sorted(list(val.values())))
