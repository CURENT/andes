from .jit import JIT


__all__ = ['base',
           'bus',
           ]

non_jits = {'bus': {'Bus': 'Bus'},
            'pq': {'PQ': 'PQ'},
            'pv': {'PV': 'PV',
                   'Slack': 'SW'},
            'line': {'Line': 'Line'},
            'shunt': {'Shunt': 'Shunt'},
            'zone': {'Zone': 'Zone',
                     'Area': 'Area',
                     'Region': 'Region',
                     },
            'dcbase': {'Node': 'Node',
                       'RLine': 'RLine',
                       'Ground': 'Ground',
                       },
            'synchronous': {'Syn2': 'Syn2',
                            'Syn6a': 'Syn6a',
                            },
            'fault': {'Fault': 'Fault'},
            'governor': {'TG2': 'TG2',
                         'TG1': 'TG1'},
            'measurement':{'BusFreq': 'BusFreq',
                           'PMU': 'PMU',
                           },
            'avr': {'AVR3': 'AVR3',
                    }

           }

jits = {'vsc': {'VSC': 'VSC',
                'VSC1': 'VSC1',
                'VSC2': 'VSC2',
                'VSC3': 'VSC3'
                },
        }

