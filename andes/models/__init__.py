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
                     }

           }

jits = {}

