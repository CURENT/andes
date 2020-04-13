"""
This file is used to generate reStructuredText tables for Group and Model references.
"""

import andes
ss = andes.prepare()

out = ''
out += '.. _modelref:\n\n'
out += '****************\n'
out += 'Model References\n'
out += '****************\n'
out += '\n'

for group in ss.groups.values():
    out += group.doc_all(export='rest')

with open('modelref.rst', 'w') as f:
    f.write(out)


out = ''
out += '.. _configref:\n\n'
out += '*****************\n'
out += 'Config References\n'
out += '*****************\n'
out += '\n'

out += ss.config.doc(export='rest', target=True, symbol=False)

for r in ss.routines.values():
    out += r.config.doc(export='rest', target=True, symbol=False)

with open('configref.rst', 'w') as f:
    f.write(out)
