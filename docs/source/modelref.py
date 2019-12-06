"""
This file is used to generate reStructuredText tables for Group and Model references
"""

import andes
import dill
dill.settings['recurse'] = True

ss = andes.system.SystemNew()
ss.prepare()

out = ''
out += '.. _modelref:\n\n'
out += '********************************************************************************\n'
out += 'Model References\n'
out += '********************************************************************************\n'
out += '\n'

for group in ss.groups.values():
    out += group.doc_all(export='rest')

with open('modelref.rst', 'w') as f:
    f.write(out)
