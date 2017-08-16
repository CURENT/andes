#!usr/bin/env python3

from setuptools import setup


setup(name='andes',
      version='0.2',
      description='ANDES - A Python Package for Power System Research',
      author='Hantao Cui',
      author_email='hcui7@utk.edu',
      url='http://curent.utk.edu',
      install_requires=[
          'cvxopt',
          'numpy',
          'texttable',
          'blist',
          'matplotlib',
          'progressbar2',
          'python_utils',
          'pyzmq',
      ],
      packages=[
          'andes',
          'andes.filters',
          'andes.formats',
          'andes.models',
          'andes.routines',
          'andes.settings',
          'andes.tests',
          'andes.utils',
          'andes.variables'
      ],
      classifiers=[
            "Development Status :: 3 - Alpha",
            "Topic :: System :: Power",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            "Environment :: Console",

      ],
      entry_points={
            'console_scripts': [
                  'andes = andes:main',
                  'andesplot = andes:plot'
            ]
      },

      )
