#!usr/bin/env python3

from setuptools import setup


setup(name='andes',
      version='0.3.0',
      description='ANDES - A Python Package for Power System Research',
      author='Hantao Cui',
      author_email='hcui7@utk.edu',
      url='http://curent.utk.edu',
      install_requires=[
          'cvxopt',
          'numpy',
          'texttable',
          'matplotlib',
      ],
      packages=[
          'andes',
          'andes.filters',
          'andes.formats',
          'andes.models',
          'andes.routines',
          'andes.settings',
          'andes.utils',
          'andes.variables'
      ],
      classifiers=[
          "Development Status :: 4 - Beta",
          "Topic :: Scientific/Engineering",
          "License :: OSI Approved :: Apache Software License",
          "Environment :: Console",
      ],
      entry_points={
            'console_scripts': [
                  'andes = andes:main',
                  'andesplot = andes:plot'
            ]
      },
      )
