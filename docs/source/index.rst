.. ANDES documentation master file, created by
   sphinx-quickstart on Thu Jun 21 11:11:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

############
Introduction
############

ANDES is a Python-based free software package for power system simulation, control and analysis.
It establishes a unique **hybrid symbolic-numeric framework** for modeling differential algebraic
equations (DAEs) for numerical analysis. Main features of ANDES include

..
   ANDES offers a symbolic library for discrete components
   and transfer functions that can be easily imported to DAE models.
   ANDES supports power flow calculation, time domain simulation and eigenvalue analysis for transmission
   networks.

- Symbolic DAE modeling and automated code generation for numerical simulation
- Numerical DAE modeling for cases when symbolic implementations are difficult
- Rapid modeling with block library with common transfer functions.
- Discrete component library such as hard limiter, dead band, and anti-windup limiter.
- Newton-Raphson and Newton-Krylov based power flow calculation.
- Trapezoidal method for time domain simulation of semi-explicit DAE.
- Complete documentation of supported DAE models.

ANDES is currently under active development. Please report issues on the
`GitHub Issues page <https://github.com/cuihantao/andes/issues>`_.

.. toctree::
   :caption: ANDES Manual
   :maxdepth: 2

   copyright.rst
   release-notes.rst
   install.rst
   tutorial.rst
   formats.rst
   modeling.rst
   cases.rst
   cheatsheet.rst
   modelref.rst

.. toctree::
   :caption: API References
   :maxdepth: 4

   andes.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
