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

- Symbolic DAE modeling and automated code generation for numerical simulation.
- Numerical DAE modeling for cases when symbolic implementations are difficult.
- Rapid modeling library with transfer functions and discrete components.
- Automatic sequential and iterative initialization (experimental) for dynamic models.
- Newton-Raphson power flow, trapezoidal method-based time domain simulation, and full eigenvalue analysis.
- Full equation documentation of supported DAE models.

ANDES is currently under active development.

- Jupyter Notebook Examples:
  `Checkout on GitHub <https://github.com/cuihantao/andes/tree/master/examples>`_.
- Online Documentation:
  `andes.readthedocs.io <https://andes.readthedocs.io>`_.
- PDF Documentation:
  `download from andes.readthedocs.io <https://andes.readthedocs.io/_/downloads/en/stable/pdf/>`_.
- Report issues:
  `GitHub issues page <https://github.com/cuihantao/andes/issues>`_.

.. toctree::
   :caption: ANDES Manual
   :maxdepth: 3
   :hidden:

   copyright.rst
   install.rst
   tutorial.rst
   modeling.rst
   cases.rst
   modelref.rst
   configref.rst
   misc.rst
   release-notes.rst


.. toctree::
   :hidden:
   :caption: API References
   :maxdepth: 3

   andes.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
