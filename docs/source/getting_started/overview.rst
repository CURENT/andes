.. _package-overview:

================
Package Overview
================

ANDES is an open-source Python package for power system modeling, computation,
analysis, and control. It establishes a unique **hybrid symbolic-numeric
framework** for modeling differential algebraic equations (DAEs) for numerical
analysis. The main features of ANDES include

  - a unique hybrid symbolic-numeric approach to modeling and simulation that
    enables descriptive DAE modeling and automatic numerical code generation
  - a rich library of transfer functions and discontinuous components (including
    limiters, dead-bands, and saturation) available for prototyping models,
    which can be readily instantiated as multiple devices for system analysis
  - industry-grade second-generation renewable models (solar PV, type 3 and type
    4 wind), distributed PV, and energy storage model
  - comes with the Newton method for power flow calculation, the implicit
    trapezoidal method for time-domain simulation, and full eigenvalue
    calculation
  - rigorously verified models with commercial software. ANDES obtains identical
    time-domain simulation results for IEEE 14-bus and NPCC system with GENROU
    and multiple controller models. See the verification link for details.
  - developed with performance in mind. While written in Python, ANDES comes
    with a performance package and can finish a 20-second transient simulation
    of a 2000-bus system in a few seconds on a typical desktop computer
  - out-of-the-box PSS/E raw and dyr file support for available models. Once a
    model is developed, inputs from a dyr file can be readily supported
  - always up-to-date equation documentation of implemented models

ANDES is currently under active development. To get involved,

* Follow the tutorial at
  `https://andes.readthedocs.io <https://andes.readthedocs.io/en/stable/tutorial.html>`_
* Checkout the Notebook examples in the
  `examples folder <https://github.com/curent/andes/tree/master/examples>`_
* Try ANDES in Jupyter Notebook
  `with Binder <https://mybinder.org/v2/gh/curent/andes/master>`_
* Download the PDF manual at
  `download <https://andes.readthedocs.io/_/downloads/en/stable/pdf/>`_
* Report issues in the
  `GitHub issues page <https://github.com/curent/andes/issues>`_
* Learn version control with
  `the command-line git <https://git-scm.com/docs/gittutorial>`_ or
  `GitHub Desktop <https://help.github.com/en/desktop/getting-started-with-github-desktop>`_
* If you are looking to develop models, read the
  `Modeling Cookbook <https://andes.readthedocs.io/en/stable/modeling.html>`_

This work was supported in part by the Engineering Research Center Program of
the National Science Foundation and the Department of Energy under NSF Award
Number EEC-1041877 and the CURENT_ Industry Partnership Program. ANDES is made
open source as part of the CURENT Large Scale Testbed project.

ANDES is developed and actively maintained by `Hantao Cui <https://cui.eecps.com>`_.
See the GitHub repository for a full list of contributors.

.. _CURENT: https://curent.utk.edu
