.. ANDES documentation master file, created by
   sphinx-quickstart on Thu Jun 21 11:11:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <embed>
   <h1 style="letter-spacing: 0.4em; font-size: 2.5em !important;
   margin-bottom: 0; padding-bottom: 0"> ANDES </h1>

   <p style="color: #00746F; font-variant: small-caps; font-weight: bold;
   margin-bottom: 2em">
   Python Software for Symbolic Power System Modeling and Numerical Analysis</p>
   </embed>

****
Home
****

ANDES is a Python-based free software package for power system simulation, control and analysis.
It establishes a unique **hybrid symbolic-numeric framework** for modeling differential algebraic
equations (DAEs) for numerical analysis. Main features of ANDES include

..
   ANDES offers a symbolic library for discrete components
   and transfer functions that can be easily imported to DAE models.
   ANDES supports power flow calculation, time domain simulation and eigenvalue analysis for transmission
   networks.

- a unique hybrid symbolic-numeric approach to modeling and simulation that enables descriptive DAE modeling and
  automatic numerical code generation
- a rich library of transfer functions and discontinuous components (including limiters, dead-bands, and
  saturation) available for prototyping models, which can be readily instantiated as multiple devices for
  system analysis
- comes with the Newton method for power flow calculation, the implicit trapezoidal method for time-domain
  simulation, and full eigenvalue calculation
- strictly verified models with commercial software. ANDES obtains identical time-domain simulation results for
  IEEE 14-bus and NPCC system with GENROU and multiple controller models. See the verification link for details.
- developed with performance in mind. While written in Python, ANDES comes with a performance package and can
  finish a 20-second transient simulation of a 2000-bus system in a few seconds on a typical desktop computer
- out-of-the-box PSS/E raw and dyr file support for available models. Once a model is developed, inputs from a
  dyr file can be readily supported
- an always up-to-date equation documentation of implemented models

ANDES is currently under active development. To get involved,

- Follow the tutorial at
  `https://andes.readthedocs.io <https://andes.readthedocs.io/en/stable/tutorial.html>`_
- Checkout the Notebook examples in the
  `examples folder <https://github.com/cuihantao/andes/tree/master/examples>`_
- Try ANDES in Jupyter Notebook
  `with Binder <https://mybinder.org/v2/gh/cuihantao/andes/master>`_
- Download the PDF manual at
  `download <https://andes.readthedocs.io/_/downloads/en/stable/pdf/>`_
- Report issues in the
  `GitHub issues page <https://github.com/cuihantao/andes/issues>`_
- Learn version control with
  `the command-line git <https://git-scm.com/docs/gittutorial>`_ or
  `GitHub Desktop <https://help.github.com/en/desktop/getting-started-with-github-desktop>`_
- If you are looking to develop models, read the
  `Modeling Cookbook <https://andes.readthedocs.io/en/stable/modeling.html>`_

This work was supported in part by the Engineering Research Center Program of
the National Science Foundation and the Department of Energy under NSF Award
Number EEC-1041877 and the `CURENT <https://curent.utk.edu>`_ Industry Partnership Program.
**ANDES is made open source as part of the CURENT Large Scale Testbed project.**

ANDES is developed and actively maintained by `Hantao Cui <https://cui.eecps.com>`_.
See the GitHub repository for a full list of contributors.

.. toctree::
   :caption: ANDES Manual
   :maxdepth: 3
   :hidden:

   install.rst
   tutorial.rst
   modeling.rst
   cases.rst
   modelref.rst
   configref.rst
   faq.rst
   troubleshooting.rst
   misc.rst
   release-notes.rst
   copyright.rst


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
