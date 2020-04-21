.. ANDES documentation master file, created by
   sphinx-quickstart on Thu Jun 21 11:11:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####
Home
####

.. raw:: html

   <embed>
   <h1 style="letter-spacing: 0.4em; font-size: 2.5em !important;
   margin-bottom: 0; padding-bottom: 0"> ANDES </h1>

   <p style="color: #00746F; font-variant: small-caps; font-weight: bold;
   margin-bottom: 2em">
   Python Software for Symbolic Power System Modeling and Numerical Analysis</p>
   </embed>


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
Use the following resources in addition the tutorial to get involved.

- Jupyter Notebook Examples:
  `Checkout on GitHub <https://github.com/cuihantao/andes/tree/master/examples>`_.
- Try ANDES in Jupyter Notebook:
  `with Binder <https://mybinder.org/v2/gh/cuihantao/andes/master>`_.
- Online Documentation:
  `https://andes.readthedocs.io <https://andes.readthedocs.io>`_.
- PDF Documentation:
  `download <https://andes.readthedocs.io/_/downloads/en/stable/pdf/>`_.
- Report issues:
  `GitHub issues page <https://github.com/cuihantao/andes/issues>`_.
- Learn version control:
  `command-line tutorial <https://git-scm.com/docs/gittutorial>`_, and
  `using GitHub Desktop <https://help.github.com/en/desktop/getting-started-with-github-desktop>`_.

Acknowledgement
***************

This work was supported in part by the Engineering Research Center Program of
the National Science Foundation and the Department of Energy under NSF Award
Number EEC-1041877 and the CURENT Industry Partnership Program.

**ANDES is made open source as part of the CURENT Large Scale Testbed project.**

This project was originally inspired by the book `Power System Modelling and Scripting <https://www.springer
.com/gp/book/9783642136689>`_
by `Prof. Federico Milano <http://faraday1.ucd.ie>`_.

Authors and Contributors
************************

ANDES is developed by `Hantao Cui <https://cuihantao.github.io>`_.

The following contributors are sincerely acknowledged:

- Christopher Lacknr: for contributing the EAGC model.
- Qiwei Zhang: for contributing a solar PV model.
- `Yichen Zhang <https://sites.google.com/view/yichenzhangee/home>`_: for feedback on documentation.


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
