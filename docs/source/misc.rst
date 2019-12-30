.. _misc:

**********************
Miscellaneous
**********************

Per Unit System
==============================

The bases for AC system are

- :math:`S_b^{ac}`: three-phase power in MVA. By default, :math:`S_b^{ac}=100 MVA` (in ``System.config.mva``).

- :math:`V_b^{ac}`: phase-to-phase voltage in kV.

- :math:`I_b^{ac}`: current base :math:`I_b^{ac} = \frac{S_b^{ac}} {\sqrt{3} V_b^{ac}}`

The bases for DC system are

- :math:`S_b^{dc}`: power in MVA. It is assumed to be the same as :math:`S_b^{ac}`.

- :math:`V_b^{dc}`: voltage in kV.

Profiling Import
========================================
To speed up the command-line program, import profiling is used to breakdown the program loading time.

With tool ``profimp``, ``andes`` can be profiled with ``profimp "import andes" --html > andes_import.htm``. The
report can be viewed in any web browser.
