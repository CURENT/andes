.. _verification:

============
Verification
============

This section presents the verification of the models and algorithms implemented
in ANDES by comparing the time-domain simulation results with commercial tools.

ANDES produces identical results for the IEEE 14-bus and the NPCC systems with
several models. For the CURENT WECC system, ANDES, TSAT and PSS/E produce
slightly different results. In fact, results from different tools can hardly
match for large systems with a variety of dynamic models.

.. toctree::
   :maxdepth: 2

   ../_examples/verification/andes-ieee14-verification.ipynb
   ../_examples/verification/andes-npcc-verification.ipynb
   ../_examples/verification/andes-wecc-verification.ipynb
