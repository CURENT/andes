
.. _test-cases:

============
Test Cases
============

ANDES ships with with test cases in the ``andes/cases`` folder.
The cases can be found in the `online repository`_.

.. _`online repository`: https://github.com/cuihantao/andes/tree/master/andes/cases

Summary
=======

Below is a summary of the folders and the corresponding test cases. Some folders
contain a README file with notes. When viewing the case folder on GitHub, one
can conveniently read the README file below the file listing.

- ``smib``: single machine infinite bus (SMIB) system [Sauer]_.
- ``5bus``: a small PJM 5-bus test case for power flow study [PJM5]_.
- ``GBnetwork``: a 2,000-bus system for the Great Britain network [GB]_. Dynamic
  data is randomly generated.
- ``EI``: the CURENT Eastern Interconnection network [EI]_.
- ``ieee14`` and ``ieee39``: the IEEE 14-bus and 39-bus test cases [IEEE]_.
- ``kundur``: a modified Kundur's two area system from [RLGC]_. The modified
  system is different in the number of buses and lines from the textbook.
- ``matpower``: a subset of test cases from [MATPOWER]_.
- ``nordic44``: Nordpool 44-bus test case [Nordic]_. Not all dynamic models are
  supported.
- ``npcc``: the 140-bus Northeast Power Coordinating Council (NPCC) test case
  originated from Power System Toolbox [PST]_.
- ``wecc``: the 179-bus Western Electric Coordinating Council (WECC) test case
  [WECC]_.
- ``wscc``: the 9-bus WSCC (succeeded by WECC) power flow data converted from
  [PSAT]_.

.. Note::

    Different systems exhibit different dynamics, thus the appropriate systems
    should be used to study power system stability. For example:

    - The Kundur's two-area system has under-damped modes and two coherent
      groups of generators. It is suitable for oscillation study and transient
      stability studies.
    - The WECC system is known for the inter-area oscillation.
    - The IEEE 14-bus system and the 140-bus NPCC system is are frequently used
      for frequency control studies. So is the Eastern Interconnection system.

Currently, the Kundur's 2-area system, IEEE 14-bus system,
NPCC 140-bus system, and the WECC 179-bus system has been verified
against DSATools TSAT.

Example data
============

When developing models, we manually create cases with example data to verify the
models. The Kundur's system and the IEEE 14-bus system are used as the bases for
adding specific models. One can find many cases in the folder
``andes/cases/kundur``. The case file names typically indicate the specific
model added to the file. These example cases with specific models are useful
when one needs to find example parameters for the model. For example:

- ``kundur_ieeeg1`` indicates the use of ``IEEEG1`` model in a Kundur's sytem
- ``ieee14_solar.xlsx`` contains the solar PV models (REGCA1, REECA1, and
  REPCA1)
- ``ieee14_plbvfu1.xlsx`` is the case for ``PLBVFU1`` (playback of voltage and
  frequency). The case provides an example of specifying ``plbvf.xlsx``
- ``ieee14_timeseries.xlsx`` is an example for specifying timeseries for load
  data, which is provided in ``pqts.xlsx``

MATPOWER cases
==============

MATPOWER cases has been tested in ANDES for power flow calculation.
All following cases are calculated with the provided initial values
using the full Newton-Raphson iterative approach.

Benchmark
---------

See :ref:`matpower-benchmark` for the benchmark of MATPOWER cases.

Synthetic systems
-----------------

The 70K and the USA synthetic systems have difficulties to converge using the
provided initial values. One can solve the case in MATPOWER and save it as a new
case for ANDES. For example, the SyntheticUSA case can be converted in MATLAB
with

.. code:: matlab

    mpc = runpf(case_SyntheticUSA)
    savecase('USA.m', mpc)

And then solve it with ANDES from command line:

.. code:: bash

    andes run USA.m

The output should look like

.. code:: console

    -> Power flow calculation
    Sparse solver: KLU
    Solution method: NR method
    Power flow initialized.
    0: |F(x)| = 140.5782767
    1: |F(x)| = 29.61673314
    2: |F(x)| = 4.161452394
    3: |F(x)| = 0.2337870537
    4: |F(x)| = 0.001149488448
    5: |F(x)| = 3.646516689e-08
    Converged in 6 iterations in 1.6082 seconds.

How to contribute
=================

We welcome the contribution of test cases! You can make a pull request to
contribute new test cases. Please follow the structure in the ``cases`` folder
and provide an example Jupyter notebook (see ``examples/demonstration``) to
showcase the results of your system.

.. [Sauer] P. W. Sauer, M. A. Pai, and J. H. Chow, Power system dynamics
        and stability: with synchrophasor measurement and power system toolbox,
        Second edition. Hoboken, NJ, USA: IEEE Press, Wiley, 2017.
.. [PJM5] F. Li and R. Bo, "Small test systems for power system economic
        studies," IEEE PES General Meeting, 2010, pp. 1-4, doi:
        10.1109/PES.2010.5589973.
.. [GB] The University of Edinburgh, "Power Systems Test Case Archive",
        https://www.maths.ed.ac.uk/optenergy/NetworkData/fullGB
.. [EI]  D. Osipov and M. Arrieta-Paternina, "Reduced Eastern Interconnection
        System Model", [Online]. Available:
        https://curent.utk.edu/2016SiteVisit/EI_LTB_Report.pdf.
.. [IEEE] University of Washington, "Power Systems Test Case Archive", [Online].
        Available: https://labs.ece.uw.edu/pstca/
.. [RLGC] Qiuhua Huang, "RLGC repository", [Online]. Available:
        https://github.com/RLGC-Project/RLGC
.. [MATPOWER] R. D. Zimmerman, "MATPOWER", [Online]. Available:
        https://matpower.org/
.. [Nordic] ALSETLab, "Nordpool test system", [Online]. Available:
        https://github.com/ALSETLab/Nordic44-Nordpool/tree/master/nordic44/models
.. [PST] Power System Toolbox, [Online]. Available:
        https://sites.ecse.rpi.edu/~chowj/PSTMan.pdf
.. [WECC] K. Sun, "Test Cases Library of Power System Sustained Oscillations".
       Available: http://web.eecs.utk.edu/~kaisun/Oscillation/basecase.html
.. [PSAT] F. Milano, "Power System Analysis Toolbox", [Online]. Available:
        http://faraday1.ucd.ie/psat.html