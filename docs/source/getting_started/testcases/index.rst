
.. _test-cases:

============
Test Cases
============

.. toctree::
    :hidden:

    folder
    benchmark

ANDES ships with with test cases in the ``andes/cases`` folder.
The cases can be found in the `online repository`_.

.. _`online repository`: https://github.com/cuihantao/andes/tree/master/andes/cases

Summary
=======

Below is a summary of the folders and the corresponding test cases. Some folders
contain a README file. When viewing on GitHub, the README is automatically
rendered in the folder for quick reference.

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

..
    todo: verification

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