
Directory
=========


The test case library will continue to build as more models get implemented.

A tree view of the test directory is as follows. ::

    cases/
    ├── 5bus/
    │   └── pjm5bus.xlsx
    ├── GBnetwork/
    │   ├── GBnetwork.m
    │   ├── GBnetwork.xlsx
    │   └── README.md
    ├── ieee14/
    │   ├── ieee14.dyr
    │   └── ieee14.raw
    ├── kundur/
    │   ├── kundur.raw
    │   ├── kundur_aw.xlsx
    │   ├── kundur_coi.xlsx
    │   ├── kundur_coi_empty.xlsx
    │   ├── kundur_esdc2a.xlsx
    │   ├── kundur_esst3a.xlsx
    │   ├── kundur_exdc2_zero_tb.xlsx
    │   ├── kundur_exst1.xlsx
    │   ├── kundur_freq.xlsx
    │   ├── kundur_full.dyr
    │   ├── kundur_full.xlsx
    │   ├── kundur_gentrip.xlsx
    │   ├── kundur_ieeeg1.xlsx
    │   ├── kundur_ieeest.xlsx
    │   ├── kundur_sexs.xlsx
    │   └── kundur_st2cut.xlsx
    ├── matpower/
    │   ├── case118.m
    │   ├── case14.m
    │   ├── case300.m
    │   └── case5.m
    ├── nordic44/
    │   ├── N44_BC.dyr
    │   ├── N44_BC.raw
    │   └── README.md
    ├── npcc/
    │   ├── npcc.raw
    │   └── npcc_full.dyr
    ├── wecc/
    │   ├── wecc.raw
    │   ├── wecc.xlsx
    │   ├── wecc_full.dyr
    │   ├── wecc_gencls.dyr
    └── wscc9/
        ├── wscc9.raw
        └── wscc9.xlsx

MATPOWER Cases
==============================

MATPOWER cases has been tested in ANDES for power flow calculation.
All following cases are calculated with the provided initial values
using the full Newton-Raphson iterative approach.

Note:

The 70K and the USA synthetic systems have difficulties to converge
using the provided initial values. One can solve the case in MATPOWER
and save it as a new case for ANDES.
For example, the SyntheticUSA case can be converted in MATLAB with

.. code:: matlab

    mpc = runpf(case_SyntheticUSA)
    savecase('USA.m', mpc)

And then solve it with ANDES from command line:

.. code:: bash

    andes run USA.m

The output should look like ::

    -> Power flow calculation
    Sparse solver: KLU
    Solution method: NR method
    Power flow initialized.
    0: \|F(x)\| = 140.5782767
    1: \|F(x)\| = 29.61673314
    2: \|F(x)\| = 4.161452394
    3: \|F(x)\| = 0.2337870537
    4: \|F(x)\| = 0.001149488448
    5: \|F(x)\| = 3.646516689e-08
    Converged in 6 iterations in 1.6082 seconds.
