.. _cases:

***********************
Test Cases and Parsers
***********************

Directory
=========

ANDES comes with several test cases in the ``andes/cases/`` folder.
Currently, the Kundur's 2-area system, IEEE 14-bus system,
NPCC 140-bus system, and the WECC 179-bus system has been verified
against DSATools TSAT.

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

Performance
```````````
The numerical library used for sparse matrix factorization is KLU.
In addition, Jacobians are updated in place ``kvxopt.spmatrix.ipadd``.
Computations are performed on WSL2 Ubunbu 20.04 with AMD Ryzen 9 5950X,
64 GB 3200 MHz DDR4, running ANDES 1.5.3, KVXOPT 1.2.7.1, NumPy 1.20.3,
and numba 0.54.1. NumPy and KVXOPT use OpenBLAS 0.3.18.
Numba is enabled, and the generated code are precompiled.
Network connectivity checking is turned off.
Time to read numba cache (~0.3s) is not counted.

The computation time may vary depending on operating system and hardware.
All the cases are original in MATPOWER 7.0.
Cases not listed below will not solve with ANDES 1.5.3.

+----------------------+------------+-----------------+----------------+
|      File Name       | Converged? | # of Iterations | ANDES Time [s] |
+======================+============+=================+================+
|  case1354pegase.m    | 1          | 4               | 0.034          |
+----------------------+------------+-----------------+----------------+
|  case13659pegase.m   | 1          | 5               | 0.276          |
+----------------------+------------+-----------------+----------------+
|  case14.m            | 1          | 2               | 0.009          |
+----------------------+------------+-----------------+----------------+
|  case145.m           | 1          | 3               | 0.014          |
+----------------------+------------+-----------------+----------------+
|  case15nbr.m         | 1          | 17              | 0.024          |
+----------------------+------------+-----------------+----------------+
|  case17me.m          | 1          | 3               | 0.010          |
+----------------------+------------+-----------------+----------------+
|  case18.m            | 1          | 3               | 0.011          |
+----------------------+------------+-----------------+----------------+
|  case1888rte.m       | 1          | 2               | 0.025          |
+----------------------+------------+-----------------+----------------+
|  case18nbr.m         | 1          | 18              | 0.026          |
+----------------------+------------+-----------------+----------------+
|  case1951rte.m       | 1          | 3               | 0.031          |
+----------------------+------------+-----------------+----------------+
|  case2383wp.m        | 1          | 6               | 0.059          |
+----------------------+------------+-----------------+----------------+
|  case24_ieee_rts.m   | 1          | 4               | 0.012          |
+----------------------+------------+-----------------+----------------+
|  case2736sp.m        | 1          | 4               | 0.053          |
+----------------------+------------+-----------------+----------------+
|  case2737sop.m       | 1          | 5               | 0.060          |
+----------------------+------------+-----------------+----------------+
|  case2746wop.m       | 1          | 4               | 0.053          |
+----------------------+------------+-----------------+----------------+
|  case2746wp.m        | 1          | 4               | 0.054          |
+----------------------+------------+-----------------+----------------+
|  case2848rte.m       | 1          | 3               | 0.043          |
+----------------------+------------+-----------------+----------------+
|  case2868rte.m       | 1          | 4               | 0.056          |
+----------------------+------------+-----------------+----------------+
|  case2869pegase.m    | 1          | 6               | 0.084          |
+----------------------+------------+-----------------+----------------+
|  case30.m            | 1          | 3               | 0.010          |
+----------------------+------------+-----------------+----------------+
|  case300.m           | 1          | 5               | 0.019          |
+----------------------+------------+-----------------+----------------+
|  case30Q.m           | 1          | 3               | 0.009          |
+----------------------+------------+-----------------+----------------+
|  case30pwl.m         | 1          | 3               | 0.010          |
+----------------------+------------+-----------------+----------------+
|  case39.m            | 1          | 1               | 0.008          |
+----------------------+------------+-----------------+----------------+
|  case4_dist.m        | 1          | 3               | 0.010          |
+----------------------+------------+-----------------+----------------+
|  case4gs.m           | 1          | 3               | 0.011          |
+----------------------+------------+-----------------+----------------+
|  case5.m             | 1          | 3               | 0.011          |
+----------------------+------------+-----------------+----------------+
|  case57.m            | 1          | 3               | 0.010          |
+----------------------+------------+-----------------+----------------+
|  case60nordic.m      | 1          | 1               | 0.008          |
+----------------------+------------+-----------------+----------------+
|  case6468rte.m       | 1          | 6               | 0.144          |
+----------------------+------------+-----------------+----------------+
|  case6470rte.m       | 1          | 4               | 0.111          |
+----------------------+------------+-----------------+----------------+
|  case6495rte.m       | 1          | 5               | 0.130          |
+----------------------+------------+-----------------+----------------+
|  case6515rte.m       | 1          | 4               | 0.116          |
+----------------------+------------+-----------------+----------------+
|  case6ww.m           | 1          | 3               | 0.010          |
+----------------------+------------+-----------------+----------------+
|  case8387pegase.m    | 1          | 3               | 0.143          |
+----------------------+------------+-----------------+----------------+
|  case89pegase.m      | 1          | 5               | 0.015          |
+----------------------+------------+-----------------+----------------+
|  case9.m             | 1          | 3               | 0.011          |
+----------------------+------------+-----------------+----------------+
|  case9241pegase.m    | 1          | 6               | 0.243          |
+----------------------+------------+-----------------+----------------+
|  case9Q.m            | 1          | 3               | 0.011          |
+----------------------+------------+-----------------+----------------+
|  case9target.m       | 1          | 4               | 0.010          |
+----------------------+------------+-----------------+----------------+
|  case_ACTIVSg10k.m   | 1          | 4               | 0.157          |
+----------------------+------------+-----------------+----------------+
|  case_ACTIVSg200.m   | 1          | 2               | 0.010          |
+----------------------+------------+-----------------+----------------+
|  case_ACTIVSg2000.m  | 1          | 3               | 0.042          |
+----------------------+------------+-----------------+----------------+
|  case_ACTIVSg25k.m   | 1          | 7               | 0.549          |
+----------------------+------------+-----------------+----------------+
|  case_ACTIVSg500.m   | 1          | 3               | 0.015          |
+----------------------+------------+-----------------+----------------+
|  case_ACTIVSg70k.m   | 1          | 5               | 1.398          |
+----------------------+------------+-----------------+----------------+
|  case_RTS_GMLC.m     | 1          | 3               | 0.013          |
+----------------------+------------+-----------------+----------------+
|  case_SyntheticUSA.m | 1          | 5               | 1.727          |
+----------------------+------------+-----------------+----------------+
|  case_ieee30.m       | 1          | 2               | 0.008          |
+----------------------+------------+-----------------+----------------+


PSS/E Dyr Parser
================
ANDES supporting parsing PSS/E dynamic files in the format of ``.dyr``.
Support new dynamic models can be added by editing the input and output
conversion definition file in ``andes/io/psse-dyr.yaml``,
which is in the standard YAML format.
To add support for a new dynamic model, it is recommended to start with
an existing model of similar functionality.

Consider a ``GENCLS`` entry in a dyr file. The entry looks like ::

      1 'GENCLS' 1    13.0000  0.000000  /

where the fields are in the order of bus index, model name,
generator index on the bus, inertia (H) and damping coefficient (D).

The input-output conversion definition for GENCLS is as follows ::

    GENCLS:
        destination: GENCLS
        inputs:
            - BUS
            - ID
            - H
            - D
        find:
            gen:
                StaticGen:
                    bus: BUS
                    subidx: ID
        get:
            u:
                StaticGen:
                    src: u
                    idx: gen
            Sn:
                StaticGen:
                    src: Sn
                    idx: gen
            Vn:
                Bus:
                    src: Vn
                    idx: BUS
            ra:
                StaticGen:
                    src: ra
                    idx: gen
            xs:
                StaticGen:
                    src: xs
                    idx: gen
        outputs:
            u: u
            bus: BUS
            gen: gen
            Sn: Sn
            Vn: Vn
            D: D
            M: "GENCLS.H; lambda x: 2 * x"
            ra: ra
            xd1: xs

It begins with a base-level definition of the model name to be parsed from the
dyr file, namely, ``GENCLS``. Five directives can be defined for each model:
``destination``, ``inputs``, ``outputs``, ``find`` and ``get``.
Note that ``find`` and ``get`` are optional, but the other three are mandatory.

- ``destination`` is ANDES model to which the original PSS/E model will be
  converted. In this case, the ANDES model have the same name ``GENCLS``.
- ``inputs`` is a list of the parameter names for the PSS/E data.
  Arbitrary names can be used, but it is recommended to use the same notation
  following the PSS/E manual.
- ``outputs`` is a dictionary where the keys are the ANDES model parameter and
  the values are the input parameter or lambda functions that processes the inputs
  (see notes below).
- ``find`` is a dictionary with the keys being the temporary parameter name to store
  the ``idx`` of
  external devices and the values being the criteria to locate the devices.
  In the example above, ``GENCLS`` will try to find the ``idx`` of ``StaticGen``
  with ``bus == BUS`` and the ``subidx == ID``, where ``BUS`` and ``ID`` are from
  the dyr file.
- ``get`` is a dictionary with each key being a temporary parameter name for storing
  an external parameter and each value being the criteria to find the external parameter.
  In the example above, a temporary parameter ``u`` is the ``u`` parameter of ``StaticGen``
  whose ``idx == gen``. Note that ``gen`` is the ``idx`` of ``StaticGen`` retrieved
  in the above ``find`` section.

For the ``inputs`` section, one will need to skip the model name
because for any model, the second field is always the model name.
That is why for ``GENCLS`` below, we only list four input parameters. ::

    1 'GENCLS' 1    13.0000  0.000000  /

For the ``outputs`` section, the order can be arbitrary, but it is recommended
to follow the input order as much as possible for maintainability.
In particular, the right-hand-side of the outputs can be either an input parameter name
or an anonymous expression that processes the input parameters.
For the example of GENCLS, since ANDES internally uses the parameter of ``M = 2H``,
the input ``H`` needs to be multiplied by 2.
It is done by the following ::

    M: "GENCLS.H; lambda x: 2 * x"

where the left-hand-side is the output parameter name (destination ANDES model parameter name),
and the right-hand-side is arguments and the lambda function separated by semi-colon, all in a
pair of double quotation marks.
Multiple arguments are accepted and should be separated by comma.
Arguments can come from the same model or another model.
In the case of the same model, the model name can be neglected, namely, by writing
``M: "H; lambda x: 2 * x"``.
