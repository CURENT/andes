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

The numerical library used for sparse matrix factorization is KLU.
In addition, Jacobians are updated in place ``spmatrix.ipadd``.
Computations are performed on macOS 10.15.4 with i9-9980H, 16 GB
2400 MHz DDR4, running ANDES 0.9.1, CVXOPT 1.2.4 and NumPy 1.18.1.

The statistics of convergence, number of iterations, and solution time
(including equation evaluation, Jacobian, and factorization time) are
reported in the following table.
The computation time may vary depending on operating system and hardware.

+--------------------------+------------+-----------------+----------+
|        File Name         | Converged? | # of Iterations | Time [s] |
+==========================+============+=================+==========+
|  case30.m                | 1          | 3               | 0.012    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg500.m       | 1          | 3               | 0.019    |
+--------------------------+------------+-----------------+----------+
|  case13659pegase.m       | 1          | 5               | 0.531    |
+--------------------------+------------+-----------------+----------+
|  case9Q.m                | 1          | 3               | 0.011    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg200.m       | 1          | 2               | 0.013    |
+--------------------------+------------+-----------------+----------+
|  case24_ieee_rts.m       | 1          | 4               | 0.014    |
+--------------------------+------------+-----------------+----------+
|  case300.m               | 1          | 5               | 0.026    |
+--------------------------+------------+-----------------+----------+
|  case6495rte.m           | 1          | 5               | 0.204    |
+--------------------------+------------+-----------------+----------+
|  case39.m                | 1          | 1               | 0.009    |
+--------------------------+------------+-----------------+----------+
|  case18.m                | 1          | 4               | 0.013    |
+--------------------------+------------+-----------------+----------+
|  case_RTS_GMLC.m         | 1          | 3               | 0.014    |
+--------------------------+------------+-----------------+----------+
|  case1951rte.m           | 1          | 3               | 0.047    |
+--------------------------+------------+-----------------+----------+
|  case6ww.m               | 1          | 3               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case5.m                 | 1          | 3               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case69.m                | 1          | 3               | 0.014    |
+--------------------------+------------+-----------------+----------+
|  case6515rte.m           | 1          | 4               | 0.168    |
+--------------------------+------------+-----------------+----------+
|  case2383wp.m            | 1          | 6               | 0.084    |
+--------------------------+------------+-----------------+----------+
|  case30Q.m               | 1          | 3               | 0.011    |
+--------------------------+------------+-----------------+----------+
|  case2868rte.m           | 1          | 4               | 0.074    |
+--------------------------+------------+-----------------+----------+
|  case1354pegase.m        | 1          | 4               | 0.047    |
+--------------------------+------------+-----------------+----------+
|  case2848rte.m           | 1          | 3               | 0.063    |
+--------------------------+------------+-----------------+----------+
|  case4_dist.m            | 1          | 3               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case6470rte.m           | 1          | 4               | 0.175    |
+--------------------------+------------+-----------------+----------+
|  case2746wp.m            | 1          | 4               | 0.074    |
+--------------------------+------------+-----------------+----------+
|  case_SyntheticUSA.m     | 1          | 21              | 11.120   |
+--------------------------+------------+-----------------+----------+
|  case118.m               | 1          | 3               | 0.014    |
+--------------------------+------------+-----------------+----------+
|  case30pwl.m             | 1          | 3               | 0.021    |
+--------------------------+------------+-----------------+----------+
|  case57.m                | 1          | 3               | 0.017    |
+--------------------------+------------+-----------------+----------+
|  case89pegase.m          | 1          | 5               | 0.024    |
+--------------------------+------------+-----------------+----------+
|  case6468rte.m           | 1          | 6               | 0.232    |
+--------------------------+------------+-----------------+----------+
|  case2746wop.m           | 1          | 4               | 0.075    |
+--------------------------+------------+-----------------+----------+
|  case85.m                | 1          | 3               | 0.011    |
+--------------------------+------------+-----------------+----------+
|  case22.m                | 1          | 2               | 0.008    |
+--------------------------+------------+-----------------+----------+
|  case4gs.m               | 1          | 3               | 0.012    |
+--------------------------+------------+-----------------+----------+
|  case14.m                | 1          | 2               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg10k.m       | 1          | 4               | 0.251    |
+--------------------------+------------+-----------------+----------+
|  case2869pegase.m        | 1          | 6               | 0.136    |
+--------------------------+------------+-----------------+----------+
|  case_ieee30.m           | 1          | 2               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case2737sop.m           | 1          | 5               | 0.087    |
+--------------------------+------------+-----------------+----------+
|  case9target.m           | 1          | 5               | 0.013    |
+--------------------------+------------+-----------------+----------+
|  case1888rte.m           | 1          | 2               | 0.037    |
+--------------------------+------------+-----------------+----------+
|  case145.m               | 1          | 3               | 0.018    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg2000.m      | 1          | 3               | 0.059    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg70k.m       | 1          | 15              | 7.043    |
+--------------------------+------------+-----------------+----------+
|  case9241pegase.m        | 1          | 6               | 0.497    |
+--------------------------+------------+-----------------+----------+
|  case9.m                 | 1          | 3               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case141.m               | 1          | 3               | 0.012    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg25k.m       | 1          | 7               | 1.040    |
+--------------------------+------------+-----------------+----------+
|  case118.m               | 1          | 3               | 0.015    |
+--------------------------+------------+-----------------+----------+
|  case1354pegase.m        | 1          | 4               | 0.048    |
+--------------------------+------------+-----------------+----------+
|  case13659pegase.m       | 1          | 5               | 0.523    |
+--------------------------+------------+-----------------+----------+
|  case14.m                | 1          | 2               | 0.011    |
+--------------------------+------------+-----------------+----------+
|  case141.m               | 1          | 3               | 0.013    |
+--------------------------+------------+-----------------+----------+
|  case145.m               | 1          | 3               | 0.017    |
+--------------------------+------------+-----------------+----------+
|  case18.m                | 1          | 4               | 0.012    |
+--------------------------+------------+-----------------+----------+
|  case1888rte.m           | 1          | 2               | 0.037    |
+--------------------------+------------+-----------------+----------+
|  case1951rte.m           | 1          | 3               | 0.052    |
+--------------------------+------------+-----------------+----------+
|  case22.m                | 1          | 2               | 0.011    |
+--------------------------+------------+-----------------+----------+
|  case2383wp.m            | 1          | 6               | 0.086    |
+--------------------------+------------+-----------------+----------+
|  case24_ieee_rts.m       | 1          | 4               | 0.015    |
+--------------------------+------------+-----------------+----------+
|  case2736sp.m            | 1          | 4               | 0.074    |
+--------------------------+------------+-----------------+----------+
|  case2737sop.m           | 1          | 5               | 0.108    |
+--------------------------+------------+-----------------+----------+
|  case2746wop.m           | 1          | 4               | 0.093    |
+--------------------------+------------+-----------------+----------+
|  case2746wp.m            | 1          | 4               | 0.089    |
+--------------------------+------------+-----------------+----------+
|  case2848rte.m           | 1          | 3               | 0.065    |
+--------------------------+------------+-----------------+----------+
|  case2868rte.m           | 1          | 4               | 0.079    |
+--------------------------+------------+-----------------+----------+
|  case2869pegase.m        | 1          | 6               | 0.137    |
+--------------------------+------------+-----------------+----------+
|  case30.m                | 1          | 3               | 0.033    |
+--------------------------+------------+-----------------+----------+
|  case300.m               | 1          | 5               | 0.102    |
+--------------------------+------------+-----------------+----------+
|  case30Q.m               | 1          | 3               | 0.013    |
+--------------------------+------------+-----------------+----------+
|  case30pwl.m             | 1          | 3               | 0.013    |
+--------------------------+------------+-----------------+----------+
|  case39.m                | 1          | 1               | 0.008    |
+--------------------------+------------+-----------------+----------+
|  case4_dist.m            | 1          | 3               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case4gs.m               | 1          | 3               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case5.m                 | 1          | 3               | 0.011    |
+--------------------------+------------+-----------------+----------+
|  case57.m                | 1          | 3               | 0.015    |
+--------------------------+------------+-----------------+----------+
|  case6468rte.m           | 1          | 6               | 0.229    |
+--------------------------+------------+-----------------+----------+
|  case6470rte.m           | 1          | 4               | 0.170    |
+--------------------------+------------+-----------------+----------+
|  case6495rte.m           | 1          | 5               | 0.198    |
+--------------------------+------------+-----------------+----------+
|  case6515rte.m           | 1          | 4               | 0.169    |
+--------------------------+------------+-----------------+----------+
|  case69.m                | 1          | 3               | 0.012    |
+--------------------------+------------+-----------------+----------+
|  case6ww.m               | 1          | 3               | 0.011    |
+--------------------------+------------+-----------------+----------+
|  case85.m                | 1          | 3               | 0.013    |
+--------------------------+------------+-----------------+----------+
|  case89pegase.m          | 1          | 5               | 0.020    |
+--------------------------+------------+-----------------+----------+
|  case9.m                 | 1          | 3               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case9241pegase.m        | 1          | 6               | 0.487    |
+--------------------------+------------+-----------------+----------+
|  case9Q.m                | 1          | 3               | 0.013    |
+--------------------------+------------+-----------------+----------+
|  case9target.m           | 1          | 5               | 0.015    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg10k.m       | 1          | 4               | 0.257    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg200.m       | 1          | 2               | 0.014    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg2000.m      | 1          | 3               | 0.058    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg25k.m       | 1          | 7               | 1.118    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg500.m       | 1          | 3               | 0.027    |
+--------------------------+------------+-----------------+----------+
|  case_ACTIVSg70k.m       | 1          | 15              | 6.931    |
+--------------------------+------------+-----------------+----------+
|  case_RTS_GMLC.m         | 1          | 3               | 0.014    |
+--------------------------+------------+-----------------+----------+
|  case_SyntheticUSA.m     | 1          | 21              | 11.103   |
+--------------------------+------------+-----------------+----------+
|  case_ieee30.m           | 1          | 2               | 0.010    |
+--------------------------+------------+-----------------+----------+
|  case3375wp.m            | 0          | -               | 0.061    |
+--------------------------+------------+-----------------+----------+
|  case33bw.m              | 0          | -               | 0.007    |
+--------------------------+------------+-----------------+----------+
|  case3120sp.m            | 0          | -               | 0.037    |
+--------------------------+------------+-----------------+----------+
|  case3012wp.m            | 0          | -               | 0.082    |
+--------------------------+------------+-----------------+----------+
|  case3120sp.m            | 0          | -               | 0.039    |
+--------------------------+------------+-----------------+----------+
|  case3375wp.m            | 0          | -               | 0.059    |
+--------------------------+------------+-----------------+----------+
|  case33bw.m              | 0          | -               | 0.007    |
+--------------------------+------------+-----------------+----------+

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
