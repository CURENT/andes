.. _input-matpower:

MATPOWER
--------

ANDES supports MATPOWER data in version 2 in part for power flow calculation.
The following fields are supported:

- ``mpc.busMVA``
- ``mpc.bus``
- ``mpc.gen``
- ``mpc.branch``
- ``mpc.area``
- ``mpc.bus_name``

Other fields are not supported, most notably, ``mpc.gencost``.

Power flow calculation results for MATPOWER cases are typically identical to
that from MATPOWER using default settings. These settings include no reactive
power limits and following all generator connectivity status. Discrepencies in
the power flow solution between ANDES and MATPOWER are typically due to
configuration issues or different interpretation of the data, rather than in the
power flow models.

