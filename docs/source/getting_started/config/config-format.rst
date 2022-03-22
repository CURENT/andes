Format
------

The ANDES config file uses the format provided by Python module
:py:mod:`configparser`. The syntax is like the following:

.. code::

    [System]
    freq = 60
    mva = 100
    ...

    [PFlow]
    sparselib = klu
    linsolve = 0
    tol = 1e-6
    ...

    [TGOV1]
    allow_adjust = 1
    adjust_lower = 0
    adjust_upper = 1

In the above, ``System``, ``PFlow`` and ``TGOV1`` are two sections. ``freq =
60``, for example, is a pair of option and value in the ``[System]`` section.
Note the space before and after the equal sign.

The meaning of the fields in each section can be found in :ref:`configref`,
which contains the default values and acceptable values for each option. The
values for config fields can be a string or a number. Fields with acceptable
values being ``(0, 1)`` can only accept ``0`` or ``1`` to indicate true or
false. Non-binary values for such options will cause unexpected errors in the
program.

Limits in models
................

All models have three config options:

- ``allow_adjust``: allow limits of limiters in this model to be adjusted if the
  inputs, at steady state, is out of the limits. ``allow_adjust = 0`` is the
  global off-switch for this model.
- ``adjust_lower``: allow reducing the lower limit to the input value, if the
  input at steady-state is below the lower limit. This is disabled by default.
- ``adjust_upper``: allow increasing the upper limit to the steady-state input.
  This is enabled by default.

Note that setting ``allow_adjust = 0`` is equal to setting ``adjust_lower = 0``
and ``adjust_upper = 0``, but the former saves some time for function calls.

The limit adjustment feature is to alleviate issues caused by the model
parameters. Commercial tools have a more sophisticated mechanism for
autocorrection, and the limit adjustment is part of it. However, if you see an
limit adjustment warning or even initialization error, it is important not to
rely on autocorrection but fix the data by yourself. Autocorrected data can
yield some results but issues can remain hidden.

